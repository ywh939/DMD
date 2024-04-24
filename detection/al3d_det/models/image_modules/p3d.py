import torch
import torch.nn as nn
from functools import partial
from al3d_det.utils.attention_utils import simam_module

def conv_S(in_planes,out_planes,stride=1,padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(in_planes,out_planes,kernel_size=(1,3,3),stride=1,
                     padding=padding,bias=False)

def conv_T(in_planes,out_planes,stride=1,padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(in_planes,out_planes,kernel_size=(3,1,1),stride=1,
                     padding=padding,bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,n_s=0,depth_3d=47,ST_struc=('A','B','C')):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.depth_3d=depth_3d
        self.ST_struc=ST_struc
        self.len_ST=len(self.ST_struc)

        stride_p=stride
        # if not self.downsample ==None:
        #     stride_p=(1,2,2)
        if n_s<self.depth_3d:
            if n_s==0:
                stride_p=1
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False,stride=stride_p)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            if n_s==self.depth_3d:
                stride_p=2
            else:
                stride_p=1
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.id=n_s
        self.ST=list(self.ST_struc)[self.id%self.len_ST]
        if self.id<self.depth_3d:
            self.conv2 = conv_S(planes,planes, stride=1,padding=(0,1,1))
            self.bn2 = nn.BatchNorm3d(planes)
            #
            self.conv3 = conv_T(planes,planes, stride=1,padding=(1,0,0))
            self.bn3 = nn.BatchNorm3d(planes)
        else:
            self.conv_normal = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1,bias=False)
            self.bn_normal = nn.BatchNorm2d(planes)

        if n_s<self.depth_3d:
            self.conv4 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm3d(planes * 4)
        else:
            self.conv4 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        
    def ST_A(self,x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x

    def ST_B(self,x):
        tmp_x = self.conv2(x)
        tmp_x = self.bn2(tmp_x)
        tmp_x = self.relu(tmp_x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x+tmp_x

    def ST_C(self,x):
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        tmp_x = self.conv3(x)
        tmp_x = self.bn3(tmp_x)
        tmp_x = self.relu(tmp_x)

        return x+tmp_x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.id<self.depth_3d: # C3D parts: 

            if self.ST=='A':
                out=self.ST_A(out)
            elif self.ST=='B':
                out=self.ST_B(out)
            elif self.ST=='C':
                out=self.ST_C(out)
        else:
            out = self.conv_normal(out)   # normal is res5 part, C2D all.
            out = self.bn_normal(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class P3D(nn.Module):
    def __init__(self, layers, modality='RGB', dropout=0.5,ST_struc=('A','B','C')):
        
        self.inplanes = 64
        super(P3D, self).__init__()

        self.input_channel = 3 if modality=='RGB' else 2  # 2 is for flow 
        self.ST_struc=ST_struc

        self.conv1_custom = nn.Conv3d(self.input_channel, 64, kernel_size=(1,7,7), stride=(1,2,2),
                                padding=(0,3,3), bias=False)

        self.depth_3d=sum(layers[:3])# C3D layers are only (res2,res3,res4),  res5 is C2D

        self.bn1 = nn.BatchNorm3d(64) # bn1 is followed by conv1
        self.cnt=0
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 16, layers[1], stride=(1,2,2))
        self.layer3 = self._make_layer(Bottleneck, 8, layers[2])
        self.layer4 = self._make_layer(Bottleneck, 16, layers[3])

        self.ConvSim = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=1, bias=False),
            simam_module(),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        stride_p=stride #especially for downsample branch.

        if self.cnt<self.depth_3d:
            # if self.cnt==0:
            #     stride_p=1
            # else:
            #     stride_p=(1,2,2)

            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride_p, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        else:
            # if stride != 1 or self.inplanes != planes * block.expansion:
            #     downsample = nn.Sequential(
            #         nn.Conv2d(self.inplanes, planes * block.expansion,
            #                     kernel_size=1, stride=2, bias=False),
            #         nn.BatchNorm2d(planes * block.expansion)
            #     )
            self.inplanes = planes * block.expansion
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
        self.cnt+=1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,n_s=self.cnt,depth_3d=self.depth_3d,ST_struc=self.ST_struc))
            self.cnt+=1

        return nn.Sequential(*layers)

    def forward(self, x):#[2, 3, 2, 600, 960], return [2, 256, 150, 240]

        x = self.conv1_custom(x)#[2, 64, 2, 300, 480]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)   #[2, 256, 2, 300, 480]
        x = self.layer2(x)   #[2, 64, 2, 150, 240]
        x = self.layer3(x)  #[2, 32, 2, 150, 240]

        sizes=x.size()
        x = x.view(sizes[0],-1,sizes[3],sizes[4])  #[2, 64, 150, 240]
        x = self.layer4(x) # [2, 64, 150, 240]

        # x = self.ConvSim(x) # [2, 64, 150, 240]
 
        return x
    
if __name__ == '__main__':
    model = P3D([3, 8, 36, 3])
    model = model.cuda()
    data=torch.autograd.Variable(torch.rand(1,3,2,160,160)).cuda()
    out=model(data)
