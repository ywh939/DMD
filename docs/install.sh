#!/bin/bash

cd detection/al3d_det/models/image_modules/swin_model/ && python setup.py develop && cd ../../../../../utils && python setup.py develop && cd ../detection && python setup.py develop && cd al3d_det/models/ops  && python setup.py develop
