
from eval_utils import eval_one_epoch
    
def get_eval_datas(cfg, args, is_dist, logger, output_dir):
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # convert to be 'train' datase
    # for evaluate train datasets
    import copy
    train_eval_cfg = copy.deepcopy(cfg)
    train_eval_cfg.DATA_CONFIG.DATA_SPLIT['test'] = 'train'
    train_eval_cfg.DATA_CONFIG.INFO_PATH['test'] = train_eval_cfg.DATA_CONFIG.INFO_PATH['train']
    train_dataloader_eval = get_eval_dataloader(train_eval_cfg.DATA_CONFIG, train_eval_cfg.CLASS_NAMES, args.batch_size, is_dist, args.workers, logger)
    
    # evaluate value datasets
    val_eval_cfg = copy.deepcopy(cfg)
    val_dataloader_eval = get_eval_dataloader(val_eval_cfg.DATA_CONFIG, val_eval_cfg.CLASS_NAMES, args.batch_size, is_dist, args.workers, logger)
    
    return eval_output_dir, train_eval_cfg, val_eval_cfg, train_dataloader_eval, val_dataloader_eval

def eval_train_with_val_per_epoch(train_eval_cfg, val_eval_cfg, model, train_dataloader_eval, val_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file, tb_log):
    train_eval_ret_dict = eval_with_tensorboard(train_eval_cfg, model, train_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file)
    val_eval_ret_dict = eval_with_tensorboard(val_eval_cfg, model, val_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file)
    
    logger.info("train_eval_ret_dict: ---------------------------------")
    logger.error(train_eval_ret_dict)
    logger.info("val_eval_ret_dict: ---------------------------------")
    logger.error(val_eval_ret_dict)
    
    scalars = ['Car_3d', 'Pedestrian_3d', 'Cyclist_3d']
    [log_tensorboard(scalar, train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id) for scalar in scalars]
    
def eval_train_with_val_per_epoch2(train_eval_cfg, val_eval_cfg, model, train_dataloader_eval, val_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file, tb_log):
    train_eval_ret_dict = eval_with_tensorboard(train_eval_cfg, model, train_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file)
    val_eval_ret_dict = eval_with_tensorboard(val_eval_cfg, model, val_dataloader_eval, epoch_id, logger, is_dist, eval_output_dir, save_to_file)
    logger.error("#######################################")
    logger.error(train_eval_ret_dict)
    logger.error(val_eval_ret_dict)
    logger.error("#######################################")
    return train_eval_ret_dict, val_eval_ret_dict

def log_2_tensorboard(train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id):
    scalars = ['Car_3d', 'Pedestrian_3d', 'Cyclist_3d']
    [log_tensorboard(scalar, train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id) for scalar in scalars]
          
def test_eval_train_with_val(model, epoch_id, cfg, args, is_dist, tb_log):
    # convert to be 'train' datase
    # for evaluate train datasets
    import copy
    data_config = copy.deepcopy(cfg.DATA_CONFIG)
    data_config.DATA_SPLIT['test'] = 'train'
    data_config.INFO_PATH['test'] = data_config.INFO_PATH['train']
    train_eval_ret_dict = eval_model(cfg, args, epoch_id, data_config, is_dist, model)
    
    # evaluate value datasets
    data_config = cfg.DATA_CONFIG
    val_eval_ret_dict = eval_model(cfg, args, epoch_id, data_config, is_dist, model)
    
    scalars = ['Car_3d', 'Pedestrian_3d', 'Cyclist_3d']
    [log_tensorboard(scalar, train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id) for scalar in scalars]
    # log_tensorboard('Car_3d', train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id)
    # log_tensorboard('Pedestrian_3d', train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id)
    # log_tensorboard('Cyclist_3d', train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id)
    
def eval_model(cfg, args, epoch_id, data_config, is_dist, model):
    eval_output_dir = get_eval_output_dir(cfg, args, epoch_id, data_config.DATA_SPLIT)
    logger = get_eval_logger(cfg, eval_output_dir)
    data_loader = get_eval_dataloader(data_config, cfg.CLASS_NAMES, args.batch_size, is_dist, args.workers, logger)
    return eval_with_tensorboard(cfg, model, data_loader, epoch_id, logger, is_dist, eval_output_dir, args.save_to_file)

def get_eval_output_dir(cfg, args, epoch_id, data_split):
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / data_split['test']
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    return eval_output_dir

def get_eval_logger(cfg, eval_output_dir):
    from datetime import datetime as my_datetime
    from al3d_utils.common_utils import create_logger
    
    log_file = eval_output_dir / ('log_eval_%s.txt' % my_datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    return create_logger(log_file, rank=cfg.LOCAL_RANK)

def get_eval_dataloader(data_config, class_names, batch_size, is_dist, workers, logger):
    from al3d_det.datasets import build_dataloader
    _, data_loader, _ = build_dataloader(
        dataset_cfg=data_config,
        class_names=class_names,
        batch_size=batch_size,
        dist=is_dist, workers=workers, logger=logger, training=False
    )
    return data_loader

def eval_with_tensorboard(cfg, model, dataloader, epoch_id, logger, is_dist, eval_output_dir, save_to_file):
    eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    if save_to_file:
        eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    ret_dict = eval_one_epoch(
        cfg, model, dataloader, epoch_id, logger, dist_test=is_dist,
        result_dir=eval_output_dir, save_to_file=save_to_file
    )
    
    logger.error('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    logger.error(ret_dict)
    logger.error('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    return ret_dict

def log_train_and_val_in_tensorboard(train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id):
    scalars = ['Car_3d']
    # scalars = ['Car_3d', 'Pedestrian_3d', 'Cyclist_3d']
    [log_tensorboard(scalar, train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id) for scalar in scalars]
    
    
def log_tensorboard(class_name, train_eval_ret_dict, val_eval_ret_dict, tb_log, epoch_id):
    easy_scalar = {'train_easy': train_eval_ret_dict[class_name + '/easy_R40'], 'val_easy': val_eval_ret_dict[class_name + '/easy_R40']}
    moderate_scalar = {'train_moderate': train_eval_ret_dict[class_name + '/moderate_R40'], 'val_moderate': val_eval_ret_dict[class_name + '/moderate_R40']}
    hard_scalar = {'train_hard': train_eval_ret_dict[class_name + '/hard_R40'], 'val_hard': val_eval_ret_dict[class_name + '/hard_R40']}
    
    tb_log.add_scalars(class_name + '/R40_easy', easy_scalar, epoch_id)
    # tb_log.add_scalars(class_name + '/R40_moderate', moderate_scalar, epoch_id)
    # tb_log.add_scalars(class_name + '/R40_hard', hard_scalar, epoch_id)