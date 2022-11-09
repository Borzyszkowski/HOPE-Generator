""" Run the training logic """

import argparse
import collections
from datetime import datetime

import torch

import data_loaders.baseline_data_loaders as module_data
import models.baseline_models as module_arch
import training.training_tools.losses as module_loss
import training.training_tools.metrics as module_metric
from tools.cfg_parser import ConfigParser
from tools.utils import prepare_device, set_random_seed
from training.trainer import Trainer


def run_training(cfg):
    """ Prepares and starts the training """
    logger = cfg.get_logger('train')
    logger.info(f'[{cfg.run_id}] - HOPE Generator has started!')
    logger.info(f'log dir: {cfg.log_dir}')
    logger.info(f'checkpoint dir: {cfg.save_dir}')
    logger.info(f'Torch Version: {torch.__version__}')

    # setup data_loader instances
    data_loader = cfg.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = cfg.init_obj('arch', module_arch)
    logger.info(f"Selected model name: {cfg.config['arch']['type']}")

    # build optimizer and learning rate scheduler
    params = [var[1] for var in model.named_parameters()]
    params_number = sum(p.numel() for p in params if p.requires_grad)
    optimizer = cfg.init_obj('optimizer', torch.optim, params)
    lr_scheduler = cfg.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    logger.info('Total trainable parameters of the model: %2.2f M.' % (params_number * 1e-6))

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"Using {device_ids} GPUs for training")

    # get function handles of loss and metrics
    criterion = getattr(module_loss, cfg['loss'])
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]

    # run the training loop
    start_time = datetime.now().replace(microsecond=0)
    logger.info(f"Started training at {start_time} for {cfg.config['trainer']['epochs']} epochs\n")
    trainer = Trainer(model, criterion, metrics, optimizer,
                      cfg=cfg,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()

    # finish the training
    end_time = datetime.now().replace(microsecond=0)
    logger.info(f'Finished Training at {end_time}')
    logger.info(f'Training done in {(end_time - start_time)}!')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HOPE Generator')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=True, type=str, help='sets a random seed for reproducibility')
    args.add_argument('-id', '--run_id', default="V00", type=str, help='experiment ID')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [CustomArgs(['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
               CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')]
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()

    # set random seed for reproducibility
    if args.seed is True:
        set_random_seed()

    # start the main training logic
    run_training(config)
