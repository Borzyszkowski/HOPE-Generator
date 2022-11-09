""" Run the training logic """

import argparse
import collections

import torch

import data_loaders.baseline_data_loaders as module_data
import models.baseline_models as module_arch
import training.training_tools.losses as module_loss
import training.training_tools.metrics as module_metric
from tools.cfg_parser import ConfigParser
from tools.utils import prepare_device, set_random_seed
from training.trainer import Trainer


def run_training(cfg):
    logger = cfg.get_logger('train')

    # setup data_loader instances
    data_loader = cfg.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = cfg.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, cfg['loss'])
    metrics = [getattr(module_metric, met) for met in cfg['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = cfg.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = cfg.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # run the training loop
    trainer = Trainer(model, criterion, metrics, optimizer,
                      cfg=cfg,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HOPE Generator')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--seed', default=False, type=str, help='sets a random seed for reproducibility')
    args.add_argument('-id', '--run_id', default='VOO', type=str, help='experiment ID')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
               CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')]
    config = ConfigParser.from_args(args, options)

    # TODO: fix this
    # fix random seeds for reproducibility
    # if config.seed is True:
    #     set_random_seed()

    # start the main training logic
    run_training(config)
