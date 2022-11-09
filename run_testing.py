""" Run the testing logic """

import argparse

import torch
from tqdm import tqdm

import data_loaders.baseline_data_loaders as module_data
import models.baseline_models as module_arch
import training.training_tools.loss as module_loss
import training.training_tools.metric as module_metric
from tools.cfg_parser import ConfigParser


def main(cfg):
    logger = cfg.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, cfg['data_loader']['type'])(
        cfg['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = cfg.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, cfg['loss'])
    metric_fns = [getattr(module_metric, met) for met in cfg['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(cfg.resume))
    checkpoint = torch.load(cfg.resume)
    state_dict = checkpoint['state_dict']
    if cfg['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HOPE Generator - Test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
