""" Parse configuration file """

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from easydict import EasyDict as edict

from logger.custom_logger import setup_logging
from tools.utils import read_json, write_json


def read_cfg(cfg_file):
    """ Read the json configuration file """
    with open(cfg_file) as f:
        cfg = json.loads(f.read())
    cfg = edict(cfg)
    return cfg


def config_parser(config):
    """ Parse and return updated contents of the json configuration file """
    config = read_cfg(config)

    # set save_dir where trained model and log will be saved.
    result_dir = Path(config['result_dir'])
    exper_name = config['model']['used']

    run_id = config['run_id']
    if run_id is None:  # use timestamp as default run-id
        run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    _save_dir = result_dir / 'models' / exper_name / run_id
    _log_dir = result_dir / 'log' / exper_name / run_id
    config["log_dir"] = str(_log_dir)

    # make directory for saving checkpoints
    if os.path.exists(_save_dir):
        shutil.rmtree(_save_dir)
        logging.warning(f"Removed previous experiment in the same location {_save_dir}")
    _save_dir.mkdir(parents=True)
    config["save_dir"] = str(_save_dir)

    # make directory for saving logs
    if os.path.exists(_log_dir):
        shutil.rmtree(_log_dir)
        logging.warning(f"Removed previous logs in the same location {_log_dir}")
    _log_dir.mkdir(parents=True)

    # save updated config file to the checkpoint dir
    write_json(config, result_dir / 'config.json')

    # configure logging module
    setup_logging(_log_dir)
    return config
