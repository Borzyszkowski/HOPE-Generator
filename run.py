import os
import sys
import json
import time
import argparse
from easydict import EasyDict

# import models from agents
from agents.base_agent import BaseAgent


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                         default="", help="model to train")
    parser.add_argument('--mode', type=str,
                         default="train", help="training or testing")
    parser.add_argument('--config', type=str,
                         default="configs/config.json", 
                         help="path to the config file")

    args = parser.parse_args()
    return args


def read_cfg(cfg_file):
    with open(cfg_file) as f:
      cfg = json.loads(f.read())
    cfg = EasyDict(cfg)
    return cfg


def main():
    args = parse_arguments()
    cfg = read_cfg(args.config)
    test = args.mode == "test"
    model = args.model
    # pattern matching to find the model
    if model == "BaseAgent":
       agent = BaseAgent(cfg, test)

    if test:
        agent.test()
    elif args.mode == 'train':
        agent.train()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
