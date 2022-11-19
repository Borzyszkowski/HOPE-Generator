""" Run the main project logic """

import argparse

from agents.base_agent import BaseAgent
from tools.utils import set_random_seed, read_cfg


def main(args):
    """ Run the main project logic """

    # set random seed for reproducibility
    if args.seed is True:
        set_random_seed()

    cfg = read_cfg(args.config)
    test = args.mode == "test"
    model = args.model

    # pattern matching to find the model
    if model == "BaseAgent":
        agent = BaseAgent(cfg, test)

    # start the main training or testing logic
    if test:
        agent.test()
    elif args.mode == 'train':
        agent.train()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='HOPE Generator')
    args.add_argument('--seed', default=True, type=str, help='sets a random seed for reproducibility')
    args.add_argument('--run_id', default="V00", type=str, help='experiment ID')
    args.add_argument('--model', type=str, default="BaseAgent", help="model to train")
    args.add_argument('--mode', type=str, default="train", help="training or testing")
    args.add_argument('--config', type=str, default="configs/config.json", help="path to the config file")
    args = args.parse_args()
    main(args)
