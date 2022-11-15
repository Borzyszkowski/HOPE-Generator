from agents.base_agent import BaseAgent
from tools.utils import set_random_seed, parse_args, read_cfg


def main():
    args = parse_args()
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
    set_random_seed()
    main()
