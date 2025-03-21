import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.diloco_island_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def main():
    parser = arg_parse()

    parser.add_argument("--diloco_interval", type=int, default=100)
    parser.add_argument('--outer_lr', type=float, default=0.7)
    parser.add_argument("--nesterov", type=bool, default=True)
    parser.add_argument("--outer_momentum", type=float, default=0.9)
    parser.add_argument("--island_size", type=int, default=None)

    args = parser.parse_args()

    if args.island_size is None:
        args.island_size = args.num_nodes

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = DiLoCoIslandGradient
    config.gradient_config.diloco_interval = args.diloco_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
        'nesterov': args.nesterov,
        'momentum': args.outer_momentum,
    }
    config.gradient_config.island_size = args.island_size

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()