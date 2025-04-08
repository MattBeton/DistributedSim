import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.fedavg_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config


def main():
    parser = arg_parse()

    parser.add_argument("--outer_interval", type=int, default=100)
    parser.add_argument('--outer_lr', type=float, default=1.0)
    parser.add_argument('--merge_method', type=str, default='mean')
    parser.add_argument('--sync_opt_state', action='store_true')

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = FedAvgGradient
    config.gradient_config.outer_interval = args.outer_interval
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.sync_opt_state = args.sync_opt_state
    config.gradient_config.merge_method = args.merge_method
    config.gradient_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()
