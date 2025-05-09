import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.sparta_gradient import *

from DistributedSim.models.nanogpt import GPT, GPTConfig
from DistributedSim.dataset.build_dataset import *

from nanogpt import arg_parse, config_gen, gen_gpt_config

def gen_wandb_name(args):
    name = f"p{args.p_sparta}_n{args.num_nodes}_lr{args.learning_rate:.0e}"
    return name

def main():
    parser = arg_parse()

    parser.add_argument("--p_sparta", type=float, default=0.005)
    parser.add_argument("--async_sparta_delay", type=int, default=0)
    parser.add_argument("--schedule_p", action='store_true')
    parser.add_argument("--p_min_factor", type=float, default=0.1)
    parser.add_argument("--fault_rate", type=float, default=0.0)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = SPARTAGradient
    config.gradient_config.p_sparta = args.p_sparta
    config.gradient_config.async_sparta_delay = args.async_sparta_delay
    config.gradient_config.schedule_p = args.schedule_p
    config.gradient_config.p_min_factor = args.p_min_factor
    config.gradient_config.fault_rate = args.fault_rate

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()