import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *
from DistributedSim.gradient_strategy.demo_gradient import *

from ablang import arg_parse, config_gen, gen_model_config


def main():
    parser = arg_parse()
    args = parser.parse_args()

    model_config = gen_model_config(args)

    config = config_gen(args, model_config)

    config.gradient_class = SimpleReduceGradient

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()


if __name__ == "__main__":
    main()