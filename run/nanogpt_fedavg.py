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
    parser.add_argument('--damping_meantrick', action='store_true')
    parser.add_argument('--forward_only', action='store_true')
    parser.add_argument('--damping', type=float, default=0.01)
    parser.add_argument('--skip_embed_curv', action='store_true')

    parser.add_argument('--skip_mlp_fc', action='store_true')
    parser.add_argument('--skip_mlp_proj', action='store_true')
    parser.add_argument('--skip_attn_attn', action='store_true')
    parser.add_argument('--skip_attn_proj', action='store_true')
    parser.add_argument("--outer_warmup", type=int, default=1000)

    args = parser.parse_args()

    gpt_config = gen_gpt_config(args)

    config = config_gen(args, gpt_config)

    config.gradient_class = FedAvgGradient
    config.gradient_config.outer_interval = args.outer_interval
    config.gradient_config.outer_warmup = args.outer_warmup
    config.gradient_config.outer_optimizer_cls = torch.optim.SGD
    config.gradient_config.sync_opt_state = args.sync_opt_state
    config.gradient_config.merge_method = args.merge_method
    config.gradient_config.damping = args.damping
    config.gradient_config.damping_meantrick = args.damping_meantrick
    config.gradient_config.forward_only = args.forward_only
    config.gradient_config.skip_embed_curv = args.skip_embed_curv
    config.gradient_config.skip_mlp_fc = args.skip_mlp_fc
    config.gradient_config.skip_mlp_proj = args.skip_mlp_proj
    config.gradient_config.skip_attn_attn = args.skip_attn_attn
    config.gradient_config.skip_attn_proj = args.skip_attn_proj

    config.gradient_config.outer_optimizer_kwargs = {
        'lr': args.outer_lr,
    }

    simbuilder = LocalSimBuilder(config)

    simbuilder.execute()

if __name__ == "__main__":
    main()
