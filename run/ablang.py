import torch

import argparse
import numpy as np

from DistributedSim.sim_builder import *
from DistributedSim.sim_config import *
from DistributedSim.gradient_strategy.gradient_strategy import *

from DistributedSim.dataset.dataset import DatasetConfig
from DistributedSim.models.ablang.ablang_2.ablang import AbLangConfig, AblangWrapper
from DistributedSim.dataset.ablang.datamodule import AbDataModule
from DistributedSim.models.ablang.ablang_2.tokenizers import ABtokenizer


def get_dataset(
    d_config: DatasetConfig,
    device: torch.device,
    start_pc: float = 0.0,
    end_pc: float = 1.0,
):
    """Hacky way to get processed antibody dataset by wrapping existing dataloading methods."""
    tokenizer = ABtokenizer
    data_module = AbDataModule(data_hparams=d_config, tokenizer_class=tokenizer)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    inputs = []
    labels = []
    for d in train_dataloader:
        inputs.append(d['input'])
        labels.append(d['labels'])
        print("input shape: ", d['input'].shape)
        print("labels shape: ", d['labels'].shape)
        print("sequences shape: ", len(d['sequences']))
        print("input: ", d['input'][0])
        print("labels: ", d['labels'][0])
        print("sequences: ", d['sequences'][0])
        break
    return train_dataloader.dataset


def gen_wandb_name(args):
    name = f"bs{args.batch_size}_lr{args.lr:.0e}_warm{args.warmup_steps}_max{args.max_steps}"
    return name


def arg_parse():
    # Command line arguments
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    parser.add_argument("--start_pc", type=float, default=0.0)
    parser.add_argument("--end_pc", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--device_type", type=str, default="")
    parser.add_argument("--devices", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "base", "medium", "large", "xl"],
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--cosine_anneal", action="store_true")
    parser.add_argument("--autocast", action="store_true")

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--dataset_proportion", type=float, default=1.0)
    parser.add_argument("--val_proportion", type=float, default=0.1)
    parser.add_argument("--correlation_interval", type=int, default=None)
    parser.add_argument("--over_sample_data", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--out_path", type=str, default="./outputs")
    parser.add_argument("--eval_path", type=str, default="./evals")
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Number of cpus to use on data handling (4xGPUs is the recommended). \
                                                                    0 uses the main process to load the data.",
    )

    return parser


def gen_model_config(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_config = AbLangConfig(
        vocab_size=100,
        hidden_embed_size=1024,
        n_attn_heads=16,
        n_encoder_blocks=12,
        padding_tkn=0,
        mask_tkn=1,
        start_tkn=2,
    )

    return model_config


def config_gen(args, model_config):
    config = SimConfig(
        model_class=AblangWrapper,
        model_config=model_config,
        num_epochs=args.epochs,
        num_nodes=args.num_nodes,
        device_type=args.device_type,
        devices=args.devices,
        autocast=args.autocast,
        minibatch_size=args.minibatch_size if args.minibatch_size else args.batch_size,
        block_size=args.block_size,
        val_size=args.val_size,
        save_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        correlation_interval=args.correlation_interval,
        gradient_config=GradientConfig(
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": args.lr,
            },
            max_norm=args.max_norm,
            lr_scheduler="lambda_cosine",
            warmup_steps=args.warmup_steps,
            cosine_anneal=args.cosine_anneal,
            max_local_steps=args.max_steps,
        ),
        dataset_config=DatasetConfig(
            dataset_name="ablang",
            dataset_load_fn=get_dataset,
            batch_size=args.batch_size,
            data_path=args.data_path,
            out_path=args.out_path,
            eval_path=args.eval_path,
            device=args.device_type,
            block_size=args.block_size,
            dataset_proportion=args.dataset_proportion,
            val_proportion=args.val_proportion,
            over_sample_data=args.over_sample_data,
            model_config=model_config,
            train_batch_size=args.batch_size,
            cpus=args.cpus,
        ),
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name if args.wandb_name else gen_wandb_name(args),
    )

    return config
