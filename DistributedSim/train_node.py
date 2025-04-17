import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader

import os
import copy

from .sim_config import *
from .gradient_strategy.gradient_strategy import *
from .wandb_logger import *
from .gradient_strategy.communicate import *

from .dataset.dataset import get_dataset


import curvlinops



class TrainNode:
    '''
    Single node of distributed training process. Should be the same regardless of rank topology/architecture.
    '''
    def __init__(self, 
                 config: SimConfig,
                 device: torch.device,
                 rank: int):
        self.config = config
        self.device = device
        self.rank = rank

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.get_datasets()

        self.config.gpt_config.vocab_size = self.vocab_size
        
        self.model = self.config.model_class(self.config.gpt_config).to(self.device)

        # Remove weight tying as this will break the parameter-to-layer detection
        self.model.transformer.wte.weight = nn.Parameter(
            data=self.model.transformer.wte.weight.data.detach().clone()
        )

        self.model.transformer.wte.weight.requires_grad = False
        self.model.transformer.wpe.weight.requires_grad = False

        #self.model.lm_head.weight.requires_grad = False

        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = False
    
        print(f"model parameter count: ", self.model.get_num_params() / 1e6)

        ## Ensure all process models share the same params
        if self.config.num_nodes > 1:
            for _, param in self.model.named_parameters():
                broadcast(param.data, src=0)

        self.local_step = 0
        self.max_steps = len(self.train_dataloader) * self.config.num_epochs
        if self.config.gradient_config.max_local_steps:
            self.max_steps = min(self.max_steps, 
                                 self.config.gradient_config.max_local_steps)

            self.config.gradient_config.max_local_steps = self.max_steps

        if self.rank == 0:
            self.logger = WandbLogger(rank=self.rank, 
                                      device=self.device, 
                                      config=self.config, 
                                      model=self.model, 
                                      max_steps=self.max_steps)

        self.gradient_strategy = self.config.gradient_class(self.rank, 
                                                            self.model, 
                                                            self.config,
                                                            self.logger if self.rank == 0 else None,
                                                            curv_dataloader=self.curv_dataloader)

        self.epoch = 0
        
    
    def get_datasets(self):
        ## Import Datasets
        dataset_id = self.config.dataset_name.split('_')[0]

        train_start = (1 - self.config.val_proportion) * self.rank / self.config.num_nodes
        train_end = (1 - self.config.val_proportion) * (self.rank + 1) / self.config.num_nodes
        val_start = (1 - self.config.val_proportion)
        val_end = 1.0

        if self.config.same_curv_data: # with this setting, the curvature always uses the split of rank=0
            curv_start = (1 - self.config.val_proportion) * 0 / self.config.num_nodes
            curv_end = (1 - self.config.val_proportion) * (0 + 1) / self.config.num_nodes
        else:
            curv_start = (1 - self.config.val_proportion) * self.rank / self.config.num_nodes
            curv_end = (1 - self.config.val_proportion) * (self.rank + 1) / self.config.num_nodes

        self.train_dataset, self.vocab_size = get_dataset(dataset_id,
                                             train_start * self.config.dataset_proportion,
                                             train_end * self.config.dataset_proportion,
                                             block_size=self.config.block_size,
                                             char=self.config.char_dataset)

        self.val_dataset, self.vocab_size = get_dataset(dataset_id,
                                             val_start,
                                             val_end,
                                             block_size=self.config.block_size,
                                             char=self.config.char_dataset)

        self.curv_dataset, self.vocab_size = get_dataset(dataset_id,
                                             curv_start * self.config.dataset_proportion,
                                             curv_end * self.config.dataset_proportion,
                                             block_size=self.config.block_size,
                                             char=self.config.char_dataset,
                                             just_one_chunk=True)


        def collate_fn_flatten_target(batch):
            inputs, targets = zip(*batch)
            inputs = torch.stack(inputs)  # Standard batching
            targets = torch.stack(targets).view(-1)  # Flatten the target across batch
            return inputs, targets

        ## Build Dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=True)

        self.val_dataloader = DataLoader(self.val_dataset, 
                          batch_size=self.config.batch_size,
                          shuffle=False)

        self.curv_dataloader = DataLoader(self.curv_dataset,
                          batch_size=8, #self.config.batch_size,
                          shuffle=False,
                          collate_fn=collate_fn_flatten_target)

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)
        self.curv_data_iter = iter(self.curv_dataloader)

    def _save_checkpoint(self):
        save_path = os.path.join(self.config.save_dir, 
                                 self.config.wandb_project, 
                                 self.config.wandb_run_name if self.config.wandb_run_name else 'unnamed',
                                 str(self.rank))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        filename = f"{self.local_step}.pt"
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))

    def _get_batch(self, eval=False):
        if eval:
            try:
                x, y = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                x, y = next(self.val_data_iter)
        else:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)

        x, y = x.to(self.device), y.to(self.device)

        return x, y

    def _train_step(self):
        x, y = self._get_batch()
        self.gradient_strategy.zero_grad()
        
        minibatch_size = self.config.local_minibatch_size if self.config.local_minibatch_size else self.config.batch_size

        for i in range(0, len(x), minibatch_size):
            x_batch = x[i:i+minibatch_size]
            y_batch = y[i:i+minibatch_size]

            if self.config.autocast:
                with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                    _, loss = self.model(x_batch, y_batch)
            else:
                _, loss = self.model(x_batch, y_batch)

            loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad /= (len(x) / minibatch_size)

        loss = loss.detach()
        torch.cuda.empty_cache()
        
        self.gradient_strategy.step()

        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(loss) for _ in range(world_size)] if dist.get_rank() == 0 else None
        dist.gather(loss.detach(), gather_list=gathered, dst=0)
        if self.rank == 0:
            for rank, train_loss in enumerate(gathered):
                self.logger.log_pure(loss=train_loss.cpu().item(), name=f"train_loss_{rank}")

        if self.config.checkpoint_interval and self.local_step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()


    def _evaluate_loss(self):
        self.model.eval()
        
        loss_total = 0

        with torch.no_grad():
            num_evals = int(self.config.val_size / self.config.batch_size)
            for val_i, (x, y) in enumerate(self.val_dataloader): # Deterministic on purpose: always same first batches!
                if val_i >= num_evals:
                    break

                minibatch_size = self.config.local_minibatch_size if self.config.local_minibatch_size else self.config.batch_size
                for i in range(0, len(x), minibatch_size):
                    x_batch = x[i:i+minibatch_size].to(self.device)
                    y_batch = y[i:i+minibatch_size].to(self.device)

                    if self.config.autocast:
                        with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                            _, loss = self.model(x_batch, y_batch)
                    else:
                        _, loss = self.model(x_batch, y_batch)

                    loss_total += loss.item() / (self.config.batch_size // minibatch_size)

            loss_total /= num_evals

        return loss_total


    def _log(self, name, number):
        world_size = dist.get_world_size()
        gathered = [torch.zeros(1, device=self.device) for _ in range(world_size)] if dist.get_rank() == 0 else None
        dist.gather(torch.ones(1, device=self.device) * number , gather_list=gathered, dst=0)
        if self.rank == 0:
            for rank, value in enumerate(gathered):
                self.logger.log_pure(loss=value.cpu().item(), name=f'{name}_{rank}')


    def train(self):
        while self.local_step < self.max_steps:
            do_eval = self.local_step % self.config.eval_interval == 0
            if do_eval:
                eval_loss = self._evaluate_loss()
                self._log('eval_loss', eval_loss)

            self._train_step()

            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            dist.barrier()

            if do_eval:
                eval_loss_after_merge = self._evaluate_loss()
                merge_gain = eval_loss_after_merge - eval_loss
                self._log('merge_gain', merge_gain)

        eval_loss = self._evaluate_loss()
        self._log('eval_loss', eval_loss)
