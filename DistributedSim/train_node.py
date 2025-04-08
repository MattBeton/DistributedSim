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
                                             train_start * self.config.dataset_proportion,
                                             train_end * self.config.dataset_proportion,
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
                          shuffle=True)

        self.curv_dataloader = DataLoader(self.curv_dataset,
                          batch_size=self.config.batch_size,
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
        if not eval or self.val_data_iter is None:
            try:
                x, y = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                x, y = next(self.train_data_iter)
        else:
            try:
                x, y = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                x, y = next(self.val_data_iter)

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
        
        self.gradient_strategy.step()

        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        if self.config.checkpoint_interval and self.local_step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

    def _evaluate(self):
        model_clone = self.config.model_class(self.config.gpt_config).to(self.device)
        model_clone.load_state_dict(copy.deepcopy(self.model.state_dict()))

        for name, param in model_clone.named_parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            this_model = self.model

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            this_model = model_clone

        

        if self.rank == 0 or self.rank == 1:
            this_model.eval()
            
            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.config.val_size / self.config.batch_size)):
                    x, y = self._get_batch(eval=True)

                    minibatch_size = self.config.local_minibatch_size if self.config.local_minibatch_size else self.config.batch_size
                    for i in range(0, len(x), minibatch_size):
                        x_batch = x[i:i+minibatch_size]
                        y_batch = y[i:i+minibatch_size]

                        if self.config.autocast:
                            with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                                _, loss = this_model(x_batch, y_batch)
                        else:
                            _, loss = this_model(x_batch, y_batch)

                        loss_total += loss.item() / (self.config.batch_size // minibatch_size)

        # Rank 0 logs the local evaluation.
        if self.rank == 0:
            # print(f"LOCAL: Eval Loss: {loss_total / int(self.config.val_size / self.config.batch_size):.4f}, "
            #         f"Eval Perplexity: {math.exp(loss_total / int(self.config.val_size / self.config.batch_size)):.4f}")
            self.logger.log_pure(loss=loss_total / int(self.config.val_size / self.config.batch_size), 
                                    name='val_local')

        # Broadcast the global loss from rank 1 to all ranks.
        if self.config.num_nodes > 1:
            # All ranks create a dummy tensor to participate.
            global_loss_tensor = torch.empty(1, device=next(self.model.parameters()).device)
            if self.rank == 1:
                global_loss_tensor[0] = loss_total / int(self.config.val_size / self.config.batch_size)
            broadcast(global_loss_tensor, src=1)

            # Only rank 0 logs the global evaluation.
            if self.rank == 0:
                global_loss = global_loss_tensor.item()
                # print(f"GLOBAL: Eval Loss: {global_loss:.4f}, Eval Perplexity: {math.exp(global_loss):.4f}")
                self.logger.log_pure(loss=global_loss, name='global')

        del model_clone

    def train(self):
        while self.local_step < self.max_steps:
            if self.local_step % self.config.eval_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            # if self.local_step == 5:
            #     break

            dist.barrier()


        self._evaluate()
