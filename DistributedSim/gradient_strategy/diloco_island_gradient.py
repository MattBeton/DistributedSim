import torch.distributed as dist
from copy import deepcopy
import random

import torch
from torch.nn import utils as nn_utils

from .gradient_strategy import GradientStrategy
from .communicate import *

class DiLoCoIslandGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0
        self.island_size = self.gradient_config.island_size

        device = next(model.parameters()).device
        self.master_model = deepcopy(model).to(device)
        for param in self.master_model.parameters():
            param.requires_grad = True

        self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                        **self.gradient_config.outer_optimizer_kwargs)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

    def _select_partners(self):
        world_size = dist.get_world_size()
        
        # Only rank 0 creates the island assignments
        islands = None
        if self.rank == 0:
            # Create a list of all rank numbers and shuffle it.
            ranks = list(range(world_size))
            ## TODO: Switch to pytorch shuffle.
            random.shuffle(ranks)
        else:
            ## TODO: Switch to pytorch broadcast.
            ranks = [None] * world_size

        dist.broadcast_object_list(ranks, src=0)

        islands = []
        for i in range(0, len(ranks), self.island_size):
            islands.append(set(ranks[i:i+self.island_size]))
        
        # Ugh seems so unoptimal but it's fine for now.
        my_island = None
        for island in islands:
            if self.rank in island:
                my_island = island
                break
        
        # print(f'Rank {self.rank} has partners {my_island}')
        
        return my_island

    def _average_models(self, island_members) -> None:
        ## Average model parameters across all members in the island
        for param in self.model.parameters():
            ## At the moment we are doing a full all_gather - this will be optimized in a full-scale training implementation.
            tensor_list = [torch.zeros_like(param.data) for _ in range(self.config.num_nodes)]
            all_gather(tensor_list, param.data)
            
            # Compute average only from ranks in the same island
            island_tensors = [tensor_list[rank] for rank in island_members]
            island_average = sum(island_tensors) / len(island_tensors)
            
            param.data = island_average

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data

    def _synchronize_master_model(self) -> None:
        # Save updated master model parameters to model.
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.gradient_config.diloco_interval == 0 and self.local_step > 0:
            if self.island_size < self.config.num_nodes:
                island_members = self._select_partners()
            else:
                island_members = list(range(self.config.num_nodes))

            self._average_models(island_members)

            self.outer_optimizer.zero_grad()
            self._set_master_grad()
            self.outer_optimizer.step()

            self._synchronize_master_model()

        super().step()

        self.local_step += 1