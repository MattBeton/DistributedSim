import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from .gradient_strategy import GradientStrategy
from .communicate import *

class FedAvgGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True

            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                            **self.gradient_config.outer_optimizer_kwargs)

        print(self.gradient_config.optimizer_kwargs)
        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

    def _reduce_models(self) -> None:
        for param in self.model.parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes
            # reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
            # if self.rank == 0:
            #     param.data /= self.config.num_nodes


    def _sync_opt_state(self) -> None:
        for param in self.model.parameters():
            if param.requires_grad:
                #if 'exp_avg' in self.optim.state[param].keys():
                all_reduce(self.optim.state[param]['exp_avg'], op=dist.ReduceOp.SUM)
                self.optim.state[param]['exp_avg'] /= self.config.num_nodes

            if param.requires_grad:
                #if 'exp_avg_sq' in self.optim.state[param].keys():
                all_reduce(self.optim.state[param]['exp_avg_sq'], op=dist.ReduceOp.SUM)
                self.optim.state[param][ 'exp_avg_sq'] /= self.config.num_nodes


    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - self.model.state_dict()[name].data.to('cpu')

    def _synchronize_master_model(self) -> None:
        for name, param in self.model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)

    def step(self):
        if self.gradient_config.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_config.max_norm)

        # We have just calculated the loss and done the backward pass. 
        # Therefore we do inner step first.
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad.mean())

        self.optim.step()

        # Outer step if needed.
        if self.local_step % self.gradient_config.outer_interval == 0 and self.local_step > 0:
            if self.gradient_config.merge_method == 'mean':
                self._reduce_models()
            else:
                raise NotImplementedError(f"Merge method unknown: {merge_method}")

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

            #if self.gradient_config.sync_opt_state:
            #    self._sync_opt_state()


        self.local_step += 1
