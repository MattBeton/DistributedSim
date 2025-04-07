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

        self.master_model = deepcopy(model).to("cpu")

        if self.rank == 0:
            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                            **self.gradient_config.outer_optimizer_kwargs)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

    def _reduce_models(self) -> None:
        if self.gradient_config.merge_method == 'mean':
            for param in self.model.parameters():
                reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    param.data /= self.config.num_nodes
        elif self.gradient_config.merge_method == 'fisher1':
            for (name, param), (master_name, master_param) in zip(self.model.named_parameters(), self.master_model.named_parameters()):
                assert name == master_name, f"{name} != {master_name}?"

                master_param_data = master_param.data.to(param.device)
                param_grad = param.data - master_param_data
                
                pass

        elif self.gradient_config.merge_method == 'fisher2':
            for (name, param), (master_name, master_param) in zip(self.model.named_parameters(), self.master_model.named_parameters()):
                assert name == master_name, f"{name} != {master_name}?"
                if param.requires_grad:
                    master_param_data = master_param.data.to(param.device)
                    param_grad = param.data - master_param_data

                    beta1, beta2 = 0.9, 0.999
                    exp_avg = self.optim.state[param]['exp_avg']
                    exp_avg_sq = self.optim.state[param]['exp_avg_sq']
                    bias_correction1 = 1 - beta1 ** self.optim.state[param]["step"]
                    bias_correction2 = 1 - beta2 ** self.optim.state[param]["step"]

                    corr1 = exp_avg / bias_correction1
                    corr2 = exp_avg_sq / bias_correction2

                    var = torch.sqrt(corr2) / corr1

                    E = 10.0 * torch.mean(var)

                    param_grad.mul_(var + E)

                    all_reduce(exp_avg, op=dist.ReduceOp.SUM)
                    exp_avg /= self.config.num_nodes
                    all_reduce(exp_avg_sq, op=dist.ReduceOp.SUM)
                    exp_avg_sq /= self.config.num_nodes

                    corr1 = exp_avg / bias_correction1
                    corr2 = exp_avg_sq / bias_correction2
                    var = torch.sqrt(corr2) / corr1

                    param_grad.div_(var + E)

                    param.data = master_param_data + param_grad
                else:
                    assert torch.isclose(param, master_param.to(param.device)).all()
        else:
            raise NotImplementedError(f"Merge method unknown: {merge_method}")


    def _sync_opt_state(self) -> None:
        for param in self.model.parameters():
            if param.requires_grad:
                all_reduce(self.optim.state[param]['exp_avg'].data, op=dist.ReduceOp.SUM)
                self.optim.state[param]['exp_avg'].data /= self.config.num_nodes

            if param.requires_grad:
                all_reduce(self.optim.state[param]['exp_avg_sq'], op=dist.ReduceOp.SUM)
                self.optim.state[param][ 'exp_avg_sq'].data /= self.config.num_nodes

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            broadcast(param.data, src=0)

        for master_param in self.master_model.parameters():
            broadcast(master_param.data, src=0)

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
        self.optim.step()

        # Outer step if needed.
        if self.local_step == 0:
            if self.gradient_config.sync_opt_state:
                self._sync_opt_state()

        if self.local_step % self.gradient_config.outer_interval == 0 and self.local_step > 0:
            self._reduce_models()

            if self.rank == 0:
                self.outer_optimizer.zero_grad()
                self._set_master_grad()
                self.outer_optimizer.step()
                self._synchronize_master_model()

            self._broadcast_model_params()

            if self.gradient_config.sync_opt_state:
                self._sync_opt_state()


        super().step()

        self.local_step += 1
