import torch.distributed as dist
from copy import deepcopy

import numpy as np
from torch import nn
from torch.nn import utils as nn_utils
import torch

from .gradient_strategy import GradientStrategy
from .communicate import *

from curvlinops import CGInverseLinearOperator, GGNLinearOperator
from curvlinops import KFACInverseLinearOperator, KFACLinearOperator
from curvlinops import FisherType, KFACType
from backpack.utils.convert_parameters import vector_to_parameter_list


class GPTWrapper(nn.Module):
    """Wraps Karpathy's nanoGPT model repo so that it produces the flattened logits."""

    def __init__(self, gpt: nn.Module):
        """Store the wrapped nanoGPT model.

        Args:
            gpt: The nanoGPT model.
        """
        super().__init__()
        self.gpt = gpt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the nanoGPT model.

        Args:
            x: The input tensor. Has shape ``(batch_size, sequence_length)``.

        Returns:
            The flattened logits.
            Has shape ``(batch_size * sequence_length, vocab_size)``.
        """
        y_dummy = torch.zeros_like(x)

        logits, _ = self.gpt(x, y_dummy)

        return logits.view(-1, logits.size(-1))



class FedAvgGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None, curv_dataloader=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                            **self.gradient_config.outer_optimizer_kwargs)

        self.optim = self.gradient_config.optimizer_class(model.parameters(), 
                                                          **self.gradient_config.optimizer_kwargs)
        self._setup_scheduler()

        self.wrapped_model = GPTWrapper(model)
        self.device = list(model.parameters())[0].device

        if curv_dataloader is not None:
            self.curv_dataloader = curv_dataloader
        else:
            self.curv_dataloader = None

    def _fisher(self) -> None:
        loss_function = nn.CrossEntropyLoss(ignore_index=-1)

        self.wrapped_model.eval()

        fisher = KFACLinearOperator(
                self.wrapped_model,
                loss_function,
                    [p for p in self.wrapped_model.parameters() if p.requires_grad],
                self.curv_dataloader,
                fisher_type=FisherType.FORWARD_ONLY if self.gradient_config.forward_only else FisherType.MC,
                separate_weight_and_bias=False,
                #num_per_example_loss_terms=100,
                #kfac_approx=KFACType.REDUCE,
                check_deterministic=False,
                progressbar=True
            ) #.to_scipy()

        return fisher


    def _reduce_models(self) -> None:
        if self.gradient_config.merge_method == 'mean':
            for param in self.model.parameters():
                reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    param.data /= self.config.num_nodes
        elif self.gradient_config.merge_method in ('curv0', 'curv1'):
            fisher = self._fisher()

            # flatten and convert to numpy
            theta = nn.utils.parameters_to_vector((p for p in self.model.parameters() if p.requires_grad))
            theta = theta.detach()
            #theta = theta.cpu().detach().numpy()

            USE_EXACT_DAMPING = not self.gradient_config.damping_meantrick
            damping = self.gradient_config.damping

            # RHS of Fisher merge
            #rhs = fisher.to_scipy() @ theta
            rhs = fisher @ theta #.clone()
            #np.save("rhs.npy", rhs)
            #np.save("theta.npy", theta)
            torch.save(rhs, 'rhs.pt')
            torch.save(theta, 'theta.pt')

            #print('mae', np.mean(np.abs(rhs - theta)))
            print('mae', torch.mean(torch.abs(rhs - theta)))
            if self.gradient_config.merge_method == 'curv0':
                pass
            else: # curv1
                rhs = rhs + damping * theta

            #rhs = torch.tensor(rhs)

            all_reduce(rhs)
            rhs /= self.config.num_nodes

            #rhs = rhs.numpy()

            # Reduce and invert
            fisher_dict = fisher.state_dict()
            for key in fisher_dict['input_covariances'].keys():
                in_mat = fisher_dict['input_covariances'][key].clone().to(self.device)
                torch.save(fisher_dict['input_covariances'][key], f"{self.rank}-{key}-A.pt")
                torch.save(fisher_dict['gradient_covariances'][key], f"{self.rank}-{key}-G.pt")

                all_reduce(in_mat)
                in_mat /= self.config.num_nodes
                fisher_dict['input_covariances'][key].copy_(in_mat.cpu())

            fisher.load_state_dict(fisher_dict)

            fisher_sum_inv = KFACInverseLinearOperator(fisher, damping=damping, use_exact_damping=USE_EXACT_DAMPING, use_heuristic_damping=self.gradient_config.damping_meantrick) # FIX

            #fisher_weighted_params = fisher_sum_inv.to_scipy() @ rhs
            fisher_weighted_params = fisher_sum_inv @ rhs

            params = [p for p in self.model.parameters() if p.requires_grad]
            # theta_fisher = vector_to_parameter_list(
            #     torch.from_numpy(fisher_weighted_params), params
            # )
            theta_fisher = vector_to_parameter_list(
                fisher_weighted_params, params
            )
            for theta, (name, param) in zip(theta_fisher, [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]):
                if self.gradient_config.skip_embed_curv and (('wte' in name) or ('wpe' in name) or ('lm_head' in name)):
                    print(f"Using naive merging for: ", name)
                    reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
                    if self.rank == 0:
                        param.data /= self.config.num_nodes
                else:
                    param.data = theta.to(param.device).to(param.dtype).data
        elif self.gradient_config.merge_method == 'curv1':
            raise NotImplementedError(f"CURV1")
            fisher = self._fisher()


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
                    #master_param_data = master_param.data.to(param.device)
                    #param_grad = param.data - master_param_data
                    param_data = param.daga

                    beta1, beta2 = 0.9, 0.999
                    exp_avg = self.optim.state[param]['exp_avg']
                    exp_avg_sq = self.optim.state[param]['exp_avg_sq']
                    bias_correction1 = 1 - beta1 ** self.optim.state[param]["step"]
                    bias_correction2 = 1 - beta2 ** self.optim.state[param]["step"]

                    corr1 = exp_avg / bias_correction1
                    corr2 = exp_avg_sq / bias_correction2

                    var = torch.sqrt(corr2) / corr1

                    #E = 10.0 * torch.mean(var)
                    E = 1e-8

                    param_data.mul_(var + E)

                    all_reduce(exp_avg, op=dist.ReduceOp.SUM)
                    exp_avg /= self.config.num_nodes
                    all_reduce(exp_avg_sq, op=dist.ReduceOp.SUM)
                    exp_avg_sq /= self.config.num_nodes

                    corr1 = exp_avg / bias_correction1
                    corr2 = exp_avg_sq / bias_correction2
                    var = torch.sqrt(corr2) / corr1

                    param_data.div_(var + E)

                    param.data = param_data #master_param_data + param_grad
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

        #for master_param in self.master_model.parameters():
        #    broadcast(master_param.data, src=0)

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

        # sync weights that would otherwise have been shared
        self.model.transformer.wte.weight.data.copy_(0.5 * (self.model.transformer.wte.weight.data + self.model.lm_head.weight.data))
        self.model.lm_head.weight.data.copy_(self.model.transformer.wte.weight.data)

        self.local_step += 1
