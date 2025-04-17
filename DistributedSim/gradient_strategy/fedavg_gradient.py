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
from curvlinops import EKFACLinearOperator
from curvlinops import FisherType, KFACType
from backpack.utils.convert_parameters import vector_to_parameter_list

from ..soap import SOAP


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



class CombinedOptimizer:

    def __init__(self, optimizers):
        assert all(len(opt.param_groups) == 1 for opt in optimizers)
        self.optimizers = optimizers
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]['lr'] for opt in self.optimizers]

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]['lr'] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]


class FedAvgGradient(GradientStrategy):
    def __init__(self, rank, model, config, logger=None, curv_dataloader=None):
        super().__init__(rank, model, config, logger)

        self.local_step = 0

        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(self.master_model.parameters(), 
                                                                            **self.gradient_config.outer_optimizer_kwargs)

        params = [param for name, param in model.named_parameters() if ('wt' not in name) and ('embed' not in name)]

        self.optim = CombinedOptimizer([
                torch.optim.AdamW(model.lm_head.parameters(), lr=0.0018, betas=(0.9, 0.95), weight_decay=0),
                SOAP(params, lr=0.01, betas=(.95, .95), weight_decay=0, precondition_frequency=10)
            ])

        #OrthogonalNesterov(self.transformer.h.parameters(), lr=10 * learning_rate, momentum=0.95)
        #self.optim = self.gradient_config.optimizer_class(model.parameters(), 
        #                                     **self.gradient_config.optimizer_kwargs)
        #self._setup_scheduler()

        self.wrapped_model = GPTWrapper(model)
        self.device = list(model.parameters())[0].device

        if curv_dataloader is not None:
            self.curv_dataloader = curv_dataloader
        else:
            self.curv_dataloader = None

    def _fisher(self) -> None:
        self.wrapped_model.eval()

        loss_function = nn.CrossEntropyLoss(ignore_index=-1)

        fisher = KFACLinearOperator(
                self.wrapped_model,
                loss_function,
                    [p for p in self.wrapped_model.parameters() if p.requires_grad],
                self.curv_dataloader,
                fisher_type=FisherType.FORWARD_ONLY if self.gradient_config.forward_only else FisherType.MC,
                #fisher_type=FisherType.EMPIRICAL,
                separate_weight_and_bias=False,
                check_deterministic=False,
                progressbar=True
            ) #.to_scipy()

        return fisher


    def _reduce_models(self) -> None:
        if self.gradient_config.merge_method == 'nothing':
            pass
        elif self.gradient_config.merge_method == 'mean':
            for param in self.model.parameters():
                #reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
                #if self.rank == 0:
                #    param.data /= self.config.num_nodes
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= self.config.num_nodes
        elif self.gradient_config.merge_method == 'soap':
            self._sync_opt_state(skip_exp=True)

            for name, param in self.model.named_parameters():
                if any(param is p for group in self.optim.optimizers[1].param_groups for p in group['params']):
                    if not param.requires_grad:
                        continue
                    param_avg = param.data.clone()
                    all_reduce(param_avg, op=dist.ReduceOp.SUM)
                    param_avg /= self.config.num_nodes

                    outer_grad = param.data - param_avg

                    state = self.optim.optimizers[1].state[param]

                    # START FISHER ZONE
                    outer_grad_projected = self.optim.optimizers[1].project(outer_grad, state, merge_dims=False, max_precond_dim=10000)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    beta1, beta2 = 0.95, 0.95
                    eps = 1e-8

                    fisher = exp_avg_sq #.sqrt().add_(eps)
                    #fisher += 1000.0 #* torch.mean(fisher)

                    outer_grad_projected *= fisher

                    all_reduce(outer_grad_projected, op=dist.ReduceOp.SUM)
                    outer_grad_projected /= self.config.num_nodes

                    all_reduce(fisher, op=dist.ReduceOp.SUM)
                    fisher /= self.config.num_nodes

                    fisher += self.gradient_config.damping * torch.mean(fisher)

                    outer_grad_projected /= fisher.add_(1e-8)

                    fisher_grad = self.optim.optimizers[1].project_back(outer_grad_projected, state, merge_dims=False, max_precond_dim=10000)

                    # END FISHER ZONE
                    param.data.copy_(param_avg + fisher_grad)
                else:
                    all_reduce(param.data, op=dist.ReduceOp.SUM)
                    param.data /= self.config.num_nodes

            self._sync_opt_state(skip_GG=True)
        elif self.gradient_config.merge_method in ('curv0', 'curv1'):
            fisher = self._fisher()

            # flatten and convert to numpy
            theta = nn.utils.parameters_to_vector((p.detach() for p in self.model.parameters() if p.requires_grad))

            USE_EXACT_DAMPING = not self.gradient_config.damping_meantrick
            damping = self.gradient_config.damping

            theta_avg = theta.clone()
            all_reduce(theta_avg)
            theta_avg /= self.config.num_nodes

            # RHS of Fisher merge
            rhs = fisher @ (theta - theta_avg)
            #torch.save(rhs, 'rhs.pt')
            #torch.save(theta, 'theta.pt')

            #print('mae', np.mean(np.abs(rhs - theta)))
            #print('mae', torch.mean(torch.abs(rhs - theta)))
            if self.gradient_config.merge_method == 'curv0':
                pass
            else: # curv1
                #rhs = rhs + damping * theta
                rhs.add_(damping * theta)

            all_reduce(rhs)
            rhs /= self.config.num_nodes
            #torch.save(rhs, 'reduced_rhs.pt')

            # Reduce and invert
            fisher_dict = fisher.state_dict()
            for key in fisher_dict['input_covariances'].keys():
                in_mat = fisher_dict['input_covariances'][key].clone().to(self.device)

                all_reduce(in_mat)
                in_mat /= self.config.num_nodes
                fisher_dict['input_covariances'][key].copy_(in_mat.cpu())

                #torch.save(in_mat.detach().cpu(), f"{self.rank}-{key}-A.pt")

            for key in fisher_dict['gradient_covariances'].keys():
                gradient_mat = fisher_dict['gradient_covariances'][key]

                if type(gradient_mat) == tuple:
                    #if self.gradient_config.skip_embed_curv and (('wte' in key) or ('wpe' in key) or ('lm_head' in key)):
                    #    pass
                    #else:
                    print(f"Weird in mat for key {key}: {gradient_mat}")
                    v = torch.ones(1, device=self.device) * gradient_mat[2] #.item()
                    all_reduce(v)
                    v /= self.config.num_nodes
                    gradient_mat = ("IDENTITY", gradient_mat[1], v.item())
                else:
                    gradient_mat = gradient_mat.clone().to(self.device)
                    all_reduce(gradient_mat)
                    gradient_mat /= self.config.num_nodes
                    fisher_dict['gradient_covariances'][key].copy_(gradient_mat.cpu())
                    torch.save(gradient_mat.detach().cpu(), f"{self.rank}-{key}-G.pt")

            fisher.load_state_dict(fisher_dict)

            fisher_sum_inv = KFACInverseLinearOperator(fisher, damping=damping,
                                                       use_exact_damping=USE_EXACT_DAMPING,
                                                       use_heuristic_damping=self.gradient_config.damping_meantrick,
                                                       #use_heuristic2_damping=True,
                                                       min_damping=1e-10) # FIX

            #fisher_weighted_params = fisher_sum_inv.to_scipy() @ rhs
            fisher_weighted_params = theta_avg + fisher_sum_inv @ rhs
            
            #torch.save(fisher_weighted_params, 'final.pt')

            params = [p.detach() for p in self.model.parameters() if p.requires_grad]
            # theta_fisher = vector_to_parameter_list(
            #     torch.from_numpy(fisher_weighted_params), params
            # )
            theta_fisher = vector_to_parameter_list(
                fisher_weighted_params, params
            )
            for theta, (name, param) in zip(theta_fisher, [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]):
                skip = False
                if self.gradient_config.skip_mlp_fc and ('mlp.c_fc' in name):
                    skip = True
                if self.gradient_config.skip_mlp_proj and ('mlp.c_proj' in name):
                    skip = True
                if self.gradient_config.skip_attn_attn and ('attn.c_attn' in name):
                    skip = True
                if self.gradient_config.skip_attn_proj and ('attn.c_proj' in name):
                    skip = True
                if self.gradient_config.skip_embed_curv and (('wte' in name) or ('wpe' in name) or ('lm_head' in name)):
                    skip = True

                if skip:
                    print(f"Naive merging : ", name)
                    reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
                    if self.rank == 0:
                        param.data /= self.config.num_nodes
                else:
                    print(f"Fisher merging : ", name)
                    param.data.copy_(theta.to(param.device).to(param.dtype).data)

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

    def _sync_opt_state(self, skip_exp=False, skip_GG=False) -> None:
        for name, param in self.model.named_parameters():
            if not skip_exp:
                for opt_i, optimizer in enumerate(self.optim.optimizers):
                    if param.requires_grad:
                        if any(param is p for group in optimizer.param_groups for p in group['params']):
                            all_reduce(optimizer.state[param]['exp_avg'].data, op=dist.ReduceOp.SUM)
                            optimizer.state[param]['exp_avg'].data /= self.config.num_nodes

                            all_reduce(optimizer.state[param]['exp_avg_sq'], op=dist.ReduceOp.SUM)
                            optimizer.state[param][ 'exp_avg_sq'].data /= self.config.num_nodes

            if not skip_GG:
                if param.requires_grad:
                    if any(param is p for group in self.optim.optimizers[1].param_groups for p in group['params']):
                        num_GG = len(self.optim.optimizers[1].state[param]['GG'])
                        for gg_i in range(num_GG):
                            if not self.optim.optimizers[1].state[param]['GG'][0] == []:
                                all_reduce(self.optim.optimizers[1].state[param]['GG'][gg_i].data, op=dist.ReduceOp.SUM)
                                self.optim.optimizers[1].state[param]['GG'][gg_i].data /= self.config.num_nodes

                        self.optim.optimizers[1].state[param]['Q'] = self.optim.optimizers[1].get_orthogonal_matrix_QR(self.optim.optimizers[1].state[param]) #['GG'])

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
        self.optim.step()

        # Outer step if needed.
        if (self.local_step == self.gradient_config.outer_warmup):
            if self.gradient_config.sync_opt_state:
                #print(f"STEP: {self.local_step} (sync opt)")
                self._sync_opt_state()

        if self.local_step % self.gradient_config.outer_interval == 0 and (self.local_step > self.gradient_config.outer_warmup):
            #print(f"STEP: {self.local_step} (merging)")
            self._reduce_models()

            # if self.rank == 0:
            #     self.outer_optimizer.zero_grad()
            #     self._set_master_grad()
            #     self.outer_optimizer.step()
            #     self._synchronize_master_model()

            # self._broadcast_model_params()

            if self.gradient_config.sync_opt_state:
                #print(f"STEP: {local_step} (sync opt)")
                self._sync_opt_state()

        # sync weights that would otherwise have been shared
        self.model.transformer.wte.weight.data.copy_(0.5 * (self.model.transformer.wte.weight.data + self.model.lm_head.weight.data))
        self.model.lm_head.weight.data.copy_(self.model.transformer.wte.weight.data)

        super().step()

        self.local_step += 1
