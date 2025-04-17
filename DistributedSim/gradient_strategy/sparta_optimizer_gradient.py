import math
import torch
import torch.distributed as dist
from torch import nn
from copy import deepcopy

from .gradient_strategy import GradientStrategy
from .sparta_gradient import RandomIndexSelector
from .communicate import *

class SPARTAOptimizerGradient(GradientStrategy):
    """
    Two‑level SPARTA strategy in which the *inner* optimiser runs on every worker
    and an *outer* optimiser updates a master model on rank 0 using sparse
    pseudo‑gradients.  With SGD(lr=1) this reproduces the behaviour of the simpler
    SPARTAGradient strategy.
    """

    def __init__(self, rank, model, config, logger=None):
        super().__init__(rank, model, config, logger)

        # ---------- inner optimiser ----------
        self.optim = self.gradient_config.optimizer_class(
            model.parameters(),
            **self.gradient_config.optimizer_kwargs
        )
        self._setup_scheduler()

        # ---------- outer optimiser & master model (rank 0 only) ----------
        if self.rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for p in self.master_model.parameters():
                p.requires_grad = True
            self.outer_optimizer = self.gradient_config.outer_optimizer_cls(
                self.master_model.parameters(),
                **self.gradient_config.outer_optimizer_kwargs
            )
            # handy lookup table for real Parameter objects                   ← FIX
            self._master_params = dict(self.master_model.named_parameters())

        self.index_selector = RandomIndexSelector(self.gradient_config.p_sparta)

    # ---------------------------------------------------------------------
    def step(self):
        # ---------- inner step -------------------------------------------------
        if self.gradient_config.max_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.gradient_config.max_norm)
        self.optim.step()

        # ---------- sparse synchronisation & outer step ------------------------
        if self.config.num_nodes > 1:
            # keep the indices used for every parameter this round
            param_indices = {}

            # zero outer grads on rank 0
            if self.rank == 0:
                for p in self.master_model.parameters():
                    p.grad = None

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue

                    # ---- choose indices (rank 0) & broadcast -----------------
                    indices = self.index_selector.get_indices(param)
                    broadcast(indices, src=0)
                    param_indices[name] = indices

                    # ---- average sparse data across workers -----------------
                    sparse = param.data[indices]
                    all_reduce(sparse, op=dist.ReduceOp.SUM)
                    sparse /= dist.get_world_size()
                    param.masked_scatter_(indices, sparse)

                    # ---- build pseudo‑gradient on rank 0 --------------------
                    if self.rank == 0:
                        idx_cpu = indices.to("cpu")
                        master_param = self._master_params[name]              # ← FIX
                        # grad = θ_master – θ_avg  (so SGD(lr=1) sends θ_master → θ_avg)
                        grad_sparse = master_param.data[idx_cpu] - sparse.to("cpu")
                        # create / fill full‑shape gradient tensor
                        if master_param.grad is None:
                            master_param.grad = torch.zeros_like(master_param.data)
                        master_param.grad[idx_cpu] = grad_sparse

                # ---- outer optimiser step (rank 0) --------------------------
                if self.rank == 0:
                    self.outer_optimizer.step()

                # ---- broadcast updated sparse parts back to all ranks -------
                for name, param in self.model.named_parameters():
                    if name not in param_indices:
                        continue
                    indices = param_indices[name]

                    if self.rank == 0:
                        idx_cpu = indices.to("cpu")
                        updated = self._master_params[name].data[idx_cpu].to(param.device)
                    else:
                        updated = torch.empty_like(param.data[indices])
                    broadcast(updated, src=0)
                    param.masked_scatter_(indices, updated)

        # ---------- bookkeeping ----------------------------------------------
        super().step()
