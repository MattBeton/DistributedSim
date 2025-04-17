import torch
from torch.optim.optimizer import Optimizer, required
import warnings # Import warnings

class SparseAwareSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum)
    with support for sparse gradients.

    For sparse gradients (torch.sparse_coo_tensor), the optimizer state (momentum buffer)
    is only updated for parameters corresponding to non-zero gradient entries.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0).
            Note: Weight decay is currently NOT supported for sparse gradients
            and will raise a ValueError if used with sparse gradients.
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> # Dense Gradients
        >>> optimizer = SparseAwareSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward() # Creates dense gradients
        >>> optimizer.step()
        >>>
        >>> # Sparse Gradients (Example setup)
        >>> # Assume 'model' has parameters where sparse gradients make sense
        >>> optimizer = SparseAwareSGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> # --- Manual creation of a sparse grad for demonstration ---
        >>> for p in model.parameters():
        >>>    if p.requires_grad and p.dim() > 1: # Example: apply to a weight matrix
        >>>        i = torch.LongTensor([[0], [0]]) # Index of non-zero element
        >>>        v = torch.FloatTensor([1.0])     # Value of non-zero element
        >>>        p.grad = torch.sparse_coo_tensor(i, v, p.size(), device=p.device)
        >>>    elif p.requires_grad:
        >>>        p.grad = torch.zeros_like(p) # Ensure other grads exist but are zero
        >>> # ---------------------------------------------------------
        >>> optimizer.step() # Will use sparse logic for the parameter with sparse grad

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as

        Dense:
            $v_{t+1} = \mu * v_{t} + g_{t+1}$
            $p_{t+1} = p_{t} - \text{lr} * v_{t+1}$

        Sparse (This Implementation - update only non-zero gradient indices 'nz'):
            $v_{t+1}[nz] = \mu * v_{t}[nz] + g_{t+1}[nz]$
            $p_{t+1}[nz] = p_{t}[nz] - \text{lr} * (\text{nesterov ? } (g_{t+1}[nz] + \mu * v_{t+1}[nz]) \text{ : } v_{t+1}[nz])$
            (Where $v_t[nz]$ and $p_t[nz]$ refer to the state/parameter values at the
             indices corresponding to non-zero entries in $g_{t+1}$)

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
            $v_{t+1} = \mu * v_{t} + \text{lr} * g_{t+1}$
            $p_{t+1} = p_{t} - v_{t+1}$

        The Nesterov version is analogously modified.

    .. warning::
        Weight decay (`weight_decay > 0`) is not supported for sparse gradients
        in this implementation due to the ambiguity of applying parameter-based decay
        based on gradient sparsity. Using `weight_decay > 0` with sparse gradients
        will result in a `ValueError`.

    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad() # Ensure operations inside step don't track gradients
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad(): # Enable grad for closure evaluation
                 loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []
            has_sparse_grad = False # Track if any param in group has sparse grad

            # --- First pass: check grads, handle weight decay for dense ---
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True
                        if group['weight_decay'] != 0:
                             raise ValueError("Weight decay is not supported for sparse gradients in this implementation.")

                    state = self.state[p]
                    # Lazy state initialization
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # --- Check for sparse gradients and unsupported weight decay ---
            if has_sparse_grad and group['weight_decay'] != 0:
                 # This check is redundant due to the loop above, but kept for clarity
                raise ValueError("Weight decay > 0 is not supported with sparse gradients.")

            # --- Second pass: apply updates ---
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for i, p in enumerate(params_with_grad):
                d_p = grads[i]

                # ================== DENSE GRADIENT PATH ==================
                if not d_p.is_sparse:
                    if weight_decay != 0:
                        # In-place add: d_p = d_p + p * weight_decay
                        d_p = d_p.add(p, alpha=weight_decay)

                    if momentum != 0:
                        buf = momentum_buffer_list[i]

                        if buf is None:
                            # Initialize momentum buffer
                            buf = torch.clone(d_p).detach()
                            momentum_buffer_list[i] = buf
                            self.state[p]['momentum_buffer'] = buf
                        else:
                            # Update buffer: buf = buf * momentum + d_p * (1 - dampening)
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                        if nesterov:
                            # d_p = d_p + buf * momentum
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            # d_p = buf
                            d_p = buf

                    # Parameter update: p = p - lr * d_p
                    p.add_(d_p, alpha=-lr)

                # ================== SPARSE GRADIENT PATH ==================
                else:
                    # We already checked weight_decay != 0 is not allowed for sparse
                    assert weight_decay == 0

                    # Make gradient sparse (ensure it's COO format, common case)
                    d_p = d_p.coalesce()
                    grad_indices = d_p._indices() # Shape (ndim, nnz)
                    grad_values = d_p._values()   # Shape (nnz,)
                    size = d_p.size()

                    if grad_values.numel() == 0:
                        # Skip update if sparse gradient has no non-zero elements
                        continue

                    if momentum != 0:
                        buf = momentum_buffer_list[i]

                        if buf is None:
                            # Initialize momentum buffer - must be dense to store momentum correctly
                            buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                            momentum_buffer_list[i] = buf
                            self.state[p]['momentum_buffer'] = buf
                            # Initialize only the non-zero parts of the buffer for the first step
                            buf.sparse_mask(d_p).add_(grad_values * (1.0 - dampening))
                            momentum_update_values = buf.sparse_mask(d_p)._values() # Get initial values

                        else:
                            # --- Update momentum buffer only for non-zero gradient indices ---
                            # 1. Select the buffer values corresponding to sparse indices
                            #    Need to flatten indices for index_select if ndim > 1
                            if p.dim() == 1:
                                buf_nz = buf.index_select(0, grad_indices[0])
                            else:
                                # For >1D, need to convert sparse indices (e.g., [dim0_idx, dim1_idx])
                                # to flat indices for use with index_select/index_put on flattened buffer
                                flat_indices = grad_indices[0]
                                for dim in range(1, p.dim()):
                                    flat_indices = flat_indices * size[dim] + grad_indices[dim]
                                buf_nz = buf.view(-1).index_select(0, flat_indices)

                            # 2. Apply momentum update rule to these selected values
                            buf_nz.mul_(momentum).add_(grad_values, alpha=1 - dampening)
                            momentum_update_values = buf_nz # Store for potential Nesterov use

                            # 3. Put the updated values back into the main buffer
                            if p.dim() == 1:
                                buf.index_put_((grad_indices[0],), buf_nz, accumulate=False)
                            else:
                                buf.view(-1).index_put_((flat_indices,), buf_nz, accumulate=False)
                        # --- End momentum buffer update ---

                        # Prepare d_p for parameter update
                        if nesterov:
                             # d_p_values = grad_values + momentum_update_values * momentum
                             d_p_values = grad_values.add(momentum_update_values, alpha=momentum)
                        else:
                             # d_p_values = momentum_update_values (updated buffer values)
                             d_p_values = momentum_update_values
                    else: # No momentum
                         d_p_values = grad_values # Just use the gradient values directly

                    # --- Parameter update only for non-zero gradient indices ---
                    # Use sparse addition (add_ with sparse tensor) or indexing
                    # index_add_ is generally safe and handles potential index overlaps
                    if p.dim() == 1:
                        p.index_add_(0, grad_indices[0], d_p_values, alpha=-lr)
                    else:
                        # Use flat indices calculated earlier if available
                        if 'flat_indices' not in locals() or flat_indices is None:
                             flat_indices = grad_indices[0]
                             for dim in range(1, p.dim()):
                                 flat_indices = flat_indices * size[dim] + grad_indices[dim]
                        # Apply update to flattened parameter tensor using flat indices
                        p.view(-1).index_add_(0, flat_indices, d_p_values, alpha=-lr)


        return loss