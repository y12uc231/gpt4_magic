import torch
import torch.nn as nn
from torch.optim import Optimizer


class CustomOptimizer(Optimizer):
    def __init__(self, params, lr, momentum, weight_decay, d_dim, k, alpha):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, d_dim=d_dim, k=k, alpha=alpha)
        super(CustomOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            d_dim = group['d_dim']
            k = group['k']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(d_p)
                buf = state['momentum_buffer']

                # Reshape the gradient to 2D
                shape = d_p.shape
                reshaped_d_p = d_p.view(shape[d_dim], -1)

                # SVD decomposition
                u, s, v = torch.svd(reshaped_d_p)

                # Get top k and 2k singular vectors
                u_k, s_k, v_k = u[:, :k], torch.diag_embed(s[:k]), v[:, :k]
                u_2k, s_2k, v_2k = u[:, :2*k], torch.diag_embed(s[:2*k]), v[:, :2*k]

                # Compute W_k and W_{2k}
                w_k = u_k @ s_k @ v_k.t()
                w_2k = u_2k @ s_2k @ v_2k.t()

                # Normalize W_k
                w_k /= w_2k.norm() ** momentum

                # Truncate coordinates to top alpha percentile
                alpha_percentile = torch.quantile(torch.abs(w_k), alpha)
                w_k = torch.where(torch.abs(w_k) > alpha_percentile, w_k, torch.zeros_like(w_k))

                # Update the momentum buffer and the parameters
                buf.mul_(momentum).add_(w_k)
                p.data.add_(buf, alpha=-group['lr'])

        return loss

