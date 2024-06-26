"""
Courtesy of https://github.com/shacklettbp/bps-nav
See section 3.4 in https://arxiv.org/pdf/2103.07013.pdf

"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.optim
from torch import Tensor
from torch.optim.optimizer import Optimizer, _dispatch_sqrt, _get_value, _use_grad_for_differentiable


def _single_tensor_adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        # update step
        step_t += 1
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        step = _get_value(step_t)

        # pytorch adam implementation:
        # bias_correction1 = 1 - beta1 ** step
        # bias_correction2 = 1 - beta2 ** step
        # step_size = lr / bias_correction1
        # bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        # param.addcdiv_(exp_avg, denom, value=-step_size)

        # tensorflow adam implementation:
        lr_t = lr * _dispatch_sqrt(1 - beta2**step) / (1 - beta1**step)
        denom = exp_avg_sq.sqrt().add_(eps)
        param.addcdiv_(exp_avg, denom, value=-lr_t)


def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    func = _single_tensor_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


class AdamTensorFlowStyle(torch.optim.Adam):
    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


class Lamb(Optimizer):
    def __init__(
        self,
        params,
        bias_correction=True,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=1e-4,
        min_trust=0.01,
        use_look_ahead=False,
        look_ahead_alpha=0.5,
        look_ahead_k=10,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= min_trust <= 1.0:
            raise ValueError("min_trust must be in [0, 1], got {}".format(min_trust))

        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            min_trust=min_trust,
            use_look_ahead=use_look_ahead,
            look_ahead_alpha=look_ahead_alpha,
            look_ahead_k=look_ahead_k,
        )
        super().__init__(params, defaults)

    def zero_grad(self, **kwargs):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()

    def _compute_adam_step(self, group, p, weight_decay, use_look_ahead):
        grad = p.grad.data
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(p.data)

            state["step"] = 1

            if use_look_ahead:
                state["slow_param"] = p.data.clone()

        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]

        # Decay the first and second moment running average coefficient
        # m_t
        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
        # v_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

        m = exp_avg.clone()
        v = exp_avg_sq.sqrt()

        if group["bias_correction"]:
            m = m.mul_(1 / (1 - beta1 ** state["step"]))
            v = v.mul_(1 / math.sqrt(1 - beta2 ** state["step"]))

        adam_step = m.div_(v.add_(group["eps"]))

        if weight_decay > 0:
            adam_step.add_(p.data, alpha=weight_decay)

        return adam_step

    def _step_list_params(self, group):
        min_trust = group["min_trust"]
        weight_decay = group["weight_decay"]
        step_size = group["lr"]
        alpha = group["look_ahead_alpha"]
        k = group["look_ahead_k"]
        use_look_ahead = group["use_look_ahead"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError("Lamb does not support sparse gradients, consider SparseAdam instead.")

            adam_step = self._compute_adam_step(group, p, weight_decay, use_look_ahead)

            if min_trust != 1.0:
                weight_norm = torch.norm(p.data.detach()).item()
                step_norm = torch.norm(adam_step.detach()).item()

                if weight_norm == 0 or step_norm == 0 or min_trust == 1.0:
                    trust_ratio = 1
                else:
                    trust_ratio = min(weight_norm, 10.0) / step_norm
                    trust_ratio = min(max(trust_ratio, min_trust), 1.0 / min_trust)
            else:
                trust_ratio = 1.0

            state = self.state[p]

            p.data.add_(adam_step, alpha=-step_size * trust_ratio)

            if use_look_ahead and (state["step"] % k) == 0:
                state["slow_param"].mul_(1 - alpha).add_(p.data, alpha=alpha)
                p.data.copy_(state["slow_param"])

            state["step"] += 1

    def _step_flat_params(self, group):
        min_trust = group["min_trust"]
        weight_decay = group["weight_decay"]
        step_size = group["lr"]
        alpha = group["look_ahead_alpha"]
        k = group["look_ahead_k"]
        use_look_ahead = group["use_look_ahead"]

        adam_step = self._compute_adam_step(group, group["params"][0], weight_decay, use_look_ahead)

        if min_trust != 1.0:
            ptr = 0
            for p in group["list_params"]:
                weight_norm = torch.norm(p.data.detach()).item()
                step_norm = torch.norm(adam_step[ptr : ptr + p.numel()].data.detach()).item()

                if weight_norm == 0 or step_norm == 0 or min_trust == 1.0:
                    trust_ratio = 1
                else:
                    trust_ratio = min(weight_norm, 10.0) / step_norm
                    trust_ratio = min(max(trust_ratio, min_trust), 1.0 / min_trust)

                adam_step[ptr : ptr + p.numel()].mul_(trust_ratio)
                ptr += p.numel()

        p = group["params"][0]
        state = self.state[p]

        p.data.add_(adam_step, alpha=-step_size)

        if use_look_ahead and (state["step"] % k) == 0:
            state["slow_param"].mul_(1 - alpha).add_(p.data, alpha=alpha)
            p.data.copy_(state["slow_param"])

        state["step"] += 1

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if "list_params" in group:
                self._step_flat_params(group)
            else:
                self._step_list_params(group)

        return loss
