import torch
from torch import nn
from typing import Iterable

import math

def run_gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm:float,
) -> None:

    parameters = list(parameters)

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    if len(gradients) == 0:
        return 

    total_squared_norm = torch.zeros(
        (),
        device=gradients[0].device,
        dtype=torch.float32,
    )

    for gradient in gradients:
        total_squared_norm += gradient.detach().float().pow(2).sum()

    total_norm = torch.sqrt(total_squared_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for gradient in gradients:
            gradient.mul_(scale)
class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float=1e-3,
        betas: tuple[float,float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")

        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                    # 一阶和二阶指数移动平均
                exp_avg.mul_(beta1).add_(
                    gradient,
                    alpha=1 - beta1,
                )

                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                    # bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = lr / bias_correction1

                denominator = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(eps)

                    # Adam 参数更新
                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

                    # decoupled weight decay
                parameter.mul_(
                    1 - lr * weight_decay
                )

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it >= cosine_cycle_iters:
        return min_learning_rate

    decay_ratio = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    coefficient = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return (
        min_learning_rate
        + coefficient
        * (max_learning_rate - min_learning_rate)
    )