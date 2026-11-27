import math
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(d_in, d_out, weights, in_features):
    return in_features @ weights.T


def run_embedding(vocab_size, d_model, weights, token_ids):
    return weights[token_ids]


def run_silu(in_features):
    return in_features * torch.sigmoid(in_features)


def run_rmsnorm(d_model, eps, weights, in_features):
    original_dtype = in_features.dtype

    x = in_features.to(torch.float32)
    gain = weights.to(torch.float32)

    rms = torch.sqrt(
        torch.mean(x ** 2, dim=-1, keepdim=True) + eps
    )

    output = (x / rms) * gain

    return output.to(original_dtype)

def run_swiglu(
        d_model,
        d_ff,
        w1_weight,
        w2_weight,
        w3_weight,
        in_features,):
    
    x1 = in_features @ w1_weight.T
    x1 = run_silu(x1)
    x3 = in_features @ w3_weight.T
    hidden = x1 * x3
    output = hidden @ w2_weight.T
    return output

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        mask = torch.broadcast_to(mask,scores.shape)
        scores = scores.masked_fill(~mask, float("-inf"))

    max_scores = scores.max(dim=-1, keepdim=True).values

    max_scores = torch.where(
        torch.isfinite(max_scores),
        max_scores,
        torch.zeros_like(max_scores),
    )

    exp_scores = torch.exp(scores - max_scores)

    attention_weights = exp_scores / exp_scores.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(exp_scores.dtype).tiny)

    output = attention_weights @ V

    return output

def run_rope(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    if d_k % 2 != 0:
        raise ValueError("d_k must be even")
    if in_query_or_key.shape[-1] != d_k:
        raise ValueError("Input dimension does not match d_k")

    if token_positions.shape[-1] != in_query_or_key.shape[-2]:
        raise ValueError("Token positions do not match sequence length")

    original_dtype = in_query_or_key.dtype
    x = in_query_or_key.to(torch.float32)
    
    freq_indices = torch.arange(
        0,
        d_k,
        2,
        device=in_query_or_key.device,
        dtype=torch.float32,
    )

    freqs = 1.0 / (theta ** (freq_indices / d_k))
    positions = token_positions.to(
        device=in_query_or_key.device,
        dtype=torch.float32,
    )

    angles = positions.unsqueeze(-1) * freqs
    x = in_query_or_key.to(torch.float32)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    output = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1,
    ).flatten(-2)

    return output.to(original_dtype)

def run_softmax(
        in_features: Tensor,
        dim: int,
) -> Tensor:
    shifted_inputs = in_features - torch.amax(
        in_features,
        dim=dim,
        keepdim=True,
    )

    exp_inputs = torch.exp(shifted_inputs)

    denominator = torch.sum(
        exp_inputs,
        dim=dim,
        keepdim=True,
    )

    return exp_inputs / denominator

def run_cross_entropy(
        inputs: Tensor,
        targets: Tensor,
) -> Tensor:

    log_sum_exp = torch.logsumexp(
        inputs,
        dim=-1,
    )

    target_logits = inputs[
        torch.arange(
            inputs.shape[0],
            device=inputs.device,
        ),
        targets,
    ]

    losses = log_sum_exp - target_logits

    return losses.mean()