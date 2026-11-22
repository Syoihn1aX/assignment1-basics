import torch
from torch import Tensor

def _apply_rope(
        x: Tensor,
        token_positions: Tensor,
        theta: float,
) -> Tensor:

    *leading_dims, num_heads, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0

    half_dim = head_dim // 2

    pair_indices = torch.arange(
        half_dim,
        device=x.device,
        dtype=x.dtype,
    )

    frequencies = theta ** (
        -2 * pair_indices / head_dim
    )

    angles = token_positions.to(x.dtype).unsqueeze(-1) * frequencies

    cos_values = torch.cos(angles)
    sin_values = torch.sin(angles)

    cos_values = cos_values.unsqueeze(-3)
    sin_values = sin_values.unsqueeze(-3)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos_values - x_odd * sin_values
    rotated_odd = x_even * sin_values + x_odd * cos_values

    output = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1,
    )

    return output.flatten(-2)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    assert d_model % num_heads == 0

    head_dim = d_model // num_heads

    q = torch.matmul(in_features, q_proj_weight.T)
    k = torch.matmul(in_features, k_proj_weight.T)
    v = torch.matmul(in_features, v_proj_weight.T)

    *leading_dims, seq_len, _ = q.shape

    q = q.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    k = k.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    v = v.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    attention_scores = torch.matmul(
        q,
        k.transpose(-2,-1),
    )

    attention_scores = attention_scores / (head_dim ** 0.5)

    casual_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=in_features.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    attention_scores = attention_scores.masked_fill(
        casual_mask,
        float("-inf"),
    )

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1,
    )

    output = torch.matmul(
    attention_weights,
    v,
)

# (..., seq_len, num_heads, head_dim)
    output = output.transpose(-3, -2).contiguous()

    # (..., seq_len, d_model)
    output = output.reshape(
        *leading_dims,
        seq_len,
        d_model,
    )

    # 输出投影
    output = torch.matmul(
        output,
        o_proj_weight.T,
    )

    return output

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    assert d_model % num_heads == 0

    head_dim = d_model // num_heads

    # (..., sequence_length, d_model)
    q = torch.matmul(in_features, q_proj_weight.T)
    k = torch.matmul(in_features, k_proj_weight.T)
    v = torch.matmul(in_features, v_proj_weight.T)

    *leading_dims, seq_len, _ = q.shape

    assert seq_len <= max_seq_len

    q = q.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    k = k.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    v = v.reshape(
        *leading_dims,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(-3, -2)

    if token_positions is None:
        token_positions = torch.arange(
            seq_len,
            device=in_features.device,
        )

        token_positions = token_positions.expand(
            *leading_dims,
            seq_len,
        )

    q = _apply_rope(
        q,
        token_positions,
        theta,
    )

    k = _apply_rope(
        k,
        token_positions,
        theta,
    )

    attention_scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    )

    attention_scores = attention_scores / (head_dim ** 0.5)

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=in_features.device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    attention_scores = attention_scores.masked_fill(
        causal_mask,
        float("-inf"),
    )

    attention_weights = torch.softmax(
        attention_scores,
        dim=-1,
    )

    output = torch.matmul(
        attention_weights,
        v,
    )

    output = output.transpose(-3, -2).contiguous()

    output = output.reshape(
        *leading_dims,
        seq_len,
        d_model,
    )

    return torch.matmul(output, o_proj_weight.T)