import torch
from torch import Tensor
from .nn import run_rmsnorm, run_swiglu

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

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    batch_size, seq_len, feature_dim = in_features.shape

    assert feature_dim == d_model
    assert d_model % num_heads == 0
    assert seq_len <= max_seq_len

    eps = 1e-5

    x = in_features

    # 第一个 RMSNorm
    normalized_x = run_rmsnorm(
        d_model,
        eps,
        weights["ln1.weight"],
        x,
    )

    token_positions = torch.arange(
        seq_len,
        device=x.device,
    ).unsqueeze(0).expand(
        batch_size,
        seq_len,
    )

    # Multi-Head Attention，内部使用 RoPE 和 causal mask
    attention_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normalized_x,
        token_positions=token_positions,
    )

    # 第一次残差连接
    x = x + attention_output

    # 第二个 RMSNorm
    normalized_x = run_rmsnorm(
        d_model,
        eps,
        weights["ln2.weight"],
        x,
    )

    # 复用已经实现的 SwiGLU
    ffn_output = run_swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        normalized_x,
    )

    # 第二次残差连接
    x = x + ffn_output

    return x

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:

    batch_size, sequence_length = in_indices.shape

    assert in_indices.ndim == 2
    assert in_indices.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    if sequence_length > context_length:
        in_indices = in_indices[:, :context_length]
        sequence_length = context_length

    x = weights["token_embeddings.weight"][in_indices]

    for layer_idx in range(num_layers):
        layer_weights = {
            "attn.q_proj.weight": weights[
                f"layers.{layer_idx}.attn.q_proj.weight"
            ],
            "attn.k_proj.weight": weights[
                f"layers.{layer_idx}.attn.k_proj.weight"
            ],
            "attn.v_proj.weight": weights[
                f"layers.{layer_idx}.attn.v_proj.weight"
            ],
            "attn.output_proj.weight": weights[
                f"layers.{layer_idx}.attn.output_proj.weight"
            ],
            "ln1.weight": weights[
                f"layers.{layer_idx}.ln1.weight"
            ],
            "ffn.w1.weight": weights[
                f"layers.{layer_idx}.ffn.w1.weight"
            ],
            "ffn.w2.weight": weights[
                f"layers.{layer_idx}.ffn.w2.weight"
            ],
            "ffn.w3.weight": weights[
                f"layers.{layer_idx}.ffn.w3.weight"
            ],
            "ln2.weight": weights[
                f"layers.{layer_idx}.ln2.weight"
            ],
        }

        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=layer_weights,
            in_features=x,
        )

    x = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=x,
    )

    logits = torch.matmul(
        x,
        weights["lm_head.weight"].T,
    )

    return logits