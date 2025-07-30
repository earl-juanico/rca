from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_llama import CustomLlamaConfig

try:
    from apex.megatron_layer_norm import MixedFusedLayerNorm as LayerNorm
except ImportError:
    from torch.nn import LayerNorm

USE_FLASH_ATTN = False
try:
    import flash_attn
    if version.parse(flash_attn.__version__) >= version.parse("2.1.0"):
        USE_FLASH_ATTN = True
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass

logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = (attention_mask).sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0,
                       dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def get_norm(config: CustomLlamaConfig):
    norm_type = config.norm_type
    if norm_type == 'rms_norm':
        return partial(RMSNorm, eps=config.layernorm_epsilon)
    elif norm_type == 'layer_norm':
        return partial(LayerNorm, eps=config.layernorm_epsilon)
    else:
        raise ValueError(f'Unsupported norm type: {norm_type}')


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len),
                      torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, compress=1.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.compress = compress

    def forward(self, x, seq_len):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            self.inv_freq = self.inv_freq.to(x.device)
            t = (
                torch.arange(seq_len, device=self.inv_freq.device,
                             dtype=self.inv_freq.dtype)
                * self.compress
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


# rotary pos emb helpers:


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (
        cos[..., offset: q.shape[-2] + offset, :],
        sin[..., offset: q.shape[-2] + offset, :],
    )
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_torch(
    q, k, cos, sin, offset: int = 0
):  # jitting fails with bf16
    cos, sin = (
        cos[..., offset: q.shape[-2] + offset, :],
        sin[..., offset: q.shape[-2] + offset, :],
    )
    q_embed = (q.float() * cos) + (rotate_half(q.float()) * sin)
    k_embed = (k.float() * cos) + (rotate_half(k.float()) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class CustomLlamaAttention(nn.Module):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.max_positions = config.max_position_embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=config.rotary_emb_base,
            compress=config.rotary_compress,
        )
        self.norm_factor = torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        if self.use_gqa:
            self.query_dense = nn.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=getattr(config, "qkv_proj_bias", True)
            )
            self.key_value_dense = nn.Linear(
                config.hidden_size,
                self.head_size * 2 * config.num_kv_heads,
                bias=getattr(config, "qkv_proj_bias", True),
            )
        else:
            self.query_key_value = nn.Linear(
                config.hidden_size,
                3 * config.hidden_size,
                bias=getattr(config, "qkv_proj_bias", True),
            )

        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=getattr(config, "out_proj_bias", True),
        )
        self.apply_rotary_fn = (
            apply_rotary_pos_emb_torch
            if config.torch_dtype == torch.bfloat16
            else apply_rotary_pos_emb
        )

    @property
    def use_gqa(self):
        return self.num_kv_heads < self.num_attention_heads

    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        has_layer_past = layer_past is not None

        if self.use_gqa:
            # Compute Q
            # [batch, seq_len, hidden_size] --> [batch_size, seq_len, (num_heads * head_size)]
            q = self.query_dense(hidden_states)

            # [batch_size, seq_len, (num_heads * head_size)]
            #     --> [batch, seq_len, num_attention_heads, head_size]
            new_q_shape = q.size()[:-1] + \
                (self.num_attention_heads, self.head_size)
            q = q.view(*new_q_shape)

            # Compute KV
            # [batch, seq_len, hidden_size] --> [batch_size, seq_len, (num_attention_groups * 2 * head_size)]
            kv = self.key_value_dense(hidden_states)

            # [batch, seq_len, (num_attention_groups * 2 * head_size)]
            #   --> [batch, seq_len, num_attention_groups, 2 * head_size]
            new_kv_shape = kv.size()[:-1] + (
                self.num_kv_heads,
                2 * self.head_size,
            )
            kv = kv.view(*new_kv_shape)

            # [batch, num_attention_heads, seq_len, head_size]
            query = q.permute(0, 2, 1, 3)
            # [batch, num_attention_groups, seq_len, head_size]
            key = kv[..., : self.head_size].permute(0, 2, 1, 3)
            value = kv[..., self.head_size:].permute(0, 2, 1, 3)
        else:
            # Compute QKV
            # Attention heads [batch, seq_len, hidden_size]
            #   --> [batch, seq_len, (np * 3 * head_size)]
            qkv = self.query_key_value(hidden_states)

            # [batch, seq_len, (num_heads * 3 * head_size)]
            #   --> [batch, seq_len, num_heads, 3 * head_size]
            new_qkv_shape = qkv.size()[:-1] + (
                self.num_attention_heads,
                3 * self.head_size,
            )
            qkv = qkv.view(*new_qkv_shape)

            # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
            key = qkv[..., self.head_size: 2 *
                      self.head_size].permute(0, 2, 1, 3)
            value = qkv[..., 2 * self.head_size:].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims:]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims:]

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = self.apply_rotary_fn(
            query_rot, key_rot, cos, sin, offset=offset)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        if USE_FLASH_ATTN:
            # Compute attention
            attn_output, attn_weights = self._flash_attn(
                query, key, value, attention_mask, head_mask
            )

            # from [batch_size, ]
            attn_output = attn_output.reshape(
                attn_output.size(0), attn_output.size(1), self.hidden_size).contiguous()
        else:
            # Compute attention
            attn_output, attn_weights = self._attn(
                query, key, value, attention_mask, head_mask
            )

            # Reshape outputs
            attn_output = self._merge_heads(
                attn_output, self.num_attention_heads, self.head_size
            )
        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(
            tensor.size(0), tensor.size(
                1), num_attention_heads * attn_head_size
        )
        # -> [bs, seq_len, hidden_size]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q: [bs, num_attention_heads, seq_len, attn_head_size]
        # k,v: [bs, num_attention_groups, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        _, num_attention_groups, key_length, _ = key.size()

        group_size = num_attention_heads // num_attention_groups

        if not self.use_gqa:
            assert group_size == 1

        # repeat key and value, so we can use normal MHA algorithm
        key = (
            key.view(batch_size, num_attention_groups,
                     1, key_length, attn_head_size)
            .repeat(1, 1, group_size, 1, 1)
            .view(batch_size, num_attention_heads, key_length, attn_head_size)
        )
        value = (
            value.view(batch_size, num_attention_groups,
                       1, key_length, attn_head_size)
            .repeat(1, 1, group_size, 1, 1)
            .view(batch_size, num_attention_heads, key_length, attn_head_size)
        )

        query = query.view(
            batch_size * num_attention_heads, query_length, attn_head_size
        )
        key = key.view(batch_size * num_attention_heads,
                       key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(
                torch.tensor(
                    1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device
                )
                / self.norm_factor
            ),
        )
        attn_scores = attn_scores.view(
            batch_size, num_attention_heads, query_length, key_length
        )

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(
            attn_scores.device
        )

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _flash_attn(self, query, key, value, attention_mask=None, head_mask=None):
        assert head_mask is None, "head_mask is not supported in _flash_attn"
        # q: [bs, num_attention_heads, seq_len, attn_head_size]
        # k,v: [bs, num_attention_groups, seq_len, attn_head_size]

        # flash_attn need the layout to be [batch_size, sequence_length, num_heads, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query_length = query.size(1)
        causal = query_length != 1

        if attention_mask is not None:
            batch_size = query.size(0)
            query, key, value, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query, key, value, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query, key, value, 0, causal=causal
            )

        return attn_output, None

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        num_attention_heads = query_layer.shape[2]

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len,
                              num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len,
                                num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    num_attention_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


def swiglu(x):
    x1, x2 = x.chunk(2, dim=(x.ndim - 1))
    return x1 * torch.nn.functional.silu(x2)


def get_activation(act_name: str):
    if act_name == "gelu":
        return ACT2FN["gelu_new"]
    elif act_name == "swiglu":
        return swiglu
    else:
        return ACT2FN[act_name]


class CustomLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h_to_4h_out_channels = (
            config.ffn_hidden_size * 2
            if config.hidden_act == "swiglu"
            else config.ffn_hidden_size
        )
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            h_to_4h_out_channels,
            bias=getattr(config, "mlp_fc1_bias", True)
        )
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=getattr(config, "mlp_fc2_bias", True)
        )
        self.act = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class CustomLlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        norm_func = get_norm(config)
        self.input_layernorm = norm_func(config.hidden_size)
        self.post_attention_layernorm = norm_func(config.hidden_size)
        self.attention = CustomLlamaAttention(config)
        self.mlp = CustomLlamaMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        attn_in = self.input_layernorm(hidden_states)
        attention_layer_outputs = self.attention(
            attn_in,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[
            0
        ]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]
        # pseudocode:
        # x = x + attn(ln1(x))
        # x = x + mlp(ln2(x))
        attn_output = attn_output + hidden_states
        mlp_input = self.post_attention_layernorm(attn_output)
        mlp_output = self.mlp(mlp_input)
        hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (
                hidden_states,
            ) + outputs  # hidden_states, present, (attn_weights)
        else:
            # hidden_states, (attn_weights)
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class CustomLlamaPreTrainedModel(PreTrainedModel):
    config_class = CustomLlamaConfig
    base_model_prefix = "lm"
    _no_split_modules = ["CustomLlamaLayer"]


class CustomLlamaModel(CustomLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CustomLlamaLayer(config) for _ in range(config.num_layers)]
        )

        norm_func = get_norm(config)
        self.final_layer_norm = norm_func(config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        else:
            past_key_values = tuple([None] * self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)
        # Attention mask.
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_layers x num_heads]
        # and head_mask is converted to shape [num_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if USE_FLASH_ATTN:
            attention_mask = attention_mask if (
                attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + \
                    (outputs[2 if use_cache else 1],)

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CustomLlamaForCausalLM(CustomLlamaPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]
    _keys_to_ignore_on_load_unexpected = [
        r"lm.layers.\d+.attention.rotary_emb.inv_freq"
    ]

    def __init__(self, config):
        super().__init__(config)

        self.lm = CustomLlamaModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.

            Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
            `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **model_kwargs
    ):
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past
