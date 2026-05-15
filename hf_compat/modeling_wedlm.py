# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch WeDLM model."""

from typing import Optional, Tuple, Union, Dict, List, Callable

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import check_model_inputs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Import attention-related utilities
try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    FlashAttentionKwargs = dict

try:
    from transformers.integrations.flash_attention import ALL_ATTENTION_FUNCTIONS
except ImportError:
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        ALL_ATTENTION_FUNCTIONS = {}

from .configuration_wedlm import WeDLMConfig

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ============================================================================
# Core Components (self-contained, no Qwen2 dependency)
# ============================================================================

class WeDLMMLP(nn.Module):
    """WeDLM MLP module with SwiGLU activation."""
    
    def __init__(self, config: WeDLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class WeDLMRMSNorm(nn.Module):
    """WeDLM RMSNorm, equivalent to T5LayerNorm."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class WeDLMRotaryEmbedding(nn.Module):
    """WeDLM Rotary Position Embedding."""
    
    def __init__(self, config: WeDLMConfig, device=None):
        super().__init__()
        # Determine rope_type from config
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
        else:
            self.rope_type = "default"
        
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        
        # Get initialization function
        if self.rope_type == "default":
            inv_freq, self.attention_scaling = self._compute_default_rope_parameters(config, device)
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(config, device)
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def _compute_default_rope_parameters(
        config: WeDLMConfig,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies for default RoPE.
        
        Args:
            config: Model configuration
            device: Device to place the tensors on
            
        Returns:
            Tuple of (inv_freq tensor, attention_scaling factor)
        """
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        attention_factor = 1.0
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary position embeddings.
        
        Args:
            x: Input tensor, used for dtype and device
            position_ids: Position indices
            
        Returns:
            Tuple of (cos, sin) tensors
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        
        # Force float32 computation for numerical stability
        with torch.amp.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ============================================================================
# Attention Utilities
# ============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: Optional[torch.Tensor] = None, 
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key/value heads to match the number of query heads (for GQA).
    
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager (standard) attention implementation."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ============================================================================
# Attention Layer
# ============================================================================

class WeDLMAttention(nn.Module):
    """
    WeDLM Attention module.
    
    Supports both:
    - Qwen2.5 style: with QKV bias, no QK Norm
    - Qwen3 style: configurable QKV bias, with QK Norm
    """

    def __init__(self, config: WeDLMConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        # Support configurable attention_bias (Qwen2.5: True, Qwen3: False by default)
        attention_bias = getattr(config, "attention_bias", True)
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Support optional QK Norm (Qwen3 feature)
        self.qk_norm = getattr(config, "qk_norm", False)
        if self.qk_norm:
            self.q_norm = WeDLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = WeDLMRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.qk_norm:
            # Qwen3 style: apply norm after projection, before transpose
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            # Qwen2 style: no norm
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention implementation
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager" and self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ============================================================================
# Decoder Layer
# ============================================================================

class WeDLMDecoderLayer(GradientCheckpointingLayer):
    """WeDLM Decoder Layer with pre-norm architecture."""
    
    def __init__(self, config: WeDLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = WeDLMAttention(config=config, layer_idx=layer_idx)
        self.mlp = WeDLMMLP(config)
        self.input_layernorm = WeDLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = WeDLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: Input tensor of shape `(batch, seq_len, embed_dim)`
            attention_mask: Attention mask of size `(batch, sequence_length)`
            position_ids: Position indices
            past_key_values: Cached past key and value projection states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV cache
            cache_position: Position in the cache
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Feed Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


# ============================================================================
# Model Classes
# ============================================================================

@auto_docstring
class WeDLMPreTrainedModel(PreTrainedModel):
    """Base class for WeDLM models."""
    
    config_class = WeDLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WeDLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": WeDLMDecoderLayer,
        "attentions": WeDLMAttention,
    }


@auto_docstring
class WeDLMModel(WeDLMPreTrainedModel):
    """
    WeDLM base model outputting raw hidden states.
    """
    
    def __init__(self, config: WeDLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [WeDLMDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WeDLMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = WeDLMRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Prepare attention masks
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@auto_docstring
class WeDLMForCausalLM(WeDLMPreTrainedModel, GenerationMixin):
    """
    WeDLM Model for Causal Language Modeling with WeDLM block decoding support.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: WeDLMConfig):
        super().__init__(config)
        self.model = WeDLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _efficient_reorder_sequence(
        self, 
        tokens: torch.Tensor, 
        mask_indices: torch.Tensor, 
        position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function to reorder sequence by moving MASK parts to the end.
        """
        reordered_tokens = torch.cat((tokens[~mask_indices], tokens[mask_indices]))
        reordered_position_ids = torch.cat((position_ids[~mask_indices], position_ids[mask_indices]))
        return reordered_tokens, reordered_position_ids

    @torch.no_grad()
    def _generate_one_block(
        self,
        prefix_ids: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        block_size: int,
        mask_token_id: int,
        confidence_threshold: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate one block of content based on the given prefix.
        
        Args:
            prefix_ids: Current sequence token IDs
            prefix_position_ids: Position IDs for current sequence
            block_size: Number of tokens to generate in this block
            mask_token_id: Token ID for MASK token
            confidence_threshold: Minimum confidence to accept a prediction
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter (unused currently)
            top_k: Top-k sampling parameter (unused currently)
            
        Returns:
            Tuple of (updated_ids, updated_position_ids, block_statistics)
        """
        device = prefix_ids.device
        
        # 1. Append a block of MASK tokens after the current prefix
        mask_tensor = torch.full((block_size,), mask_token_id, dtype=torch.long, device=device)
        current_ids = torch.cat([prefix_ids, mask_tensor])
        
        # Create position encodings for the newly added MASKs
        start_pos = prefix_position_ids[-1].item() + 1 if len(prefix_position_ids) > 0 else 0
        mask_position_ids = torch.arange(start_pos, start_pos + block_size, dtype=torch.long, device=device)
        original_position_ids = torch.cat([prefix_position_ids, mask_position_ids])
        
        # Mark which positions are MASK
        is_mask = (current_ids == mask_token_id)
        
        # Statistics
        block_stats = {
            'steps': 0,
            'tokens_generated': 0,
            'tokens_per_step': [],
            'max_confidences': [],
        }
        
        # 2. WeDLM iteration within the block
        for step in range(block_size):
            if not is_mask.any():
                break
            
            block_stats['steps'] += 1
            
            # 2.1 Reorder sequence
            reordered_ids, reordered_position_ids = self._efficient_reorder_sequence(
                current_ids, is_mask, original_position_ids
            )
            
            # 2.2 Prepare input
            input_ids = reordered_ids.unsqueeze(0)
            position_ids = reordered_position_ids.unsqueeze(0)
            
            seq_len = input_ids.shape[1]
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
            
            # 2.3 Model forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
            
            # 2.4 Get logits for MASK positions
            num_non_mask = (~is_mask).sum().item()
            mask_logits = logits[0, num_non_mask:]
            
            if mask_logits.size(0) == 0:
                break
            
            mask_logits = mask_logits / temperature
            probs = F.softmax(mask_logits, dim=-1)
            max_probs, predicted_ids = probs.max(dim=-1)
            
            block_stats['max_confidences'].append(max_probs.max().item())
            
            # 2.5 Select positions to fill
            if confidence_threshold > 0.0:
                above_threshold_mask = max_probs >= confidence_threshold
                
                if above_threshold_mask.any():
                    indices_to_fill = above_threshold_mask.nonzero(as_tuple=True)[0]
                    num_tokens_this_step = len(indices_to_fill)
                else:
                    best_idx = max_probs.argmax()
                    indices_to_fill = best_idx.unsqueeze(0)
                    num_tokens_this_step = 1
            else:
                best_idx = max_probs.argmax()
                indices_to_fill = best_idx.unsqueeze(0)
                num_tokens_this_step = 1
            
            block_stats['tokens_per_step'].append(num_tokens_this_step)
            block_stats['tokens_generated'] += num_tokens_this_step
            
            # 2.6 Update all selected positions
            for idx in indices_to_fill:
                idx_item = idx.item()
                best_token_id = predicted_ids[idx_item].item()
                
                best_pos_in_reordered = num_non_mask + idx_item
                original_pos_value = reordered_position_ids[best_pos_in_reordered].item()
                original_pos_in_seq = (original_position_ids == original_pos_value).nonzero(as_tuple=True)[0].item()
                
                current_ids[original_pos_in_seq] = best_token_id
                is_mask[original_pos_in_seq] = False
        
        return current_ids, original_position_ids, block_stats

    @torch.no_grad()
    def generate_wedlm(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        block_size: int,
        mask_token_id: Optional[int] = None,
        confidence_threshold: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        pad_token_id: Optional[int] = None,
        return_stats: bool = True,
        **kwargs
    ) -> Union[torch.LongTensor, Dict]:
        """
        Generate text using WeDLM block decoding mode.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            block_size: Number of tokens to generate per block
            mask_token_id: Token ID for MASK token
            confidence_threshold: Minimum confidence to accept predictions (0.0-1.0)
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            pad_token_id: Token ID for padding
            return_stats: Whether to return generation statistics
            
        Returns:
            If return_stats=False: Generated token sequences
            If return_stats=True: Dict with 'sequences' and 'stats'
        """
        if mask_token_id is None:
            mask_token_id = getattr(self.config, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError("mask_token_id must be provided or set in config")
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        
        logger.debug(
            f"Starting WeDLM generation: max_new_tokens={max_new_tokens}, block_size={block_size}, "
            f"confidence_threshold={confidence_threshold}, num_blocks={num_blocks}"
        )
        
        all_generated = []
        all_sample_stats = []
        
        for batch_idx in range(batch_size):
            sample_ids = input_ids[batch_idx]
            if pad_token_id is not None:
                pad_mask = (sample_ids != pad_token_id)
                if pad_mask.any():
                    valid_length = pad_mask.sum().item()
                    prefix_ids = sample_ids[:valid_length]
                else:
                    prefix_ids = sample_ids
            else:
                prefix_ids = sample_ids
            
            prefix_length = prefix_ids.shape[0]
            current_position_ids = torch.arange(prefix_length, dtype=torch.long, device=device)
            
            current_ids = prefix_ids.clone()
            
            sample_stats = {
                'input_length': prefix_length,
                'total_steps': 0,
                'total_tokens_generated': 0,
                'blocks': [],
            }
            
            for block_idx in range(num_blocks):
                remaining_tokens = max_new_tokens - block_idx * block_size
                current_block_size = min(block_size, remaining_tokens)
                
                logger.debug(
                    f"Batch {batch_idx}, Block {block_idx}/{num_blocks}: "
                    f"generating {current_block_size} tokens"
                )
                
                current_ids, current_position_ids, block_stats = self._generate_one_block(
                    prefix_ids=current_ids,
                    prefix_position_ids=current_position_ids,
                    block_size=current_block_size,
                    mask_token_id=mask_token_id,
                    confidence_threshold=confidence_threshold,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                
                sample_stats['total_steps'] += block_stats['steps']
                sample_stats['total_tokens_generated'] += block_stats['tokens_generated']
                sample_stats['blocks'].append(block_stats)
            
            sample_stats['actual_tokens_generated'] = len(current_ids) - prefix_length
            sample_stats['output_length'] = len(current_ids)
            
            all_generated.append(current_ids)
            all_sample_stats.append(sample_stats)
        
        max_length = max(seq.shape[0] for seq in all_generated)
        padded_sequences = []
        
        for seq in all_generated:
            if seq.shape[0] < max_length:
                padding = torch.full(
                    (max_length - seq.shape[0],),
                    pad_token_id if pad_token_id is not None else 0,
                    dtype=torch.long,
                    device=device
                )
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        result_sequences = torch.stack(padded_sequences, dim=0)
        
        total_steps = sum(s['total_steps'] for s in all_sample_stats)
        total_tokens = sum(s['total_tokens_generated'] for s in all_sample_stats)
        avg_tokens_per_step = total_tokens / total_steps if total_steps > 0 else 0
        
        logger.debug(
            f"WeDLM generation completed: "
            f"total_steps={total_steps}, "
            f"total_tokens_generated={total_tokens}, "
            f"avg_tokens_per_step={avg_tokens_per_step:.2f}"
        )
        
        if not return_stats:
            return result_sequences
        
        return {
            'sequences': result_sequences,
            'stats': {
                'total_steps': total_steps,
                'total_tokens_generated': total_tokens,
                'average_tokens_per_step': avg_tokens_per_step,
                'efficiency_ratio': total_tokens / total_steps if total_steps > 0 else 0,
                'per_sample_stats': all_sample_stats,
                'config': {
                    'batch_size': batch_size,
                    'max_new_tokens': max_new_tokens,
                    'block_size': block_size,
                    'confidence_threshold': confidence_threshold,
                    'temperature': temperature,
                }
            }
        }

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, DynamicCache) and attention_mask.ndim == 2:
            model_inputs["cache_position"] = cache_position
            model_inputs["past_key_values"] = past_key_values
            model_inputs["use_cache"] = use_cache
            model_inputs["position_ids"] = position_ids
            model_inputs["attention_mask"] = attention_mask
            return model_inputs

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


__all__ = [
    "WeDLMConfig",
    "WeDLMPreTrainedModel", 
    "WeDLMModel",
    "WeDLMForCausalLM",
]