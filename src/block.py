import torch
import torch.nn as nn
from einops import rearrange

from utils import LayerNorm, SwiGLU, norm, apply_rotary
from res import Residual
from attn import CustomAttention
from rotembd import RotaryEmbedding

class ParallelTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        head_dim=64,
        causal=True,
        num_heads=8,
        use_qk_rmsnorm=False,
        qk_norm_scale=8,
        feedforward_multiplier=4,
        attention_dropout_rate=0.0,
        feedforward_dropout_rate=0.0,
        xpos_enabled=True,
        xpos_scale=512,
    ):
        super().__init__()

        self.norm_layer = LayerNorm(embedding_dim)

        attention_inner_dim = head_dim * num_heads
        feedforward_inner_dim = embedding_dim * feedforward_multiplier
        self.combined_dims = (attention_inner_dim, head_dim, head_dim, (feedforward_inner_dim * 2))

        self.use_qk_rmsnorm = use_qk_rmsnorm
        if use_qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(head_dim))
            self.k_scale = nn.Parameter(torch.ones(head_dim))

        self.attention = CustomAttention(
            causal=causal,
            dropout_rate=attention_dropout_rate
        )

        self.num_heads = num_heads
        self.attention_scale = (head_dim ** -0.5) if not use_qk_rmsnorm else qk_norm_scale
        self.is_causal = causal

        self.rotary_pos_emb = RotaryEmbedding(head_dim, scale_base=xpos_scale, use_xpos=xpos_enabled and causal)

        self.fused_attention_feedforward = nn.Linear(embedding_dim, sum(self.combined_dims), bias=False)

        self.attn_output_proj = nn.Linear(attention_inner_dim, embedding_dim, bias=False)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)

        self.feedforward_net = nn.Sequential(
            SwiGLU(),
            nn.Dropout(feedforward_dropout_rate),
            nn.Linear(feedforward_inner_dim, embedding_dim, bias=False)
        )

        self.register_buffer("pos_embedding", None, persistent=False)
        self.register_buffer("pos_scale", None, persistent=False)

    def _get_rotary_embeddings(self, seq_len, device):
        if self.pos_embedding is not None and self.pos_embedding.shape[-2] >= seq_len:
            return self.pos_embedding[:seq_len], self.pos_scale[:seq_len]

        position_embeddings, scale = self.rotary_pos_emb(seq_len, device=device)
        self.register_buffer("pos_embedding", position_embeddings, persistent=False)
        self.register_buffer("pos_scale", scale, persistent=False)
        return position_embeddings, scale

    def forward(self, inputs, attention_mask=None, finetune_modules=None):
        batch_size, sequence_length, _ = inputs.shape
        device, num_heads = inputs.device, self.num_heads

        normalized_input = self.norm_layer(inputs)

        query, key, value, feedforward = self.fused_attention_feedforward(normalized_input).split(self.combined_dims, dim=-1)

        if finetune_modules is not None:
            lora_q, lora_k, lora_v, lora_out = finetune_modules
            query += lora_q(inputs)
            key += lora_k(inputs)
            value += lora_v(inputs)

        query = rearrange(query, "batch seq (heads dim) -> batch heads seq dim", heads=num_heads)

        if self.use_qk_rmsnorm:
            query, key = map(norm, (query, key))
            query *= self.q_scale
            key *= self.k_scale

        pos_emb, scale = self._get_rotary_embeddings(sequence_length, device)
        query = apply_rotary(pos_emb, query, scale)
        key = apply_rotary(pos_emb, key, scale ** -1)

        attention_output = self.attention(query, key, value, attn_mask=attention_mask)
        attention_output = rearrange(attention_output, "batch heads seq dim -> batch seq (heads dim)")

        attn_output = self.attn_output_proj(attention_output)

        ff_output = self.feedforward_net(feedforward)

        if finetune_modules is not None:
            attn_output += lora_out(attention_output)

        return attn_output + ff_output

