import torch
import torch.nn as nn

class CustomAttention(nn.Module):
    def __init__(self, dropout_rate, causal=False):
        super(CustomAttention, self).__init__()
        self.causal = causal
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.register_buffer('causal_mask', None, persistent=False)

    def _generate_causal_mask(self, seq_len, device):
        if self.causal_mask is not None and self.causal_mask.size(0) >= seq_len:
            return self.causal_mask[:seq_len, :seq_len]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask, persistent=False)
        return mask

    def forward(self, query, key, value, attn_mask=None):
        batch_size, num_heads, seq_len, head_dim = query.size()
        scale = head_dim ** -0.5

        if key.dim() == 3:
            key = key.unsqueeze(1).expand(-1, num_heads, -1, -1)
            value = value.unsqueeze(1).expand(-1, num_heads, -1, -1)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        if self.causal:
            causal_mask = self._generate_causal_mask(seq_len, query.device)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)
        output = torch.matmul(attn_probs, value)
        return output

