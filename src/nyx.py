import torch
import torch.nn as nn

from res import Residual
from utils import LayerNorm, top_k, is_present, gumbel_sample, identity, evaluation_mode_decorator, use_default
from block import ParallelTransformerLayer
from lora import LowRankAdapter

from einops import rearrange, pack, unpack
from tqdm import tqdm

from pathlib import Path
from itertools import zip_longest

class NYX(nn.Module):
    def __init__(
        self,
        *,
        model_dim,
        vocab_size,
        layer_depth,
        use_causal_mask=True,
        head_dimension=64,
        num_attention_heads=8,
        feedforward_scale=4,
        attention_dropout_rate=0.,
        feedforward_dropout_rate=0.,
        use_qk_rmsnorm=False,
        lora_rank=8,
        rotary_pos_emb_scale=512,
        finetune_contexts=tuple(),
        loss_ignore_index=0
    ):
        super().__init__()
        self.model_dim = model_dim
        self.head_dimension = head_dimension
        self.num_attention_heads = num_attention_heads
        self.use_causal_mask = use_causal_mask
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.layers = nn.ModuleList([])

        for _ in range(layer_depth):
            transformer_block = Residual(ParallelTransformerLayer(
                embedding_dim=model_dim,            
                head_dim=head_dimension,            
                causal=use_causal_mask,
                num_heads=num_attention_heads,      
                use_qk_rmsnorm=use_qk_rmsnorm,      
                feedforward_multiplier=feedforward_scale,   
                attention_dropout_rate=attention_dropout_rate,  
                feedforward_dropout_rate=feedforward_dropout_rate,  
                xpos_enabled=True,                 
                xpos_scale=rotary_pos_emb_scale    
            ))

            self.layers.append(transformer_block)

        self.norm_layer = LayerNorm(model_dim)
        self.to_output_logits = nn.Linear(model_dim, vocab_size, bias=False)
        
        self.to_output_logits.weight = self.token_embedding.weight

        nn.init.normal_(self.token_embedding.weight, std=0.02)

        self.lora_rank = lora_rank
        self.finetune_contexts = nn.ModuleDict({})

        for context in finetune_contexts:
            self.register_finetune_params(context)

        self.loss_ignore_index = loss_ignore_index

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        path = Path(path)
        self.load_state_dict(torch.load(str(path)))

    def _dropout(self, dropout):
        for module in self.layers.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
        return self

    def register_finetune_params(self, finetune_scope, lora_rank=None):
        assert finetune_scope not in self.finetune_modules, f'Finetune scope {finetune_scope} already exists'
        
        model_dim, head_dim, num_heads, rank_value, model_device = self.dim, self.dim_head, self.heads, use_default(lora_rank, self.lora_r), self.device
        
        query_dim = num_heads * head_dim
        key_value_dim = head_dim

        lora_param_layers = nn.ModuleList([])

        for _ in range(len(self.layers)):
            lora_param_layers.append(nn.ModuleList([
                LowRankAdapter(model_dim, query_dim, rank=rank_value),
                LowRankAdapter(model_dim, key_value_dim, rank=rank_value),
                LowRankAdapter(model_dim, key_value_dim, rank=rank_value),
                LowRankAdapter(query_dim, model_dim, rank=rank_value)
            ]))

        self.finetune_modules[finetune_scope] = lora_param_layers.to(model_device)

    def remove_finetune_params(self, scope):
        assert scope in self.finetune_modules, f'Scope {scope} not found'
        return self.finetune_modules.pop(scope)

    @torch.no_grad()
    def integrate_finetune_weights(self, finetune_scope):
        
        assert finetune_scope in self.finetune_modules, f'Scope {finetune_scope} not found'

        lora_layers = self.finetune_modules.pop(finetune_scope)

        for model_layer, (lora_q_proj, lora_k_proj, lora_v_proj, lora_o_proj) in zip(self.layers, lora_layers):
            layer_block = model_layer.fn

            attn_proj_weights = layer_block.fused_attn_ff_proj.weight
            attn_output_weights = layer_block.attn_out.weight

            projection_dim = attn_proj_weights.shape[0]

            merged_qkv_weights, _ = pack([lora_q_proj.weight, lora_k_proj.weight, lora_v_proj.weight], 'i *')
            merged_qkv_weights = nn.functional.pad(merged_qkv_weights, (0, projection_dim - merged_qkv_weights.shape[1]))

            merged_qkv_weights = rearrange(merged_qkv_weights, 'i o -> o i')
            lora_output_weights = rearrange(lora_o_proj.weight, 'i o -> o i')

            attn_proj_weights.add_(merged_qkv_weights)
            attn_output_weights.add_(lora_output_weights)

    def NYX_parameters(self):
        return set(self.parameters()) - set(self.finetune_modules.parameters())

    def finetune_parameters(self, scope = 'use_default'):
        assert scope in self.finetune_modules, f'finetune parameters of scope {scope} not found'
        return self.finetune_modules[scope].parameters()

    @torch.no_grad()
    @evaluation_mode_decorator
    def generate_sequence(
        self,
        max_seq_len,
        initial_prompt=None,
        temp=1.,
        logits_filter_fn=top_k,
        filter_threshold=0.9,
        pad_token_value=0.,
        end_token=None,
        return_generated_only=True,
        show_progress=False,
        **extra_args
    ):
        if not is_present(initial_prompt):
            initial_prompt = torch.randint(0, self.vocab_size, (1, 1))
            initial_prompt = initial_prompt.to(self.device)
            return_generated_only = False

        initial_prompt, shape_info = pack([initial_prompt], '* n')
        prompt_len, generated_output = initial_prompt.shape[-1], initial_prompt.clone()

        progress_wrapper = identity if not show_progress else tqdm
        num_sampling_steps = max(1, max_seq_len - initial_prompt.shape[-1])

        for _ in progress_wrapper(range(num_sampling_steps)):
            logits, embeddings = self.forward(generated_output, return_logits_and_embeddings=True, **extra_args)
            logits, embeddings = logits[:, -1], embeddings[:, -1]

            if is_present(logits_filter_fn):
                logits = logits_filter_fn(logits, thres=filter_threshold)

            next_token = gumbel_sample(logits, temperature=temp, dim=-1)
            generated_output, _ = pack([generated_output, next_token], 'b *')

            if is_present(end_token):
                eos_detected = (generated_output == end_token)

                if eos_detected.any(dim=-1).all():
                    shifted_eos_detected = nn.functional.pad(eos_detected, (1, -1))
                    stop_mask = shifted_eos_detected.float().cumsum(dim=-1) >= 1
                    generated_output = generated_output.masked_fill(stop_mask, pad_token_value)
                    break

        generated_output, = unpack(generated_output, shape_info, '* n')

        if not return_generated_only:
            return generated_output

        return generated_output[..., prompt_len:]

    def forward(
        self,
        input_data,
        compute_loss=False,
        deactivate_lora=False,
        finetune_context=None,
        additional_embedding=None,
        return_embeddings_only=False,
        return_logits_and_embeddings=False
    ):
        if compute_loss:
            input_data, target_labels = input_data[:, :-1], input_data[:, 1:]

        if not self.use_causal_mask:
            valid_mask = input_data >= 0
            input_data = input_data.masked_fill(~valid_mask, 0)
        else:
            valid_mask = None

        input_data = self.token_embedding(input_data)

        if is_present(additional_embedding):
            input_data = input_data + additional_embedding

        finetune_adapters = tuple()
        if is_present(finetune_context) and not deactivate_lora:
            assert finetune_context in self.finetune_modules
            finetune_adapters = self.finetune_modules[finetune_context]

        for layer, adapter in zip_longest(self.layers, finetune_adapters):
            input_data = layer(input_data, attention_mask=valid_mask, finetune_modules=adapter)

        embeddings = self.norm_layer(input_data)

        if return_embeddings_only:
            return embeddings

        output_logits = self.to_output_logits(embeddings)

        output_result = (output_logits, embeddings) if return_logits_and_embeddings else output_logits

        if not compute_loss:
            return output_result

        output_logits = rearrange(output_logits, 'b n c -> b c n')
        return nn.functional.cross_entropy(output_logits, target_labels, ignore_index=self.loss_ignore_index)

Nyx = NYX(
    model_dim=128,                     
    vocab_size=1000,              
    layer_depth=2,                      
    use_causal_mask=True,                  
    head_dimension=32,                  
    num_attention_heads=4,                      
    feedforward_scale=2,                    
    attention_dropout_rate=0.1,             
    feedforward_dropout_rate=0.1,               
    use_qk_rmsnorm=False,             
    rotary_pos_emb_scale=128,   
)

batch_size = 2
seq_len = 10  
dummy_input = torch.randint(0, Nyx.vocab_size, (batch_size, seq_len))

with torch.no_grad():
    output = Nyx(dummy_input)

print(f"Output shape: {output.shape}")

seq_len = 15  
prompt = torch.randint(0, Nyx.vocab_size, (1, 5))  

with torch.no_grad():
    generated_output = Nyx.generate_sequence(
        seq_len,
        prompt,
        1.0,
        None,
        0.9,
        None,
        True,
        False
    )

