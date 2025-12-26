import os
import sys

import torch
import torch.nn as nn

from typing import Optional

class LLM_Config:
    def __init__(self,
                 embed_dim,
                 hidden_size,
                 num_heads,
                 ffn_dim,
                 vocab_size,
                 word_embed_proj_dim,
                 pad_token_id,
                 max_position_embeddings,
                 enable_bias,
                 layer_norm_elementwise_affine,
                 do_layer_norm_before):
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size
        self.word_embed_proj_dim = word_embed_proj_dim
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine
        self.do_layer_norm_before = do_layer_norm_before



class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self,
        attention_mask: torch.LongTensor,
        past_key_values_length: int = 0,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""

        if position_ids is None:
            position_ids = torch.cumsum(attention_mask, dim=1)
            position_ids = (position_ids * attention_mask - 1).long()
            # cut positions if `past_key_values_length` is > 0
            position_ids = position_ids[:, past_key_values_length:]

        return super().forward(position_ids + self.offset)

class my_opt_decoder(nn.Module):
    def __init__(self, config: LLM_Config, current_seq_len):
        super(my_opt_decoder, self).__init__()
        self.config = config

        self.head_dim = self.config.embed_dim // self.config.num_heads
        self.scaling = self.head_dim**-0.5

        # Embedding layers
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.word_embed_proj_dim, self.config.pad_token_id)
        self.embed_positions = OPTLearnedPositionalEmbedding(self.config.max_position_embeddings, config.hidden_size)
        self.project_in = nn.Linear(self.config.word_embed_proj_dim, self.config.hidden_size, bias=False)
        
        # KV Cache
        # self.past_k = torch.randn(bsz, num_heads, current_seq_len, self.head_dim)
        # self.past_v = torch.randn(bsz, num_heads, current_seq_len, self.head_dim)
        self.register_buffer(
            "past_k",
            torch.randn(bsz, num_heads, current_seq_len, self.head_dim)
        )
        self.register_buffer(
            "past_v",
            torch.randn(bsz, num_heads, current_seq_len, self.head_dim)
        )


        # QKV layers
        self.k_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=self.config.enable_bias)
        self.v_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=self.config.enable_bias)
        self.q_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=self.config.enable_bias)
        self.o_proj = nn.Linear(self.config.embed_dim, self.config.embed_dim, bias=self.config.enable_bias)

        self.self_attn_layer_norm = nn.LayerNorm(self.config.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        # FC layers
        self.activation_fn = nn.ReLU()
        self.fc1 = nn.Linear(self.config.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.config.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.config.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        # LM head
        self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        self.lm_head_linear = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

    def embed(self, input_ids):
        # input_ids: (bsz, seq_len)
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.project_in(inputs_embeds)
        bsz, seq_len, _ = inputs_embeds.size()
        attention_mask = (input_ids != self.config.pad_token_id).long()
        position_embeds = self.embed_positions(attention_mask=attention_mask)
        print(f"input shape: {input_ids.shape}, embed shape: {inputs_embeds.shape}, position shape: {position_embeds.shape}")
        hidden_states = inputs_embeds + position_embeds
        return hidden_states

    # qkv + rms
    def qkv(self, hidden_states):
        self.residual = hidden_states
        self.bsz, self.tgt_len, _ = hidden_states.size()

        if self.config.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = query_states.view(self.bsz, -1, self.config.num_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        key_states = key_states.view(self.bsz, -1, self.config.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(self.bsz, -1, self.config.num_heads, self.head_dim).transpose(1, 2)

        return query_states, key_states, value_states

    # QK^T + SV
    def attn(self, query, key, value, attention_mask, scaling, dropout, **kwargs):
        # KV cache update
        self.past_k = torch.cat([self.past_k, key], dim=2)
        self.past_v = torch.cat([self.past_v, value], dim=2)

        key = self.past_k
        value = self.past_v

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=False)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output


    # out-proj + rms
    def out_proj(self, attn_output):
        attn_output = attn_output.reshape(self.bsz, self.tgt_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        attn_output = nn.functional.dropout(attn_output, p=0.0, training=False)
        attn_output = self.residual + attn_output

        # 350m applies layer norm AFTER attention
        if not self.config.do_layer_norm_before:
            attn_output = self.self_attn_layer_norm(attn_output)

        return attn_output

    # MLP + rms
    def ffn(self, hidden_states):
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.config.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=0.0, training=False)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.config.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        return outputs

    def lm_head(self, outputs, logits_to_keep):
        hidden_states = outputs[0]
        hidden_states = self.project_out(hidden_states)
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        print(f"input hidden_states shape: {hidden_states.shape}")
        print(f"config.word_embed_proj_dim, config.vocab_size: {config.word_embed_proj_dim} {config.vocab_size}")
        logits = self.lm_head_linear(hidden_states[:, slice_indices, :]).contiguous()

        return logits

    def forward(self, x):
        # hidden = self.embed(x)
        # return hidden
        
        outputs = self.ffn(x)
        return outputs




if __name__ == "__main__":

    sys.path.append(os.environ.get('TORCHSIM_DIR', default='/root/workspace/PyTorchSim'))

    from Scheduler.scheduler import PyTorchSimRunner
    module = PyTorchSimRunner.setup_device()
    device = module.custom_device()

    embed_dim = 1024
    hidden_size = 1024
    num_heads = 16
    ffn_dim = 4096
    vocab_size = 50272
    word_embed_proj_dim = 512
    pad_token_id = 1
    max_position_embeddings = 2048
    enable_bias = True
    layer_norm_elementwise_affine = True
    do_layer_norm_before = False

    bsz = 16
    seq_len = 16

    config = LLM_Config(embed_dim = embed_dim,
                        hidden_size = hidden_size,
                        num_heads = num_heads,
                        ffn_dim = ffn_dim,
                        vocab_size = vocab_size,
                        word_embed_proj_dim = word_embed_proj_dim,
                        pad_token_id = pad_token_id,
                        max_position_embeddings = max_position_embeddings,
                        enable_bias = enable_bias,
                        layer_norm_elementwise_affine = layer_norm_elementwise_affine,
                        do_layer_norm_before = do_layer_norm_before)

    decoder = my_opt_decoder(config, seq_len)
    decoder.eval()
    decoder_device = decoder.to(device=device)
    opt_decoder = torch.compile(decoder_device, dynamic=False)

    # Embedding is not supported currently, just skip
    # input = torch.randint(0, vocab_size, (bsz, seq_len)).to(device)  # (bsz, seq_len)
    # hidden = opt_decoder.embed(input)


    hidden = torch.randn(
        bsz, 1, config.hidden_size,
        dtype=torch.float32
    )
    hidden_device = hidden.to(device)
    opt_decoder(hidden_device)