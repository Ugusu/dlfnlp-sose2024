import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_bert import BertPreTrainedModel
from utils import get_extended_attention_mask


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf.
        # Note again: in the attention_mask non-padding tokens are marked with 0 and
        # adding tokens with a large negative number.

        # attention scores are calculated by multiplying queries and keys
        scores = torch.matmul(query, key.transpose(-2, -1))

        # get back a score matrix S of shape [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized) attention score between the j-th
        # and k-th token, given by i-th attention head before normalizing the scores
        scale = torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        scores = scores / scale

        # use the attention mask to mask out the padding token scores.
        broadcasted_mask = attention_mask.expand(scores.size(0), scores.size(1), scores.size(2), scores.size(3))
        scores = scores + broadcasted_mask
        
        # Normalize the scores.
        attention_probs = F.softmax(scores, dim=-1) 
        attention_probs = self.dropout(attention_probs)
        
        # Multiply the attention scores to the value and get back V' with shape 
        #[bs, num_attention_heads, seq_len, attention_head_size]    
        attention_output = torch.matmul(attention_probs, value) 
        
        # Next, we need to concat multi-heads and recover the original shape
        # [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].
        attention_output = attention_output.transpose(1, 2) 
        attention_output = attention_output.contiguous().view(attention_output.size(0), attention_output.size(1), self.all_head_size) 

        return attention_output

    def forward(self, hidden_states, attention_mask):
       
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input: torch.Tensor, output: torch.Tensor, dense_layer: nn.Linear, dropout: nn.Dropout, ln_layer: nn.LayerNorm) -> torch.Tensor:
        

        dense_layer_output = dense_layer(output)
        post_dropout_output = dropout(dense_layer_output)
        residual_add_output = post_dropout_output + input
        normalized_output = ln_layer(residual_add_output)

        return normalized_output

        # Hint: Remember that BERT applies dropout to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized.

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
      

        attention_result = self.self_attention(hidden_states, attention_mask)
        add_norm_attention_result = self.add_norm(hidden_states, attention_result, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
        feed_forward_result = self.interm_dense(add_norm_attention_result)
        feed_forward_result = self.interm_af(feed_forward_result)
        add_norm_feed_forward_result = self.add_norm(add_norm_attention_result, feed_forward_result, self.out_dense, self.out_dropout, self.out_layer_norm)

        return add_norm_feed_forward_result
        

class BertModel(BertPreTrainedModel):
    

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = self.pos_embedding(pos_ids)

        # Get token type ids, since we are not considering token type,
        # this is just a placeholder.
        tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and
        # return the hidden states.

        hidden_state_embeds = inputs_embeds + pos_embeds + tk_type_embeds
        hidden_state_embeds = self.embed_layer_norm(hidden_state_embeds)
        hidden_state_embeds = self.embed_dropout(hidden_state_embeds)

        return hidden_state_embeds

    def encode(self, hidden_states, attention_mask):
        
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype
        )

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def mean_pooling(self, token_embeddings, attention_mask):
    
        assert attention_mask.dim() == 2, f"Expected attention_mask of shape [batch_size, seq_len], but got {attention_mask.shape}"
        
        # Expand the attention mask to match the token embeddings' shape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        
        # Sum of the embeddings across the tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Count the number of non-padded tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Mean pooling: average the sum of the embeddings
        return sum_embeddings / sum_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        
        # Get the embeddings for each token
        embedding_output = self.embed(input_ids=input_ids)

        # Encode the embeddings through BERT layers
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

        # Apply mean pooling to generate sentence embeddings
        sentence_embeddings = self.mean_pooling(sequence_output, attention_mask)

        return {"last_hidden_state": sequence_output, "pooler_output": sentence_embeddings}
