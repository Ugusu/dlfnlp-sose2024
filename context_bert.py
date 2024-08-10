import torch
import torch.nn as nn
import math


class ContextBERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(ContextBERTSelfAttention, self).__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_layer = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.context_to_query = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.context_to_key = nn.Linear(self.attention_head_size, self.attention_head_size)

        self.lambda_query_context = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_query = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_key_context = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_key = nn.Linear(self.attention_head_size, 1, bias=False)

        self.sigmoid_activation = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, context_embeddings=None):
        """
        Args:
            hidden_states (torch.Tensor): Input tensor with shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): Mask tensor with shape [batch_size, 1, 1, seq_len]
            context_embeddings (torch.Tensor, optional): Context tensor with shape [batch_size, 1, hidden_size]

        Returns:
            torch.Tensor: Contextualized output tensor
        """

        # Linear transformation for query, key, and value
        mixed_query_layer = self.query_layer(hidden_states)
        mixed_key_layer = self.key_layer(hidden_states)
        mixed_value_layer = self.value_layer(hidden_states)

        # Transpose for scores
        mixed_query_layer = self.transpose_for_scores(mixed_query_layer)
        mixed_key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Integrate global context embeddings into attention calculation if provided
        if context_embeddings is not None:
            transposed_context = self.transpose_for_scores(context_embeddings)

            # Contextualizing query
            context_query = self.context_to_query(transposed_context)
            lambda_query_context = self.lambda_query_context(context_query)
            lambda_query_query = self.lambda_query(mixed_query_layer)
            lambda_query = lambda_query_context + lambda_query_query
            lambda_query = self.sigmoid_activation(lambda_query)
            contextualized_query_layer = (1 - lambda_query) * mixed_query_layer + lambda_query * context_query

            # Contextualizing key
            context_key = self.context_to_key(transposed_context)
            lambda_key_context = self.lambda_key_context(context_key)
            lambda_key_key = self.lambda_key(mixed_key_layer)
            lambda_key = lambda_key_context + lambda_key_key
            lambda_key = self.sigmoid_activation(lambda_key)
            contextualized_key_layer = (1 - lambda_key) * mixed_key_layer + lambda_key * context_key
        else:
            raise Exception('No context representation provided')

        # Calculate attention scores
        attention_scores = torch.matmul(contextualized_query_layer, contextualized_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # Contextualized output
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

