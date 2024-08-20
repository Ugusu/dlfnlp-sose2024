import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ContextualAttention, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices for the self-attention mechanism
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.U_Q = nn.Linear(hidden_size, hidden_size)
        self.U_K = nn.Linear(hidden_size, hidden_size)

        # Gating parameters
        self.V_H = nn.Parameter(torch.randn(hidden_size, 1))
        self.V_C = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, H, C):
        # H: [batch_size, seq_len, hidden_size]
        # C: [batch_size, hidden_size]

        # Standard self-attention projections
        Q = self.W_Q(H)  # [batch_size, seq_len, hidden_size]
        K = self.W_K(H)  # [batch_size, seq_len, hidden_size]
        V = self.W_V(H)  # [batch_size, seq_len, hidden_size]

        # Gating mechanism
        lambda_Q = torch.sigmoid(
            Q.matmul(self.V_H) + C.matmul(self.U_Q.weight.t()).unsqueeze(1))  # [batch_size, seq_len, 1]
        lambda_K = torch.sigmoid(
            K.matmul(self.V_H) + C.matmul(self.U_K.weight.t()).unsqueeze(1))  # [batch_size, seq_len, 1]

        # Incorporate context into queries and keys
        Q_hat = (1 - lambda_Q) * Q + lambda_Q * C.unsqueeze(1).matmul(
            self.U_Q.weight.t())  # [batch_size, seq_len, hidden_size]
        K_hat = (1 - lambda_K) * K + lambda_K * C.unsqueeze(1).matmul(
            self.U_K.weight.t())  # [batch_size, seq_len, hidden_size]

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q_hat, K_hat.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32))  # [batch_size, seq_len, seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        O = torch.matmul(attention_probs, V)  # [batch_size, seq_len, hidden_size]

        return O, attention_scores


class GlobalContextLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalContextLayer, self).__init__()
        self.hidden_size = hidden_size

        # Instantiate the custom attention mechanism
        self.custom_attention = ContextualAttention(hidden_size)

    def forward(self, H):
        # H: [batch_size, seq_len, hidden_size]

        # Compute global context vector
        C = torch.mean(H, dim=1)  # [batch_size, hidden_size]

        # Apply custom contextual attention mechanism
        O, attention_scores = self.custom_attention(H, C)  # [batch_size, seq_len, hidden_size]

        return O, attention_scores


class ContextualAttentionRegularized(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(ContextualAttentionRegularized, self).__init__()
        self.hidden_size = hidden_size

        # Weight matrices for the self-attention mechanism
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.U_Q = nn.Linear(hidden_size, hidden_size)
        self.U_K = nn.Linear(hidden_size, hidden_size)

        # Gating parameters
        self.V_H = nn.Parameter(torch.randn(hidden_size, 1))
        self.V_C = nn.Parameter(torch.randn(hidden_size, 1))

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, H, C):
        # H: [batch_size, seq_len, hidden_size]
        # C: [batch_size, hidden_size]

        # Apply layer normalization to input
        H = self.layer_norm1(H)

        # Standard self-attention projections
        Q = self.W_Q(H)  # [batch_size, seq_len, hidden_size]
        K = self.W_K(H)  # [batch_size, seq_len, hidden_size]
        V = self.W_V(H)  # [batch_size, seq_len, hidden_size]

        # Gating mechanism
        lambda_Q = torch.sigmoid(
            Q.matmul(self.V_H) + C.matmul(self.U_Q.weight.t()).unsqueeze(1))  # [batch_size, seq_len, 1]
        lambda_K = torch.sigmoid(
            K.matmul(self.V_H) + C.matmul(self.U_K.weight.t()).unsqueeze(1))  # [batch_size, seq_len, 1]

        # Incorporate context into queries and keys
        Q_hat = (1 - lambda_Q) * Q + lambda_Q * C.unsqueeze(1).matmul(
            self.U_Q.weight.t())  # [batch_size, seq_len, hidden_size]
        K_hat = (1 - lambda_K) * K + lambda_K * C.unsqueeze(1).matmul(
            self.U_K.weight.t())  # [batch_size, seq_len, hidden_size]

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q_hat, K_hat.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=torch.float32))  # [batch_size, seq_len, seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, seq_len, seq_len]

        # Apply dropout to attention probabilities
        attention_probs = self.dropout(attention_probs)

        O = torch.matmul(attention_probs, V)  # [batch_size, seq_len, hidden_size]

        # Apply second layer normalization and residual connection
        O = self.layer_norm2(O + H)

        return O, attention_scores


class GlobalContextLayerRegularized(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(GlobalContextLayerRegularized, self).__init__()
        self.hidden_size = hidden_size

        # Instantiate the custom attention mechanism
        self.custom_attention = ContextualAttentionRegularized(hidden_size, dropout_rate)

        # Additional layer normalization and dropout for global context
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Feed-forward layer for refining global context
        self.ff_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, H):
        # H: [batch_size, seq_len, hidden_size]

        # Compute global context vector
        C = torch.mean(H, dim=1)  # [batch_size, hidden_size]

        # Refine global context through feed-forward layer
        C = self.ff_layer(C)

        # Apply dropout to refined global context
        C = self.dropout(C)

        # Apply custom contextual attention mechanism
        O, attention_scores = self.custom_attention(H, C)  # [batch_size, seq_len, hidden_size]

        # Apply final layer normalization and residual connection
        O = self.layer_norm(O + H)

        return O, attention_scores
