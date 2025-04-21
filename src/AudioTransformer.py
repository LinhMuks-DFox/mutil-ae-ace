from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

EmbeddingLayerType = torch.nn.Module


class AttentionBlock(nn.Module):
    @staticmethod
    def make_ffn(hidden_dim: int, expand_factor: int = 1, ffn_dropout_ratio: float = 0.1) -> torch.nn.Module:
        return nn.Sequential(
            OrderedDict([
                ("ffn_linear1", nn.Linear(in_features=hidden_dim, out_features=hidden_dim * expand_factor)),
                ("ffn_relu", nn.ReLU()),
                ("DropOut,", nn.Dropout(ffn_dropout_ratio)),
                ("ffn_linear2", nn.Linear(in_features=hidden_dim * expand_factor, out_features=hidden_dim))
            ])
        )

    def __init__(self, embed_dim, n_head, hidden_expand_factor: int = 1, ffn_dropout_ratio: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.hidden_expand_factor = hidden_expand_factor
        self.ffn_dropout_ratio = ffn_dropout_ratio
        self.feed_forward = self.make_ffn(embed_dim, self.hidden_expand_factor, self.ffn_dropout_ratio)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer("positional_embedding", self.sinusoids(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_embedding[:, :x.shape[-1]].to(x.device)
        x = (x + pe).to(x.dtype)
        return x

    @staticmethod
    def sinusoids(length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class AudioTransformer(nn.Module):
    def __init__(self, embedded_size: int,
                 n_classes: int, n_block: int, n_head: int, hidden_expand_factor: int = 1,
                 ffn_dropout_ratio: float = 0.1):
        super().__init__()
        self.hidden_expand_factor = hidden_expand_factor
        self.ffn_dropout_ratio = ffn_dropout_ratio
        self.attention_block = nn.Sequential(
            *[AttentionBlock(embedded_size, n_head, self.hidden_expand_factor, self.ffn_dropout_ratio)
              for _ in range(n_block)]
        )
        self.n_classes = n_classes
        self.class_mlp = nn.Linear(in_features=embedded_size, out_features=self.n_classes)
        self.embedded_size = embedded_size

        self.positional_encoder = PositionalEncoding(embedded_size)

    @staticmethod
    def concat_cls_token(x: torch.Tensor):
        # [batch_size, token_size, n_token] -> [batch_size, token_size, n_token + 1]
        batch_size, token_size, n_token = x.shape
        cls_token = torch.ones((batch_size, token_size, 1), device=x.device)
        x = torch.cat([cls_token, x], dim=2)
        return x.contiguous()

    @staticmethod
    def get_cls_token(x: torch.Tensor):
        # Extract the class token (first token) from each sequence
        return x[:, :, 0]

    @staticmethod
    def expand(x: torch.Tensor):
        # [batch_size, n_mic, n_channel, num_tokens] -> [batch_size, n_mic * n_channel, num_tokens]
        batch_size, n_mic, n_channel, num_tokens = x.shape
        return x.view(batch_size, n_mic * n_channel, num_tokens)

    def forward(self, x: torch.Tensor):
        """
        x shall be a tensor of [n_mic, n_channel, time_length] or [batch_size, n_mic, n_channel, time_length]
            * if x not in shape of [batch_size, n_mic, n_channel, time_length], then x = unsqueeze(0)
            * time_length is num_tokens.
                * if sample rate is 16k, in 10s, there shall be 160k tokens.
                * if a input tensor in shape of [10, 5, 4, 150], means 10 batch, a multi-mic-array has 5 mic, each mic
                    produce 4 channel audio, each audio in length of 150
                * Apply time direction multi-head attention, [10, 5, 4, 150] -> [10, 5 * 4, 150] -> [10, 20, 150], means,
                    10 batch, 20 is the token length(verticle), 150 tokens

        Process:
        1. x:=expand(x)
            [batch_size, n_mic, n_channel, num_tokens] -> [batch_size, n_mic * n_channel, num_tokens]
        2. x:=concat_cls_token(x)
            [batch_size, n_mic * n_channel, num_tokens] -> [batch_size, n_mic * n_channel, num_tokens + 1]
        3. x:=position_encoding(x)
            [batch_size, n_mic * n_channel, num_tokens] -> [batch_size, n_mic * n_channel, num_tokens]
        4. x:=Multi-headAttention(x)
            [batch_size, n_mic * n_channel, num_tokens + 1] -> [batch_size, n_mic * n_channel, num_tokens + 1]
            *Important*: Apply attention on *TIME* direction.
        5. x:=get_cls_token(x)
            [batch_size, n_mic * n_channel, num_tokens + 1] -> [batch_size, n_mic * n_channel, 1]
        6. y:=cls_mpl(x)
            [batch_size, n_mic * n_channel, 1] -> [1, n_classes], y is a probability vector
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.expand(x)
        x = self.concat_cls_token(x)
        # Permute for Multi-headAttention (batch_size, token_seq, token)
        x = self.positional_encoder(x)
        x = x.permute(0, 2, 1).contiguous()
        # print(x.shape)
        x = self.attention_block(x)
        x = x.permute(0, 2, 1).contiguous()  # Permute back (batch_size, n_mic * n_channel, num_tokens + 1)
        # print(x.shape)

        x = self.get_cls_token(x)
        y = self.class_mlp(x)
        return y


def make_transformer_with_embedding_layer(emb: EmbeddingLayerType,
                                          embedded_size: int,
                                          n_classes: int,
                                          n_block: int,
                                          n_head: int):
    return nn.Sequential(
        OrderedDict([("embedding_layer", emb), ("audio_transformer",
                                                AudioTransformer(embedded_size, n_classes, n_block, n_head))])
    )
