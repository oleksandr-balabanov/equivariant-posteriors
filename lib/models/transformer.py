import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec
import math


@dataclass(frozen=True)
class TransformerConfig:
    embed_d: int
    mlp_dim: int
    n_seq: int
    batch_size: int
    num_layers: int
    num_heads: int

    def serialize_human(self):
        return self.__dict__


class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig, data_spec: DataSpec):
        super().__init__()
        embed_d = config.embed_d
        self.embed = torch.nn.Linear(data_spec.input_shape[-1], embed_d, bias=True)
        lnorm = torch.nn.LayerNorm(embed_d, eps=1e-5)

        self.pos_embed = PositionalEncoding(config.embed_d, dropout=0.0)
        self.layer = torch.nn.TransformerDecoderLayer(
            d_model=embed_d,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerDecoder(
            self.layer, num_layers=config.num_layers, norm=lnorm
        )

        self.debed = torch.nn.Linear(embed_d, data_spec.output_shape[-1])
        self.register_buffer(
            "mem",
            torch.randn(
                (config.batch_size, config.n_seq, embed_d), requires_grad=False
            ),
        )

    def forward(self, batch):
        x = batch["input"]
        embed = self.embed(x) * math.sqrt(32)
        embed = self.pos_embed(embed)
        tout = self.transformer(embed, self.mem[: embed.shape[0]])
        # return torch.softmax(self.debed(tout), dim=-1)[:, 0, :]
        output = self.debed(tout[:, 0, :])
        return dict(logits=output, predictions=self.output_to_value(output))

    def output_to_value(self, output):
        return torch.softmax(output, dim=-1)

    # def forward_full(self, x):
    #     return self.output_to_value(self.forward(x))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
