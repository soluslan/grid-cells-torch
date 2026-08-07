"""Grid cell supervised-learning model.

Ports google-deepmind/grid-cells' model.py (a custom snt.RNNCore unrolled by
tf.nn.dynamic_rnn) to a plain nn.Module. Because nn.LSTM consumes a whole
[B,T,*] sequence and nn.Linear broadcasts over leading dimensions, the
original's per-step cell machinery collapses into "LSTM over the sequence,
then Linear layers over the sequence" -- a simplification, not a functional
difference. The original's unused `nh_embed` is dropped.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ensembles import CellEnsemble


@dataclass
class ModelOutput:
    logits: list[torch.Tensor]  # one [B,T,ens.n_cells] per target ensemble
    bottleneck: torch.Tensor  # [B,T,nh_bottleneck]
    lstm_output: torch.Tensor  # [B,T,nh_lstm]


def _trunc_normal_init(weight: torch.Tensor, displace: float = 0.0) -> None:
    """The original's displaced_linear_initializer: truncated normal, mean
    displace*stddev, stddev 1/sqrt(fan_in), truncated at 2 stddev."""
    fan_in = weight.shape[1]
    std = 1.0 / (fan_in ** 0.5)
    mean = displace * std
    nn.init.trunc_normal_(weight, mean=mean, std=std, a=mean - 2 * std, b=mean + 2 * std)


class GridCellsRNN(nn.Module):
    """LSTM that predicts place/head-direction cell activity from velocity."""

    def __init__(
        self,
        target_ensembles: list[CellEnsemble],
        nh_lstm: int = 128,
        nh_bottleneck: int = 512,
        dropout_rates: tuple = (0.5,),
        bottleneck_has_bias: bool = False,
        init_weight_disp: float = 0.0,
        ego_vel_dim: int = 3,
    ):
        super().__init__()
        assert nh_bottleneck % len(dropout_rates) == 0, (
            "nh_bottleneck must split evenly across dropout_rates groups")
        self.target_ensembles = nn.ModuleList(target_ensembles)
        self.dropout_rates = dropout_rates
        self.nh_lstm = nh_lstm
        self.nh_bottleneck = nh_bottleneck

        n_init_in = sum(ens.n_cells for ens in target_ensembles)
        self.lstm = nn.LSTM(ego_vel_dim, nh_lstm, batch_first=True)
        # state_init/cell_init have a learned bias, as in the original -- the
        # paper's Extended Data Fig. 1 equations show none. See README.
        self.state_init = nn.Linear(n_init_in, nh_lstm)
        self.cell_init = nn.Linear(n_init_in, nh_lstm)
        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)
        self.output_heads = nn.ModuleList(
            [nn.Linear(nh_bottleneck, ens.n_cells) for ens in target_ensembles])

        self._init_weights(init_weight_disp)

    def _init_weights(self, init_weight_disp: float) -> None:
        # Only the output heads had a custom initializer in the original; the
        # rest used Sonnet v1's default, the same family at displace=0.
        for head in self.output_heads:
            _trunc_normal_init(head.weight, displace=init_weight_disp)
            nn.init.zeros_(head.bias)
        for layer in (self.bottleneck, self.state_init, self.cell_init):
            _trunc_normal_init(layer.weight, displace=0.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # snt.LSTM adds a constant forget_bias=1.0 that nn.LSTM has no
        # equivalent for. Biases are laid out [input, forget, cell, output],
        # so add it to the forget block of one of the two bias vectors.
        with torch.no_grad():
            h = self.nh_lstm
            self.lstm.bias_hh_l0[h:2 * h] += 1.0

    def forward(self, init_conds: list[torch.Tensor], vels: torch.Tensor) -> ModelOutput:
        """init_conds: one [B, n_cells] per ensemble; vels: [B,T,ego_vel_dim]."""
        concat_init = torch.cat(init_conds, dim=1)
        h0 = self.state_init(concat_init).unsqueeze(0)  # [1,B,nh_lstm]
        c0 = self.cell_init(concat_init).unsqueeze(0)

        lstm_out, _ = self.lstm(vels, (h0, c0))  # [B,T,nh_lstm]
        bottleneck = self.bottleneck(lstm_out)  # [B,T,nh_bottleneck]

        if self.training:
            chunks = torch.chunk(bottleneck, len(self.dropout_rates), dim=-1)
            bottleneck = torch.cat(
                [F.dropout(c, p=r, training=True) for c, r in zip(chunks, self.dropout_rates)],
                dim=-1)

        return ModelOutput(logits=[head(bottleneck) for head in self.output_heads],
                           bottleneck=bottleneck, lstm_output=lstm_out)

    def decay_parameters(self, scope: str = "output_heads") -> list[torch.nn.Parameter]:
        """Weights receiving L2 decay -- always weights, never biases, per the
        paper ("the *weights* projecting from the dropout layer, g_t, to [...]
        y_t and z_t") and the original's `regularizers={"w": ...}`.

        "output_heads" is that sentence read literally. "bottleneck_and_heads"
        also decays the LSTM->g bottleneck, as the original registered.
        """
        if scope not in ("output_heads", "bottleneck_and_heads"):
            raise ValueError(f"unknown weight decay scope {scope!r}")
        params = [head.weight for head in self.output_heads]
        if scope == "bottleneck_and_heads":
            params = [self.bottleneck.weight] + params
        return params

    def no_decay_parameters(self, scope: str = "output_heads") -> list[torch.nn.Parameter]:
        decay_ids = {id(p) for p in self.decay_parameters(scope)}
        return [p for p in self.parameters() if id(p) not in decay_ids]

    def clip_parameters(self, scope: str = "output_heads") -> list[torch.nn.Parameter]:
        """Parameters that gradient clipping applies to.

        "output_heads" is the paper's Methods read literally ("parameters
        projecting from the dropout layer, g_t, to [...] y_t and z_t"), so the
        LSTM and bottleneck go unclipped. "bottleneck_and_heads" matches the
        original's unused `clip_bottleneck_gradient`; "all" matches its actual
        default, `clip_all_gradients`.
        """
        if scope == "all":
            return list(self.parameters())
        if scope not in ("output_heads", "bottleneck_and_heads"):
            raise ValueError(f"unknown grad clip scope {scope!r}")
        params = [p for head in self.output_heads for p in head.parameters()]
        if scope == "bottleneck_and_heads":
            params += list(self.bottleneck.parameters())
        return params
