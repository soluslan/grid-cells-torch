"""Grid cell supervised-learning model.

Ported from google-deepmind/grid-cells' model.py (a custom snt.RNNCore
unrolled via tf.nn.dynamic_rnn) to a plain nn.Module using nn.LSTM. Because
nn.LSTM natively processes a whole [B,T,*] sequence and nn.Linear natively
broadcasts over leading dimensions, the custom per-step RNNCore machinery the
original needed (Sonnet v1's static-graph model required the per-step Linear
layers to live *inside* the unrolled cell) collapses into a much simpler
"LSTM over the whole sequence, then Linear layers applied to the whole
sequence at once" design. This is an intentional simplification, not a
functional difference.

Two known dead-code items from the original are NOT carried over:
  - `nh_embed`: stored but never used to build an embedding layer in the
    original _build(); dropped entirely here.
  - weight decay is *actually* wired into training here (see train.py's
    optimizer param groups) since the original registered Sonnet
    `regularizers` on the bottleneck/output-head weights but never summed
    them into the optimized loss anywhere in train.py -- i.e. weight decay
    was a no-op in the original as published.
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
    """Matches the original's displaced_linear_initializer: truncated normal
    with mean=displace*stddev, stddev=1/sqrt(fan_in), truncated at 2 stddev."""
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
        self.state_init = nn.Linear(n_init_in, nh_lstm)
        self.cell_init = nn.Linear(n_init_in, nh_lstm)
        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)
        self.output_heads = nn.ModuleList(
            [nn.Linear(nh_bottleneck, ens.n_cells) for ens in target_ensembles])

        self._init_weights(init_weight_disp)

    def _init_weights(self, init_weight_disp: float) -> None:
        # Output heads (pc_logits equivalent) use the original's displaced
        # truncated-normal initializer. bottleneck/state_init/cell_init had no
        # custom initializer in the original (Sonnet v1's own default, itself
        # a truncated-normal with std=1/sqrt(fan_in) -- i.e. the same family
        # at displace=0), so the same helper is applied uniformly here.
        for head in self.output_heads:
            _trunc_normal_init(head.weight, displace=init_weight_disp)
            nn.init.zeros_(head.bias)
        for layer in (self.bottleneck, self.state_init, self.cell_init):
            _trunc_normal_init(layer.weight, displace=0.0)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Original's snt.LSTM adds a constant forget_bias=1.0 to the forget
        # gate's pre-activation at every step ("to reduce the scale of
        # forgetting in the beginning of training" -- Sonnet's own docstring).
        # nn.LSTM has no such default: bias_ih_l0/bias_hh_l0 are laid out as
        # 4 equal blocks in gate order [input, forget, cell, output], so add
        # the missing +1.0 to the forget block of one of the two bias vectors
        # (their sum is what feeds the forget gate, matching Sonnet adding
        # forget_bias once on top of its single combined bias).
        with torch.no_grad():
            h = self.nh_lstm
            self.lstm.bias_hh_l0[h:2 * h] += 1.0

    def forward(self, init_conds: list[torch.Tensor], vels: torch.Tensor) -> ModelOutput:
        """
        Args:
            init_conds: one [B, ens.n_cells] tensor per target ensemble (t=0
                encoding, from encode_initial_conditions()).
            vels: [B, T, ego_vel_dim] egocentric velocity input.
        Returns:
            ModelOutput with per-ensemble logits, bottleneck, and lstm output,
            all with a leading [B,T,...] shape.
        """
        concat_init = torch.cat(init_conds, dim=1)  # [B, n_init_in]
        h0 = self.state_init(concat_init).unsqueeze(0)  # [1,B,nh_lstm]
        c0 = self.cell_init(concat_init).unsqueeze(0)  # [1,B,nh_lstm]

        lstm_out, _ = self.lstm(vels, (h0, c0))  # [B,T,nh_lstm]
        bottleneck = self.bottleneck(lstm_out)  # [B,T,nh_bottleneck]

        if self.training:
            chunks = torch.chunk(bottleneck, len(self.dropout_rates), dim=-1)
            bottleneck = torch.cat(
                [F.dropout(c, p=r, training=True) for c, r in zip(chunks, self.dropout_rates)],
                dim=-1)

        logits = [head(bottleneck) for head in self.output_heads]
        return ModelOutput(logits=logits, bottleneck=bottleneck, lstm_output=lstm_out)

    def decay_parameters(self) -> list[torch.nn.Parameter]:
        """Weights that should receive L2 weight decay: bottleneck + output
        head *weights* only (never biases) -- matches the original's
        Sonnet `regularizers={"w": ...}`, which only ever targeted "w"."""
        params = [self.bottleneck.weight]
        params += [head.weight for head in self.output_heads]
        return params

    def no_decay_parameters(self) -> list[torch.nn.Parameter]:
        decay_ids = {id(p) for p in self.decay_parameters()}
        return [p for p in self.parameters() if id(p) not in decay_ids]
