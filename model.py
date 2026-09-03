from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ensembles import CellEnsemble


@dataclass
class ModelOutput:
    logits: list[torch.Tensor]
    bottleneck: torch.Tensor
    lstm_output: torch.Tensor


def _trunc_normal_init(weight: torch.Tensor) -> None:
    fan_in = weight.shape[1]
    std = 1.0 / (fan_in ** 0.5)
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


class GridCellsRNN(nn.Module):
    def __init__(
        self,
        target_ensembles: list[CellEnsemble],
        nh_lstm: int = 128,
        nh_bottleneck: int = 512,
        dropout_rates: tuple = (0.5,),
        bottleneck_has_bias: bool = False,
        ego_vel_dim: int = 3,
        vision_dim: int = 0,
    ):
        super().__init__()
        assert nh_bottleneck % len(dropout_rates) == 0, (
            "nh_bottleneck must split evenly across dropout_rates groups")
        self.target_ensembles = nn.ModuleList(target_ensembles)
        self.dropout_rates = dropout_rates
        self.nh_lstm = nh_lstm
        self.nh_bottleneck = nh_bottleneck
        self.vision_dim = vision_dim

        n_init_in = sum(ens.n_cells for ens in target_ensembles)
        self.lstm = nn.LSTM(ego_vel_dim, nh_lstm, batch_first=True)
        self.rl_lstm = (nn.LSTM(ego_vel_dim + vision_dim, nh_lstm, batch_first=True)
                         if vision_dim > 0 else None)
        self.state_init = nn.Linear(n_init_in, nh_lstm)
        self.cell_init = nn.Linear(n_init_in, nh_lstm)
        self.bottleneck = nn.Linear(nh_lstm, nh_bottleneck, bias=bottleneck_has_bias)
        self.output_heads = nn.ModuleList(
            [nn.Linear(nh_bottleneck, ens.n_cells) for ens in target_ensembles])

        self._init_weights()

    def _init_weights(self) -> None:
        for head in self.output_heads:
            _trunc_normal_init(head.weight)
            nn.init.zeros_(head.bias)
        for layer in (self.bottleneck, self.state_init, self.cell_init):
            _trunc_normal_init(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        with torch.no_grad():
            h = self.nh_lstm
            self.lstm.bias_hh_l0[h:2 * h] += 1.0
            if self.rl_lstm is not None:
                self.rl_lstm.bias_hh_l0[h:2 * h] += 1.0

    def init_hidden(self, init_conds: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        concat_init = torch.cat(init_conds, dim=1)
        h0 = self.state_init(concat_init).unsqueeze(0)
        c0 = self.cell_init(concat_init).unsqueeze(0)
        return h0, c0

    def init_hidden_from_predictions(
        self, predictions: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.init_hidden(predictions)

    def step(
        self, vel_t: torch.Tensor, vision_t: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        assert self.rl_lstm is not None, "step() needs a model built with vision_dim > 0"
        lstm_in = torch.cat([vel_t, vision_t], dim=-1).unsqueeze(1)
        lstm_out, hidden = self.rl_lstm(lstm_in, hidden)
        bottleneck = self.bottleneck(lstm_out.squeeze(1))
        return bottleneck, hidden

    def forward(
        self, init_conds: list[torch.Tensor], vels: torch.Tensor,
        vision_seq: torch.Tensor | None = None,
    ) -> ModelOutput:
        h0, c0 = self.init_hidden(init_conds)

        if vision_seq is None:
            lstm_out, _ = self.lstm(vels, (h0, c0))
        else:
            assert self.rl_lstm is not None, "forward(vision_seq=...) needs vision_dim > 0"
            lstm_in = torch.cat([vels, vision_seq], dim=-1)
            lstm_out, _ = self.rl_lstm(lstm_in, (h0, c0))
        bottleneck = self.bottleneck(lstm_out)

        if self.training:
            chunks = torch.chunk(bottleneck, len(self.dropout_rates), dim=-1)
            bottleneck = torch.cat(
                [F.dropout(c, p=r, training=True) for c, r in zip(chunks, self.dropout_rates)],
                dim=-1)

        return ModelOutput(logits=[head(bottleneck) for head in self.output_heads],
                           bottleneck=bottleneck, lstm_output=lstm_out)

    def decay_parameters(self, scope: str = "output_heads") -> list[torch.nn.Parameter]:
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
        if scope == "all":
            return list(self.parameters())
        if scope not in ("output_heads", "bottleneck_and_heads"):
            raise ValueError(f"unknown grad clip scope {scope!r}")
        params = [p for head in self.output_heads for p in head.parameters()]
        if scope == "bottleneck_and_heads":
            params += list(self.bottleneck.parameters())
        return params
