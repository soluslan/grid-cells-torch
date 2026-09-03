from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import VisionCNN

N_ACTIONS = 6


def _dmlab_action(*entries: int) -> np.ndarray:
    return np.array(entries, dtype=np.intc)


DISCRETE_ACTIONS = np.stack([
    _dmlab_action(-20, 0, 0, 0, 0, 0, 0),
    _dmlab_action(20, 0, 0, 0, 0, 0, 0),
    _dmlab_action(0, 0, 0, 1, 0, 0, 0),
    _dmlab_action(0, 0, 0, -1, 0, 0, 0),
    _dmlab_action(0, 0, -1, 0, 0, 0, 0),
    _dmlab_action(0, 0, 1, 0, 0, 0, 0),
])


@dataclass
class ActorCriticOutput:
    action_logits: torch.Tensor
    value: torch.Tensor
    hidden: tuple


class ActorCriticLSTM(nn.Module):
    def __init__(self, grid_code_dim: int, embed_dim: int = 256, lstm_units: int = 256):
        super().__init__()
        self.cnn = VisionCNN(embed_dim)
        input_dim = embed_dim + 1 + N_ACTIONS + 2 * grid_code_dim
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.actor = nn.Linear(lstm_units, N_ACTIONS)
        self.critic = nn.Linear(lstm_units, 1)

    def init_hidden(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return z, z.clone()

    def step(
        self,
        image: torch.Tensor,
        reward: torch.Tensor,
        prev_action: torch.Tensor,
        grid_code: torch.Tensor,
        goal_grid_code: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> ActorCriticOutput:
        e = self.cnn(image)
        prev_action_onehot = F.one_hot(prev_action, num_classes=N_ACTIONS).float()
        x = torch.cat([e, reward, prev_action_onehot, grid_code, goal_grid_code], dim=-1)
        lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)
        lstm_out = lstm_out.squeeze(1)
        return ActorCriticOutput(
            action_logits=self.actor(lstm_out), value=self.critic(lstm_out).squeeze(-1),
            hidden=hidden,
        )
