"""Actor-critic ("policy LSTM") for the RL agent (RL-agent roadmap plan, M4).

Extended Data Fig. 5: its own CNN (same architecture as vision.VisionCNN, never the same
weights -- two distinct boxes in the figure) produces e_t', concatenated with reward r_t,
previous action a_{t-1}, current grid code g_t (detached from the grid network -- stop-gradient
per Methods), and goal grid code g_* (= g_t observed the last time the goal was reached this
episode, zeros if not yet reached). Fed into an LSTM(256) with two output heads: actor
(Linear(6)+softmax over discrete actions) and critic (Linear(1), the value estimate).

Action discretization -- the paper doesn't specify the exact 6-way mapping ("the agent could
rotate in small increments, accelerate forwards, backwards, or sideways, or effect rotational
acceleration while moving" is a description, not a table), so this file makes and documents one
choice rather than guessing silently, following the DiscretizedRandomAgent pattern in
`lab/python/random_agent.py`: look-left, look-right, forward, backward, strafe-left,
strafe-right. FIRE/JUMP/CROUCH are dropped -- the paper's navigation tasks never mention combat
or jumping.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision import VisionCNN

N_ACTIONS = 6


def _dmlab_action(*entries: int) -> np.ndarray:
    return np.array(entries, dtype=np.intc)


# Index into this array == the actor head's action index. Each row is a native DMLab action
# vector: [LOOK_LR, LOOK_UD, STRAFE, MOVE, FIRE, JUMP, CROUCH]. See module docstring for the
# judgment call this encodes.
DISCRETE_ACTIONS = np.stack([
    _dmlab_action(-20, 0, 0, 0, 0, 0, 0),  # 0: look left
    _dmlab_action(20, 0, 0, 0, 0, 0, 0),   # 1: look right
    _dmlab_action(0, 0, 0, 1, 0, 0, 0),    # 2: forward
    _dmlab_action(0, 0, 0, -1, 0, 0, 0),   # 3: backward
    _dmlab_action(0, 0, -1, 0, 0, 0, 0),   # 4: strafe left
    _dmlab_action(0, 0, 1, 0, 0, 0, 0),    # 5: strafe right
])


@dataclass
class ActorCriticOutput:
    action_logits: torch.Tensor  # [B, N_ACTIONS]
    value: torch.Tensor          # [B]
    hidden: tuple  # (h, c), each [1,B,lstm_units] -- thread through the caller across steps


class ActorCriticLSTM(nn.Module):
    def __init__(self, grid_code_dim: int, embed_dim: int = 256, lstm_units: int = 256):
        super().__init__()
        self.cnn = VisionCNN(embed_dim)  # separate weights from vision.VisionModule's own CNN
        input_dim = embed_dim + 1 + N_ACTIONS + 2 * grid_code_dim
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.actor = nn.Linear(lstm_units, N_ACTIONS)
        self.critic = nn.Linear(lstm_units, 1)

    def init_hidden(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return z, z.clone()

    def step(
        self,
        image: torch.Tensor,        # [B,3,64,64] in [-1,1]
        reward: torch.Tensor,       # [B,1]
        prev_action: torch.Tensor,  # [B] long, previous discrete action index
        grid_code: torch.Tensor,    # [B, grid_code_dim] -- caller must .detach() before this
        goal_grid_code: torch.Tensor,  # [B, grid_code_dim] -- zeros if goal not yet reached
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> ActorCriticOutput:
        """One environment step. See module docstring for the input concatenation order."""
        e = self.cnn(image)
        prev_action_onehot = F.one_hot(prev_action, num_classes=N_ACTIONS).float()
        x = torch.cat([e, reward, prev_action_onehot, grid_code, goal_grid_code], dim=-1)
        lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)  # [B,1,lstm_units]
        lstm_out = lstm_out.squeeze(1)
        return ActorCriticOutput(
            action_logits=self.actor(lstm_out), value=self.critic(lstm_out).squeeze(-1),
            hidden=hidden,
        )
