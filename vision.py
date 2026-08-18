"""Vision module for the RL agent (RL-agent roadmap plan, M2).

Ports Methods' "Agent architectures" vision-module spec (Supplementary Methods 3b): a 4-layer
CNN over 64x64 RGB producing a 256-d embedding, followed by a linear layer trained via the same
supervised cross-entropy loss as the grid network (model.GridCellsRNN) to predict place/
head-direction cell activity from that embedding.

Masking placement, a paper-ambiguity judgment call (documented per this project's existing
README convention rather than guessed at silently): Methods says "the output of the
convolutional network e_t was then passed through a masking layer which zeroed the input with
probability of 95%" -- read literally, the embedding e_t itself is what gets zeroed, *before*
the place/HD heads run on it, not the heads' output afterward. Implemented that way here: a
masked frame's embedding is a zero vector, so its place/HD prediction is whatever the heads'
own bias produces from zero input (not a hard-zeroed prediction vector). This only applies to
this module's embedding -- the actor-critic's own separate-weights CNN (actor_critic.py, M4)
is never masked; only the grid-network-facing path is.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ensembles import CellEnsemble, soft_cross_entropy


@dataclass
class VisionOutput:
    embedding: torch.Tensor      # [B, embed_dim] -- e_t, possibly zeroed by masking
    place_probs: torch.Tensor    # [B, n_pc] softmax place-cell prediction
    hd_probs: torch.Tensor       # [B, n_hdc] softmax head-direction-cell prediction
    masked: torch.Tensor         # [B] bool, which rows of this batch were masked


class VisionCNN(nn.Module):
    """4-conv encoder, Supplementary Methods 3b: 16/32/64/128 filters, 5x5, stride 2, pad 2,
    ReLU, then FC-embed_dim. Shared *architecture* with the actor-critic's own CNN
    (actor_critic.py, M4) -- never shared *weights*; Extended Data Fig. 5 draws them as two
    distinct boxes with independent parameters.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), nn.ReLU(),
        )
        # 64x64 input, stride 2 four times -> 4x4 spatial; 128 channels -> 128*4*4 flattened.
        self.fc = nn.Linear(128 * 4 * 4, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,64,64] in [-1,1] -> [B, embed_dim]."""
        x = self.conv(images).flatten(1)
        return F.relu(self.fc(x))


class VisionModule(nn.Module):
    """VisionCNN + linear place/HD prediction heads + the 95%-masking layer."""

    def __init__(self, place_ensemble: CellEnsemble, hd_ensemble: CellEnsemble,
                 embed_dim: int = 256, mask_prob: float = 0.95):
        super().__init__()
        self.cnn = VisionCNN(embed_dim)
        self.place_head = nn.Linear(embed_dim, place_ensemble.n_cells)
        self.hd_head = nn.Linear(embed_dim, hd_ensemble.n_cells)
        self.mask_prob = mask_prob

    def forward(self, images: torch.Tensor) -> VisionOutput:
        e = self.cnn(images)
        # Applies unconditionally (train and eval alike): this is a described property of the
        # agent's observation model -- the grid network only sees visual corrections 5% of the
        # time -- not a training-only regularizer to disable at eval/rollout time.
        masked = torch.rand(e.shape[0], device=e.device) < self.mask_prob
        e = e * (~masked).float().unsqueeze(-1)

        place_probs = F.softmax(self.place_head(e), dim=-1)
        hd_probs = F.softmax(self.hd_head(e), dim=-1)
        return VisionOutput(embedding=e, place_probs=place_probs, hd_probs=hd_probs, masked=masked)

    def loss(self, images: torch.Tensor, place_targets: torch.Tensor,
             hd_targets: torch.Tensor) -> torch.Tensor:
        """Supervised loss, same form as the grid network's L(y,z,c,h) but per-frame (no time
        dimension, since the vision-learner thread samples single frames from the replay
        buffer, not sequences) and computed from the UNMASKED embedding -- the loss trains the
        heads to predict correctly when they do get to see the frame; masking is something that
        happens to their output on the way to the grid network, not to their own training
        signal (an always-95%-zeroed loss would never train anything).
        """
        e = self.cnn(images)
        place_logits = self.place_head(e)
        hd_logits = self.hd_head(e)
        return (soft_cross_entropy(place_logits, place_targets)
                + soft_cross_entropy(hd_logits, hd_targets)).mean()
