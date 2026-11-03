"""Actor-critic network, the same architecture as SocialJax's IPPO baseline.

Copied from third_party/SocialJax/algorithms/utils/networks.py (commit 18e2171,
Apache 2.0) so that the trainer does not depend on SocialJax's `algorithms`
package (hydra, wandb). Layer sizes and initialisations are unchanged:
Conv(32, 5x5) -> Conv(32, 3x3) -> Conv(32, 3x3) -> Dense(64), then separate
actor (Dense 64 -> logits) and critic (Dense 64 -> 1) heads.
"""

from __future__ import annotations

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        for k in (5, 3, 3):
            x = act(nn.Conv(32, (k, k), kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(x))
        x = x.reshape((x.shape[0], -1))
        return act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        emb = CNN(self.activation)(x)
        a = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        c = act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)
