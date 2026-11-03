"""Trained networks in chosen slots, scripted agents in the others.

    policy = SlotPolicy(env, learned_slots=(1,))
    params = SlotParams(scripted=make_params(roles, "mixture", w=0.0),
                        networks=(learner_params,))
    traj = rollout(env, policy, key, num_envs, num_steps, policy_params=params)

Each learned slot has its own parameters (they may be the same object).
Actions are sampled from the policy, as during training. Scripted actions
are computed for every slot and then replaced in the learned slots, so the
scripted agents see exactly the random keys they would see without learned
agents.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from virtue_rl.agents.scripted import ScriptedParams, ScriptedPolicy
from virtue_rl.train.networks import ActorCritic


@struct.dataclass
class SlotParams:
    scripted: ScriptedParams
    networks: tuple[Any, ...]


class SlotPolicy:
    def __init__(self, env, learned_slots: tuple[int, ...], activation: str = "relu"):
        self.env = env
        self.learned_slots = tuple(int(s) for s in learned_slots)
        self.scripted = ScriptedPolicy(env)
        self.network = ActorCritic(env.action_space().n, activation=activation)

    def init_params(self, key: jax.Array):
        """Fresh network params, also the template for loading a checkpoint."""
        obs, _ = jax.eval_shape(self.env.reset, jax.random.PRNGKey(0))
        return self.network.init(key, jnp.zeros((1, *obs.shape[1:])))

    def __call__(self, key, obs, state, params: SlotParams):
        k_s, k_l = jax.random.split(key)
        actions = self.scripted(k_s, obs, state, params.scripted)
        for j, slot in enumerate(self.learned_slots):
            pi, _ = self.network.apply(params.networks[j], obs[:, slot].astype(jnp.float32))
            a = pi.sample(seed=jax.random.fold_in(k_l, j)).astype(actions.dtype)
            actions = actions.at[:, slot].set(a)
        return actions


def learned_slot_mask(n: int, learned_slots: tuple[int, ...]) -> np.ndarray:
    m = np.zeros(n, bool)
    m[list(learned_slots)] = True
    return m
