"""Run policies on every map of a suite.

Each map is its own SocialJax environment instance: the map's spawn and wall
positions become compile-time constants, so every (map, teacher condition)
compiles once. Within a map, environments are vectorised over seeds.

    suite = MapSuite("maps/suite_v1.json")
    for entry in suite.entries:
        for present in (True, False):
            env = suite.env(entry.name, present)
            traj = rollout(env, random_policy, key, num_envs=64, num_steps=1000)
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from virtue_rl.envs.maps import MapEntry, load_suite
from virtue_rl.envs.roles import RoleConfig
from virtue_rl.envs.wrapper import EthicsHarvest

# policy(key, obs, state, params) -> actions. obs and state are batched over
# envs, actions has shape (num_envs, num_agents). `params` is any pytree of
# arrays (or None); it is traced, so changing it does not recompile.
Policy = Callable[[jax.Array, jax.Array, object], jax.Array]


class MapSuite:
    def __init__(self, path: str, roles: RoleConfig = RoleConfig(), **env_kwargs):
        self.path = path
        self.roles = roles
        self.env_kwargs = env_kwargs
        self.entries: list[MapEntry] = load_suite(path)
        self._by_name = {e.name: e for e in self.entries}
        self._envs: dict[tuple[str, bool], EthicsHarvest] = {}

    def entry(self, name: str) -> MapEntry:
        return self._by_name[name]

    def split(self, split: str) -> list[MapEntry]:
        return [e for e in self.entries if e.split == split]

    def env(self, name: str, teacher_present: bool) -> EthicsHarvest:
        key = (name, teacher_present)
        if key not in self._envs:
            e = self._by_name[name]
            self._envs[key] = EthicsHarvest(self.roles, map_ASCII=e.ascii(teacher_present),
                                            grid_size=e.grid_size, **self.env_kwargs)
        return self._envs[key]


def random_policy(key, obs, state, params=None):
    num_envs, num_agents = obs.shape[0], obs.shape[1]
    return jax.random.randint(key, (num_envs, num_agents), 0, 8)


@partial(jax.jit, static_argnames=("env", "policy", "num_envs", "num_steps", "keep_states"))
def rollout(env: EthicsHarvest, policy: Policy, key: jax.Array, num_envs: int,
            num_steps: int, keep_states: bool = False, policy_params=None) -> dict:
    """Roll `policy` for `num_steps` in `num_envs` parallel environments.

    Returns a dict of arrays with leading axes (num_steps, num_envs):
    "ethics" (EthicsInfo), "rewards", "actions", and "states" if requested.
    """
    key, k = jax.random.split(key)
    obs, state = jax.vmap(env.reset)(jax.random.split(k, num_envs))

    def body(carry, _):
        obs, state, key = carry
        key, ka, ks = jax.random.split(key, 3)
        actions = policy(ka, obs, state, policy_params)
        nobs, nstate, rew, done, info, ethics = jax.vmap(env.step)(
            jax.random.split(ks, num_envs), state, actions)
        out = {"ethics": ethics, "rewards": rew, "actions": actions}
        if keep_states:
            out["states"] = state
        return (nobs, nstate, key), out

    _, traj = jax.lax.scan(body, (obs, state, key), None, length=num_steps)
    return traj
