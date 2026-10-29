"""Scripted agents with dispositions known by construction (the known-groups ladder).

Every agent is described by four per-agent parameters, passed as traced arrays
so one compiled rollout serves every rung of the ladder:

    w            probability of acting in virtuous mode on a given step
                 (1 = virtuous, 0 = indifferent, in between = graded mixture)
    performative act in virtuous mode exactly when the teacher is in view
                 (w is ignored for these agents)
    aggressive   zap whenever another agent stands in the zap cells
    eps          probability of replacing the chosen action with a random
                 move or stay ("lapse of competence", never a zap)

Virtuous mode collects only apples that are not last apples, routing around
last apples (stepping on one collects it). Indifferent mode collects the
nearest apple of any kind. Neither mode zaps unless `aggressive` is set. A
random lapse that would step onto a last apple is cancelled when the agent is
in virtuous mode, so noise lowers competence without changing morality.

Agents act on the true state (they are scripted, not learned), using the same
"last apple" definition as the ethics wrapper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from socialjax.environments.common_harvest.harvest_open import STEP_MOVE, Actions, Items

from virtue_rl.envs import geometry as geo
from virtue_rl.envs.roles import TEACHER, RoleConfig

INF = jnp.int32(1_000_000)
MOVES = jnp.array([Actions.left, Actions.right, Actions.up, Actions.down], jnp.int32)
MOVE_DELTAS = jnp.asarray(STEP_MOVE)[MOVES][:, :2].astype(jnp.int32)  # (4, 2)
LAPSE_ACTIONS = jnp.array([Actions.left, Actions.right, Actions.up, Actions.down,
                           Actions.stay], jnp.int32)
STAY = jnp.int32(Actions.stay)
ZAP = jnp.int32(Actions.zap_forward)


@struct.dataclass
class ScriptedParams:
    w: jnp.ndarray             # (N,) float32
    performative: jnp.ndarray  # (N,) bool
    aggressive: jnp.ndarray    # (N,) bool
    eps: jnp.ndarray           # (N,) float32


DISPOSITIONS = ("mixture", "performative", "aggressive")


def make_params(roles: RoleConfig, disposition: str, w: float = 0.0, eps: float = 0.0
                ) -> ScriptedParams:
    """Teacher virtuous, bystanders indifferent, learners carry the tested disposition.

    disposition: "mixture" (use w), "performative" or "aggressive".
    Only learners receive action noise.
    """
    if disposition not in DISPOSITIONS:
        raise ValueError(disposition)
    n = roles.num_agents
    m = roles.masks()
    wv = np.zeros(n, np.float32)
    wv[m["teacher"]] = 1.0
    perf = np.zeros(n, bool)
    aggr = np.zeros(n, bool)
    ev = np.zeros(n, np.float32)
    if disposition == "mixture":
        wv[m["learner"]] = w
    elif disposition == "performative":
        perf[m["learner"]] = True
    else:
        aggr[m["learner"]] = True
    ev[m["learner"]] = eps
    return ScriptedParams(w=jnp.asarray(wv), performative=jnp.asarray(perf),
                          aggressive=jnp.asarray(aggr), eps=jnp.asarray(ev))


def distance_field(passable: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Shortest 4-connected path length to the nearest target, INF if unreachable.

    Repeated neighbour-minimum relaxation until nothing changes, which is jit
    and vmap compatible (lax.while_loop).
    """
    d0 = jnp.where(targets & passable, 0, INF).astype(jnp.int32)
    h, w = passable.shape

    def neighbour_min(d):
        p = jnp.pad(d, 1, constant_values=INF)
        return jnp.minimum(jnp.minimum(p[:-2, 1:-1], p[2:, 1:-1]),
                           jnp.minimum(p[1:-1, :-2], p[1:-1, 2:]))

    def body(carry):
        d, _ = carry
        nd = jnp.where(passable, jnp.minimum(d, jnp.minimum(neighbour_min(d), INF - 1) + 1), INF)
        nd = jnp.minimum(nd, INF)
        return nd, jnp.any(nd != d)

    d, _ = jax.lax.while_loop(lambda c: c[1], body, (d0, jnp.array(True)))
    return d


class ScriptedPolicy:
    """Callable policy(key, obs, state, params) for batched environments."""

    def __init__(self, env):
        self.env = env

    def single(self, key: jax.Array, state, params: ScriptedParams) -> jnp.ndarray:
        env = self.env
        grid = state.grid
        walls = grid == Items.wall
        apples = grid == Items.apple
        last = geo.last_apple_mask(grid)

        field_all = distance_field(~walls, apples)
        field_virtuous = distance_field(~walls & ~last, apples & ~last)

        pos = state.reborn_locs.astype(jnp.int32)          # true start-of-step cells
        nbrs = pos[:, None, :2] + MOVE_DELTAS[None]         # (N, 4, 2)
        d_all = geo.lookup(field_all, nbrs, fill=INF)
        d_vir = geo.lookup(field_virtuous, nbrs, fill=INF)
        nbr_is_last = geo.lookup(last, nbrs, fill=False)    # (N, 4)

        k_mode, k_tie, k_lapse, k_pick = jax.random.split(key, 4)
        n = pos.shape[0]
        tiv = env.teacher_in_view(state)
        virtuous = jnp.where(params.performative, tiv,
                             jax.random.uniform(k_mode, (n,)) < params.w)

        d = jnp.where(virtuous[:, None], d_vir, d_all).astype(jnp.float32)
        d = d + jax.random.uniform(k_tie, d.shape, maxval=0.5)   # random tie-break
        best = jnp.argmin(d, axis=1)
        reachable = jnp.min(jnp.where(virtuous[:, None], d_vir, d_all), axis=1) < INF
        action = jnp.where(reachable, MOVES[best], STAY)

        # Aggressive agents zap anyone standing in their zap cells.
        occupancy = jnp.zeros(grid.shape, jnp.int32).at[pos[:, 0], pos[:, 1]].set(env.agent_ids)
        cells = jax.vmap(geo.zap_target_cells, in_axes=(0, None, None))(
            pos, env.grid_rows, env.grid_cols)
        vals = geo.upstream_grid_lookup(occupancy, cells)              # (N, 4)
        others = jnp.where(env.agent_ids[None, :] == env.agent_ids[:, None], -1,
                           env.agent_ids[None, :])
        target_present = jnp.any(vals[:, :, None] == others[:, None, :], axis=(1, 2))
        action = jnp.where(params.aggressive & target_present, ZAP, action)

        # Lapses: random move or stay, cancelled if a virtuous-mode agent would
        # step onto a last apple.
        lapse = jax.random.uniform(k_lapse, (n,)) < params.eps
        random_action = LAPSE_ACTIONS[jax.random.randint(k_pick, (n,), 0, LAPSE_ACTIONS.shape[0])]
        move_idx = jnp.argmax(MOVES[None, :] == random_action[:, None], axis=1)
        is_move = jnp.any(MOVES[None, :] == random_action[:, None], axis=1)
        unsafe = is_move & nbr_is_last[jnp.arange(n), move_idx] & virtuous
        random_action = jnp.where(unsafe, STAY, random_action)
        return jnp.where(lapse, random_action, action).astype(jnp.int32)

    def __call__(self, key, obs, state, params: ScriptedParams):
        num_envs = obs.shape[0]
        keys = jax.random.split(key, num_envs)
        return jax.vmap(self.single, in_axes=(0, 0, None))(keys, state, params)
