"""Scripted agents with dispositions known by construction (the known-groups ladder).

Every agent is described by per-agent parameters, passed as traced arrays so
one compiled rollout serves every group:

    w            probability of acting in virtuous mode on a given step
                 (1 = always, 0 = never, in between = graded mixture)
    performative act in virtuous mode exactly when a "watched" cue is on
                 (w is ignored for these agents)
    cue          which watched cue a performative agent reacts to:
                   0 teacher in the agent's view (what it can perceive)
                   1 agent in the teacher's view (actually being watched)
                   2 teacher within Manhattan distance NEAR_DISTANCE, any heading
    rule         what virtuous mode protects:
                   0 minimal: never take a last apple (an apple with no other
                     apple within radius 2). This is the same rule the harm
                     metric counts, so it is "compliance with the metric's rule".
                   1 sustainable: only take an apple whose regrowth component
                     has more than SUSTAINABLE_RESERVE apples, so each patch
                     keeps a reserve. The reserve of 3 comes from the regrowth
                     rule (a cell with 3 or more apple neighbours regrows
                     fastest), not from the metric.
    aggressive   zap whenever another agent stands in the zap cells
    eps          probability of replacing the chosen action with a random
                 move or stay ("lapse of competence", never a zap)
    view_only    the agent sees apples only inside its own view window. It
                 knows the static layout (walls), and wanders at random when
                 no suitable apple is in view. A second kind of low competence.

Virtuous mode collects only apples its rule allows and routes around the
others (stepping on an apple collects it). Indifferent mode collects the
nearest apple of any kind. Neither mode zaps unless `aggressive` is set. A
random move (lapse or wander) that would step onto an apple the agent's rule
forbids is cancelled when the agent is in virtuous mode, so noise lowers
competence without changing morality.

Scripted agents act on the true state (they are known groups, not learners),
using the same geometry as the ethics wrapper. With rule 0, cue 0 and full
view the actions are identical to the Phase 3 implementation.
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

RULES = {"minimal": 0, "sustainable": 1}
CUES = {"in_view": 0, "seen_by_teacher": 1, "near": 2}
SUSTAINABLE_RESERVE = 3   # regrowth is fastest with >= 3 apple neighbours
NEAR_DISTANCE = 5         # lateral half-width of the 11 x 11 view


@struct.dataclass
class ScriptedParams:
    w: jnp.ndarray             # (N,) float32
    performative: jnp.ndarray  # (N,) bool
    aggressive: jnp.ndarray    # (N,) bool
    eps: jnp.ndarray           # (N,) float32
    rule: jnp.ndarray          # (N,) int32
    cue: jnp.ndarray           # (N,) int32
    view_only: jnp.ndarray     # (N,) bool


DISPOSITIONS = ("mixture", "performative", "aggressive")


def make_params(roles: RoleConfig, disposition: str, w: float = 0.0, eps: float = 0.0,
                rule: str = "minimal", cue: str = "in_view", view_only: bool = False
                ) -> ScriptedParams:
    """Teacher virtuous (minimal rule), bystanders indifferent, learners carry the tested group.

    disposition: "mixture" (use w), "performative" or "aggressive".
    rule, cue, view_only and eps apply to learners only.
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
    rv = np.zeros(n, np.int32)
    cv = np.zeros(n, np.int32)
    vo = np.zeros(n, bool)
    if disposition == "mixture":
        wv[m["learner"]] = w
    elif disposition == "performative":
        perf[m["learner"]] = True
    else:
        aggr[m["learner"]] = True
    ev[m["learner"]] = eps
    rv[m["learner"]] = RULES[rule]
    cv[m["learner"]] = CUES[cue]
    vo[m["learner"]] = view_only
    return ScriptedParams(w=jnp.asarray(wv), performative=jnp.asarray(perf),
                          aggressive=jnp.asarray(aggr), eps=jnp.asarray(ev),
                          rule=jnp.asarray(rv), cue=jnp.asarray(cv), view_only=jnp.asarray(vo))


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


def forbidden_sustainable(grid: jnp.ndarray) -> jnp.ndarray:
    """Apples the sustainable rule leaves: those in components of <= reserve apples."""
    return (grid == Items.apple) & (geo.apple_component_sizes(grid) <= SUSTAINABLE_RESERVE)


def _mode_tables(grid, walls, nbrs, with_sustainable: bool):
    """Neighbour distances and forbidden flags for the three modes, for one known grid.

    nbrs: (K, 4, 2) neighbour cells of the K agents using this grid.
    Returns d_all, d_min, d_sus, forb_min, forb_sus, each (K, 4).
    """
    apples = grid == Items.apple
    last = geo.last_apple_mask(grid)
    d_all = geo.lookup(distance_field(~walls, apples), nbrs, fill=INF)
    d_min = geo.lookup(distance_field(~walls & ~last, apples & ~last), nbrs, fill=INF)
    f_min = geo.lookup(last, nbrs, fill=False)
    if with_sustainable:
        fs = forbidden_sustainable(grid)
        d_sus = geo.lookup(distance_field(~walls & ~fs, apples & ~fs), nbrs, fill=INF)
        f_sus = geo.lookup(fs, nbrs, fill=False)
    else:
        d_sus, f_sus = d_min, f_min
    return d_all, d_min, d_sus, f_min, f_sus


class ScriptedPolicy:
    """Callable policy(key, obs, state, params) for batched environments."""

    def __init__(self, env):
        self.env = env

    def _shared_tables(self, state, params, nbrs):
        """Every agent sees the whole map: one set of distance fields for all."""
        grid = state.grid
        walls = grid == Items.wall
        return jax.lax.cond(
            jnp.any(params.rule == RULES["sustainable"]),
            lambda: _mode_tables(grid, walls, nbrs, True),
            lambda: _mode_tables(grid, walls, nbrs, False))

    def _per_agent_tables(self, state, params, nbrs):
        """View-only agents see apples only inside their own view window."""
        env = self.env
        grid = state.grid
        walls = grid == Items.wall
        vis = jax.vmap(geo.view_mask, in_axes=(0, None, None))(
            state.agent_locs, grid.shape, env.obs_size)
        vis = vis | ~params.view_only[:, None, None]
        unseen_apple = (grid[None] == Items.apple) & ~vis
        known = jnp.where(unseen_apple, Items.empty, grid[None])           # (N, H, W)
        tables = jax.vmap(lambda g, nb: _mode_tables(g, walls, nb[None], True))(known, nbrs)
        return tuple(t[:, 0] for t in tables)

    def single(self, key: jax.Array, state, params: ScriptedParams) -> jnp.ndarray:
        env = self.env
        grid = state.grid
        pos = state.reborn_locs.astype(jnp.int32)          # true start-of-step cells
        nbrs = pos[:, None, :2] + MOVE_DELTAS[None]         # (N, 4, 2)
        n = pos.shape[0]
        idx = jnp.arange(n)

        d_all, d_min, d_sus, f_min, f_sus = jax.lax.cond(
            jnp.any(params.view_only),
            lambda: self._per_agent_tables(state, params, nbrs),
            lambda: self._shared_tables(state, params, nbrs))

        k_mode, k_tie, k_lapse, k_pick = jax.random.split(key, 4)
        k_wander = jax.random.fold_in(key, 11)              # extra stream, keeps the
                                                            # Phase 3 streams unchanged
        tiv = env.teacher_in_view(state)
        sbt = env.seen_by_teacher(state)
        teacher = pos[TEACHER, :2]
        near = (jnp.abs(pos[:, :2] - teacher[None]).sum(1) <= NEAR_DISTANCE).at[TEACHER].set(False)
        cue_on = jnp.select([params.cue == 0, params.cue == 1], [tiv, sbt], near)
        virtuous = jnp.where(params.performative, cue_on,
                             jax.random.uniform(k_mode, (n,)) < params.w)

        sustainable = params.rule == RULES["sustainable"]
        d_vir = jnp.where(sustainable[:, None], d_sus, d_min)
        forb = jnp.where(sustainable[:, None], f_sus, f_min)          # (N, 4)

        d_mode = jnp.where(virtuous[:, None], d_vir, d_all)
        d = d_mode.astype(jnp.float32) + jax.random.uniform(k_tie, d_mode.shape, maxval=0.5)
        best = jnp.argmin(d, axis=1)
        reachable = jnp.min(d_mode, axis=1) < INF
        action = jnp.where(reachable, MOVES[best], STAY)

        # View-only agents with nothing suitable in view wander.
        wander_idx = jax.random.randint(k_wander, (n,), 0, 4)
        wander = jnp.where(forb[idx, wander_idx] & virtuous, STAY, MOVES[wander_idx])
        action = jnp.where(params.view_only & ~reachable, wander, action)

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
        # step onto an apple its rule forbids.
        lapse = jax.random.uniform(k_lapse, (n,)) < params.eps
        random_action = LAPSE_ACTIONS[jax.random.randint(k_pick, (n,), 0, LAPSE_ACTIONS.shape[0])]
        move_idx = jnp.argmax(MOVES[None, :] == random_action[:, None], axis=1)
        is_move = jnp.any(MOVES[None, :] == random_action[:, None], axis=1)
        unsafe = is_move & forb[idx, move_idx] & virtuous
        random_action = jnp.where(unsafe, STAY, random_action)
        return jnp.where(lapse, random_action, action).astype(jnp.int32)

    def __call__(self, key, obs, state, params: ScriptedParams):
        num_envs = obs.shape[0]
        keys = jax.random.split(key, num_envs)
        return jax.vmap(self.single, in_axes=(0, 0, None))(keys, state, params)
