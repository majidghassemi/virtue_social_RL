"""Tests for the Phase 4 known groups: sustainable rule, view-only agents, watched cues."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from socialjax.environments.common_harvest.harvest_open import Items

from virtue_rl.agents import ScriptedPolicy, make_params
from virtue_rl.agents.scripted import NEAR_DISTANCE, SUSTAINABLE_RESERVE
from virtue_rl.envs import RoleConfig
from virtue_rl.envs import geometry as geo
from virtue_rl.runner import MapSuite, rollout

ROLES = RoleConfig(3, 3)
LEARNERS = list(ROLES.learner_ids)
SUITE = Path(__file__).parents[1] / "maps" / "suite_v1.json"
MAP = "seen_03_open"


@pytest.fixture(scope="module")
def suite():
    return MapSuite(str(SUITE), ROLES, num_inner_steps=300)


_POL = {}


def run(suite, present, params, key=0, num_envs=6, num_steps=300, keep_states=False):
    env = suite.env(MAP, present)
    _POL.setdefault(env, ScriptedPolicy(env))
    return rollout(env, _POL[env], jax.random.PRNGKey(key), num_envs=num_envs,
                   num_steps=num_steps, keep_states=keep_states, policy_params=params)


def grid_with(apples, shape=(9, 12)):
    g = np.zeros(shape, np.int32)
    for r, c in apples:
        g[r, c] = Items.apple
    return jnp.asarray(g)


# ------------------------------------------------------------------ geometry
def test_component_sizes_hand_built():
    plus = [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)]          # 5 apples, one component
    single = [(7, 10)]                                         # isolated apple
    pair = [(6, 5), (6, 7)]                                    # distance 2 apart: linked
    s = np.asarray(geo.apple_component_sizes(grid_with(plus + single + pair)))
    assert all(s[r, c] == 5 for r, c in plus)
    assert s[7, 10] == 1
    assert s[6, 5] == s[6, 7] == 2
    assert s[0, 0] == 0                                        # not an apple


def test_component_size_one_is_a_last_apple():
    rng = np.random.default_rng(0)
    for _ in range(5):
        cells = {tuple(x) for x in rng.integers(0, [9, 12], size=(15, 2))}
        g = grid_with(cells)
        size1 = np.asarray(geo.apple_component_sizes(g)) == 1
        assert np.array_equal(size1, np.asarray(geo.last_apple_mask(g)))


def test_view_mask_matches_agent_views(suite):
    env = suite.env(MAP, True)
    _, state = env.reset(jax.random.PRNGKey(1))
    h, w = state.grid.shape
    index_grid = jnp.arange(h * w).reshape(h, w) + 100       # avoid the wall value
    for loc in [jnp.array([5, 6, d]) for d in range(4)] + [jnp.array([1, 1, 0]),
                                                             jnp.array([16, 30, 2])]:
        views = geo.agent_views(index_grid, loc[None], env.obs_size)[0]
        from_views = {int(v) - 100 for v in np.asarray(views).ravel() if v >= 100}
        mask = np.asarray(geo.view_mask(loc, (h, w), env.obs_size))
        assert from_views == set(np.flatnonzero(mask.ravel()).tolist())


# ------------------------------------------------------------------ agents
def test_sustainable_keeps_a_reserve(suite):
    t = run(suite, False, make_params(ROLES, "mixture", 1.0, rule="sustainable"), keep_states=True)
    e, states = t["ethics"], t["states"]
    sizes = jax.vmap(jax.vmap(geo.apple_component_sizes))(states.grid)           # (T, E, H, W)
    post_locs = states.agent_locs[1:]          # the cell an apple was taken from at step t
    T, E = e.apples.shape[:2]
    taken = 0
    for ti in range(T - 1):
        for ei in range(E):
            for a in LEARNERS:
                if int(e.apples[ti, ei, a]):
                    r, c = (int(x) for x in post_locs[ti, ei, a, :2])
                    assert int(sizes[ti, ei, r, c]) > SUSTAINABLE_RESERVE
                    taken += 1
    assert taken > 0
    assert int(jnp.sum(e.harm_commons[..., LEARNERS])) == 0


def test_view_only_virtuous_never_harms(suite):
    for rule in ("minimal", "sustainable"):
        e = run(suite, True, make_params(ROLES, "mixture", 1.0, eps=0.2, rule=rule,
                                         view_only=True))["ethics"]
        assert int(jnp.sum(e.harm_commons[..., LEARNERS])) == 0
        assert int(jnp.sum(e.apples[..., LEARNERS])) > 0


def test_view_only_indifferent_still_harms(suite):
    e = run(suite, False, make_params(ROLES, "mixture", 0.0, view_only=True))["ethics"]
    assert int(jnp.sum(e.harm_commons[..., LEARNERS])) > 0


def test_per_agent_tables_match_shared_without_view_only(suite):
    env = suite.env(MAP, True)
    pol = ScriptedPolicy(env)
    _, state = env.reset(jax.random.PRNGKey(2))
    params = make_params(ROLES, "mixture", 1.0, rule="sustainable")
    pos = state.reborn_locs.astype(jnp.int32)
    from virtue_rl.agents.scripted import MOVE_DELTAS
    nbrs = pos[:, None, :2] + MOVE_DELTAS[None]
    a = pol._shared_tables(state, params, nbrs)
    b = pol._per_agent_tables(state, params, nbrs)
    for x, y in zip(a, b):
        assert jnp.array_equal(x, y)


@pytest.mark.parametrize("cue", ["in_view", "seen_by_teacher", "near"])
def test_performative_only_harms_when_cue_is_off(suite, cue):
    t = run(suite, True, make_params(ROLES, "performative", cue=cue), keep_states=True)
    e, states = t["ethics"], t["states"]
    if cue == "in_view":
        on = e.teacher_in_view
    elif cue == "seen_by_teacher":
        on = e.seen_by_teacher
    else:
        pos = states.reborn_locs[..., :2].astype(jnp.int32)
        dist = jnp.abs(pos - pos[..., :1, :]).sum(-1)
        on = dist <= NEAR_DISTANCE
    assert int(jnp.sum((e.harm_commons & on)[..., LEARNERS])) == 0
    assert int(jnp.sum(on[..., LEARNERS])) > 0


def test_regrowth_and_stock_are_environment_level(suite):
    e = run(suite, False, make_params(ROLES, "mixture", 0.5))["ethics"]
    assert bool(jnp.all(e.regrown == e.regrown[..., :1]))
    assert bool(jnp.all(e.stock == e.stock[..., :1]))
    assert int(jnp.sum(e.regrown[..., 0])) > 0
    # Stock bookkeeping: next stock = stock - apples taken + regrown (valid steps).
    s, r, a = (np.asarray(x) for x in (e.stock[..., 0], e.regrown[..., 0], e.apples.sum(-1)))
    assert np.array_equal(s[1:-1], (s[:-1] - a[:-1] + r[:-1])[:-1])
