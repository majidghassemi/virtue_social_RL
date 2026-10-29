"""Tests for map generation, the teacher bay, the splits and the per-map runner."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtue_rl.envs import TEACHER, RoleConfig
from virtue_rl.envs import maps as M
from virtue_rl.runner import MapSuite, random_policy, rollout

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "maps" / "suite_v1.json"
N = 7


@pytest.fixture(scope="module")
def entries():
    return M.load_suite(SUITE)


# ------------------------------------------------------------- generation
def test_generation_is_deterministic():
    spec = M.MapSpec(family="rooms", seed=7)
    assert M.generate(spec)[:2] == M.generate(spec)[:2]


def test_every_family_generates_valid_maps():
    for family in M.FAMILIES:
        for seed in range(5):
            present, absent, _ = M.generate(M.MapSpec(family=family, seed=seed))
            M.validate_map(present, N, teacher_present=True)
            M.validate_map(absent, N, teacher_present=False)


def test_conditions_differ_only_in_teacher_start(entries):
    for e in entries:
        p, a = M.to_array(e.ascii_present), M.to_array(e.ascii_absent)
        diff = np.argwhere(p != a)
        assert {tuple(x) for x in diff} == {M.ARENA_Q, (M.BAY_ROW, M.BAY_COL)}


# ------------------------------------------------------------- validation
def _base():
    present, absent, _ = M.generate(M.MapSpec(family="open", seed=0))
    return present, absent


def _edit(ascii_map, cells, ch):
    g = M.to_array(ascii_map)
    for c in cells:
        g[c] = ch
    return ["".join(r) for r in g]


def test_validate_rejects_open_band():
    _, absent = _base()
    opened = _edit(absent, [(M.BAY_ROW, c) for c in range(M.ARENA_COL_RANGE[1], M.BAY_COL)], " ")
    with pytest.raises(ValueError):
        M.validate_map(opened, N, teacher_present=False)


def test_validate_rejects_bay_too_close():
    # Move the bay 1 column closer by carving a cell in the band and putting Q there.
    _, absent = _base()
    g = M.to_array(absent)
    g[M.BAY_ROW, M.BAY_COL] = "W"
    near = M.BAY_COL - 2  # within 9 columns of the last arena column
    g[M.BAY_ROW, near] = "Q"
    moved = ["".join(r) for r in g]
    with pytest.raises(ValueError):
        M.validate_map(moved, N, teacher_present=False)


def test_validate_rejects_wrong_condition_and_counts():
    present, absent = _base()
    with pytest.raises(ValueError):
        M.validate_map(present, N, teacher_present=False)
    with pytest.raises(ValueError):
        M.validate_map(absent, N, teacher_present=True)
    with pytest.raises(ValueError):
        M.validate_map(_edit(present, [(12, 3)], "Q"), N, teacher_present=True)  # two Q
    no_spawns = [row.replace("P", " ") for row in present]
    with pytest.raises(ValueError):
        M.validate_map(no_spawns, N, teacher_present=True)


# -------------------------------------------------------- structural cost
def _tiny(apples):
    g = np.full((M.ARENA_ROWS, M.GRID_COLS), " ")
    for a in apples:
        g[a] = "A"
    return ["".join(r) for r in g]


@pytest.mark.parametrize("apples, comps, cost", [
    ([(3, 3)], 1, 1.0),                                   # single apple: all of it is "last"
    ([(3, 3), (3, 9)], 2, 1.0),                           # two isolated singles
    ([(3, 3), (3, 5)], 1, 0.5),                           # two apart: same neighbourhood
    ([(3, 3), (3, 6)], 2, 1.0),                           # three apart: separate
    ([(5 + dr, 5 + dc) for dr in range(-2, 3) for dc in range(-2, 3)
      if abs(dr) + abs(dc) <= 2], 1, 1 / 13),             # radius-2 diamond
])
def test_structural_virtue_cost(apples, comps, cost):
    m = _tiny(apples)
    assert M.components(m) == comps
    assert M.structural_virtue_cost(m) == pytest.approx(cost)


# ----------------------------------------------------------------- splits
def test_split_seeds_are_disjoint(entries):
    seeds = {}
    for e in entries:
        seeds.setdefault(e.split, set()).add(e.seed)
    s = list(seeds.values())
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            assert not (s[i] & s[j])


def test_heldout_position_is_new_layout(entries):
    def cells(e):
        return {tuple(p) for p in np.argwhere(M.to_array(e.ascii_present) == "A")}
    seen = [cells(e) for e in entries if e.split == "seen"]
    for e in entries:
        if e.split == "heldout_position":
            c = cells(e)
            assert all(len(c & s) / min(len(c), len(s)) <= 0.5 for s in seen)


def test_heldout_structure_families_unseen(entries):
    seen_fams = {e.family for e in entries if e.split == "seen"}
    held = {e.family for e in entries if e.split == "heldout_structure"}
    assert held and not (held & seen_fams)


def test_heldout_splits_balanced_across_bands(entries):
    for split in ("heldout_position", "heldout_structure"):
        bands = [e.band for e in entries if e.split == split]
        counts = {b: bands.count(b) for b in ("low", "medium", "high")}
        assert len(set(counts.values())) == 1, counts


# ------------------------------------------------- environment integration
@pytest.fixture(scope="module")
def suite():
    return MapSuite(str(SUITE), RoleConfig(num_learners=3, num_bystanders=3), num_inner_steps=200)


def test_teacher_absent_is_sealed_and_unseen(suite):
    name = suite.split("heldout_structure")[0].name  # a walled map
    env = suite.env(name, teacher_present=False)
    traj = rollout(env, random_policy, jax.random.PRNGKey(0), num_envs=4, num_steps=120,
                   keep_states=True)
    teacher_rc = traj["states"].agent_locs[:, :, TEACHER, :2]
    assert jnp.all(teacher_rc == jnp.array([M.BAY_ROW, M.BAY_COL]))
    assert not jnp.any(traj["ethics"].teacher_in_view)
    assert jnp.sum(traj["ethics"].apples) > 0  # the others still play


def test_teacher_present_starts_in_arena(suite):
    name = suite.split("seen")[0].name
    env = suite.env(name, teacher_present=True)
    for seed in range(3):
        _, state = env.reset(jax.random.PRNGKey(seed))
        assert tuple(map(int, state.agent_locs[TEACHER, :2])) == M.ARENA_Q
    traj = rollout(env, random_policy, jax.random.PRNGKey(1), num_envs=4, num_steps=120)
    assert jnp.any(traj["ethics"].teacher_in_view)


def test_env_grid_matches_map(suite):
    e = suite.entries[0]
    env = suite.env(e.name, True)
    assert (env.grid_rows, env.grid_cols) == e.grid_size


def _set_agents(env, state, locs):
    """Place agents at (row, col, dir) on the map's own grid."""
    locs = jnp.array(locs, jnp.int16)
    g = jnp.where(jnp.isin(state.grid, env.env._agents), 0, state.grid).astype(state.grid.dtype)
    g = g.at[locs[:, 0], locs[:, 1]].set(env.env._agents)
    return state.replace(grid=g, agent_locs=locs, reborn_locs=locs)


def test_wall_ring_blocks_edge_wraparound_zaps(suite):
    """Regression test for the upstream edge bug (see NOTES.md, Zapping)."""
    name = suite.split("seen")[0].name
    env = suite.env(name, teacher_present=False)
    _, s = env.reset(jax.random.PRNGKey(0))
    top, bottom = M.RING, M.GRID_ROWS - 1 - M.RING
    col = M.RING + 3
    others = [(bottom - 1, M.RING + 7 + 2 * i, 0) for i in range(4)]
    # Teacher in the bay facing the right edge; agent 1 on the top arena row
    # facing up (dir 2 = -row); agent 2 on the bottom arena row, same column.
    s = _set_agents(env, s, [(M.BAY_ROW, M.BAY_COL, 1), (top, col, 2), (bottom, col, 0)] + others)
    actions = jnp.array([7, 7, 6, 6, 6, 6, 6])
    _, post, _, _, _, e = jax.jit(env.step)(jax.random.PRNGKey(1), s, actions)
    assert not bool(e.harm_zap[0]) and not bool(e.harm_zap[1])
    assert jnp.array_equal(post.reborn_locs[0, :2], jnp.array([M.BAY_ROW, M.BAY_COL]))  # no self-hit
    assert jnp.array_equal(post.reborn_locs[2, :2], jnp.array([bottom, col]))           # no wrap hit
