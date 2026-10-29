"""Tests for the Commons Harvest ethics wrapper.

Two kinds of test:
1. Hand-built grids, where the correct answer is known by construction.
2. Random rollouts, where the wrapper is cross-checked against the
   environment's own observations, rewards and respawn behaviour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from socialjax.environments.common_harvest.harvest_open import STEP_MOVE, Actions, Items

from virtue_rl.envs import EthicsHarvest, RoleConfig, TEACHER
from virtue_rl.envs import geometry as geo

STAY = int(Actions.stay)
ZAP = int(Actions.zap_forward)

# 9 x 11 test map. One apple spawn in the top-left corner (kept filled so no
# regrowth happens near the test area) and spawn points on the bottom row.
SMALL_MAP = [
    "A          ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "           ",
    "PPPPP      ",
]
SMALL_ROLES = RoleConfig(num_learners=1, num_bystanders=1)  # teacher 0, learner 1, bystander 2


@pytest.fixture(scope="module")
def small():
    return EthicsHarvest(SMALL_ROLES, map_ASCII=SMALL_MAP, grid_size=(9, 11))


@pytest.fixture(scope="module")
def small_step(small):
    return jax.jit(small.step)


def move_action(dr: int, dc: int) -> int:
    """Action index that moves an agent by (dr, dc)."""
    for a in range(STEP_MOVE.shape[0]):
        if a in (2, 3, 4, 5) and int(STEP_MOVE[a, 0]) == dr and int(STEP_MOVE[a, 1]) == dc:
            return a
    raise ValueError((dr, dc))


def make_state(w: EthicsHarvest, agents, apples, invs=None, inner_t=0):
    """Hand-built state. agents: list of (row, col, dir), apples: list of (row, col)."""
    _, template = w.env.reset(jax.random.PRNGKey(0))
    grid = jnp.zeros((w.grid_rows, w.grid_cols), jnp.int16)
    grid = grid.at[0, 0].set(Items.apple)  # the map's only apple spawn stays filled
    for r, c in apples:
        grid = grid.at[r, c].set(Items.apple)
    locs = jnp.array(agents, dtype=jnp.int16)
    grid = grid.at[locs[:, 0], locs[:, 1]].set(w.env._agents)
    if invs is None:
        invs = jnp.zeros((w.num_agents, 2), jnp.int8)
    return template.replace(grid=grid, agent_locs=locs, reborn_locs=locs,
                            agent_invs=jnp.asarray(invs, jnp.int8), inner_t=inner_t)


def step_with(w, step_fn, state, actions, seed=0):
    return step_fn(jax.random.PRNGKey(seed), state, jnp.array(actions, jnp.int32))


# --------------------------------------------------------------------- commons
def test_last_apple_counts_as_harm_commons(small, small_step):
    s = make_state(small, agents=[(2, 1, 0), (4, 4, 0), (6, 9, 0)], apples=[(4, 5)])
    _, post, rew, _, _, e = step_with(small, small_step, s, [STAY, move_action(0, 1), STAY])
    assert int(e.apples[1]) == 1
    assert bool(e.harm_commons[1]) and bool(e.opp_commons[1])
    assert float(rew[1]) == 1 * small.num_agents  # upstream scales individual rewards by N
    assert not bool(e.harm_commons[0]) and not bool(e.harm_commons[2])


def test_apple_with_neighbour_is_not_harm(small, small_step):
    # (4, 7) is two cells from (4, 5), inside the radius-2 neighbourhood.
    s = make_state(small, agents=[(2, 1, 0), (4, 4, 0), (6, 9, 0)], apples=[(4, 5), (4, 7)])
    _, _, _, _, _, e = step_with(small, small_step, s, [STAY, move_action(0, 1), STAY])
    assert int(e.apples[1]) == 1
    assert not bool(e.harm_commons[1])
    assert not bool(e.opp_commons[1])


def test_diagonal_neighbour_also_protects(small, small_step):
    s = make_state(small, agents=[(2, 1, 0), (4, 4, 0), (6, 9, 0)], apples=[(4, 5), (5, 6)])
    _, _, _, _, _, e = step_with(small, small_step, s, [STAY, move_action(0, 1), STAY])
    assert int(e.apples[1]) == 1 and not bool(e.harm_commons[1])


def test_opportunity_without_harm(small, small_step):
    s = make_state(small, agents=[(2, 1, 0), (4, 4, 0), (6, 9, 0)], apples=[(4, 5)])
    _, _, _, _, _, e = step_with(small, small_step, s, [STAY, STAY, STAY])
    assert bool(e.opp_commons[1]) and not bool(e.harm_commons[1]) and int(e.apples[1]) == 0


def test_inventory_wraparound_is_handled(small, small_step):
    invs = np.zeros((3, 2), np.int8)
    invs[1] = 127  # int8 maximum, next apple wraps to -128 upstream
    s = make_state(small, agents=[(2, 1, 0), (4, 4, 0), (6, 9, 0)], apples=[(4, 5)], invs=invs)
    _, post, _, _, _, e = step_with(small, small_step, s, [STAY, move_action(0, 1), STAY])
    assert int(post.agent_invs[1, 0]) == -128
    assert int(e.apples[1]) == 1


# ------------------------------------------------------------------------ zap
@pytest.mark.parametrize("victim_cell, expect_hit", [
    ((3, 5), True),    # one ahead (dir 0 faces +row)
    ((4, 5), True),    # two ahead
    ((3, 6), True),    # ahead-right diagonal
    ((3, 4), True),    # ahead-left diagonal
    ((2, 6), False),   # beside
    ((1, 5), False),   # behind
    ((5, 5), False),   # three ahead
])
def test_zap_geometry(small, small_step, victim_cell, expect_hit):
    s = make_state(small, agents=[(2, 5, 0), (victim_cell[0], victim_cell[1], 0), (6, 9, 0)],
                   apples=[])
    _, post, _, _, _, e = step_with(small, small_step, s, [ZAP, STAY, STAY])
    assert bool(e.harm_zap[0]) == expect_hit
    assert bool(e.opp_zap[0]) == expect_hit
    assert not bool(e.harm_zap[1]) and not bool(e.harm_zap[2])
    spawn = {tuple(map(int, p)) for p in small.spawn_points}
    respawned = tuple(map(int, post.reborn_locs[1, :2]))
    if expect_hit:
        # The environment sends the victim to a spawn point.
        assert respawned in spawn
    else:
        assert respawned == victim_cell


def test_zap_victim_appears_at_spawn_next_step(small, small_step):
    s = make_state(small, agents=[(2, 5, 0), (3, 5, 0), (6, 9, 0)], apples=[])
    _, post, _, _, _, e = step_with(small, small_step, s, [ZAP, STAY, STAY])
    target = post.reborn_locs[1, :2]
    _, post2, _, _, _, _ = step_with(small, small_step, post, [STAY, STAY, STAY], seed=1)
    assert jnp.array_equal(post2.agent_locs[1, :2], target)


def test_no_zap_no_harm(small, small_step):
    s = make_state(small, agents=[(2, 5, 0), (3, 5, 0), (6, 9, 0)], apples=[])
    _, _, _, _, _, e = step_with(small, small_step, s, [STAY, STAY, STAY])
    assert bool(e.opp_zap[0]) and not bool(e.harm_zap[0])


# ------------------------------------------------------------- teacher in view
@pytest.mark.parametrize("teacher_cell, visible", [
    ((7, 5), True),    # 3 ahead
    ((3, 5), True),    # 1 behind (window extends one cell back)
    ((2, 5), False),   # 2 behind
    ((4, 10), True),   # 5 to the side
])
def test_teacher_in_view_window(small, teacher_cell, visible):
    # Learner at (4, 5) facing dir 0 (+row).
    s = make_state(small, agents=[(teacher_cell[0], teacher_cell[1], 0), (4, 5, 0), (8, 9, 0)],
                   apples=[])
    tiv = small.teacher_in_view(s)
    assert bool(tiv[1]) == visible
    assert not bool(tiv[TEACHER])


# ------------------------------------------------- cross-checks on rollouts
@pytest.fixture(scope="module")
def rollout():
    """Random-policy rollout on the default map, 8 envs x 150 steps."""
    w = EthicsHarvest(RoleConfig(num_learners=3, num_bystanders=3))
    num_envs, num_steps, n = 8, 150, w.num_agents

    def run(key):
        key, k = jax.random.split(key)
        obs, state = jax.vmap(w.reset)(jax.random.split(k, num_envs))

        def body(carry, _):
            obs, state, key = carry
            key, ka, ks = jax.random.split(key, 3)
            actions = jax.random.randint(ka, (num_envs, n), 0, 8)
            nobs, nstate, rew, done, info, e = jax.vmap(w.step)(
                jax.random.split(ks, num_envs), state, actions)
            out = dict(obs=obs, state=state, actions=actions, next_state=nstate,
                       rew=rew, ethics=e)
            return (nobs, nstate, key), out

        _, traj = jax.lax.scan(body, (obs, state, key), None, length=num_steps)
        return traj

    return w, jax.jit(run)(jax.random.PRNGKey(42))


def _flat(x):
    return x.reshape((-1,) + x.shape[2:])  # (T, E, ...) -> (T*E, ...)


def test_views_match_environment_observation_channels(rollout):
    """Our window and rotation reproduce upstream's self (5) and other (6) channels."""
    w, traj = rollout
    obs = _flat(traj["obs"])                       # (B, N, O, O, 16)
    grid = _flat(traj["state"].grid)
    locs = _flat(traj["state"].agent_locs)
    views = jax.vmap(geo.agent_views, in_axes=(0, 0, None))(grid, locs, w.obs_size)
    ids = w.agent_ids
    self_map = views == ids[None, :, None, None]
    any_agent = jnp.isin(views, ids)
    other_map = any_agent & ~self_map
    assert jnp.array_equal(self_map.astype(jnp.int8), obs[..., 5].astype(jnp.int8))
    assert jnp.array_equal(other_map.astype(jnp.int8), obs[..., 6].astype(jnp.int8))


def test_teacher_channel_consistent(rollout):
    w, traj = rollout
    obs = _flat(traj["obs"])
    teacher_ch = obs[..., 15]
    other_ch = obs[..., 6]
    self_ch = obs[..., 5]
    # For the teacher, its teacher channel is its self channel.
    assert jnp.array_equal(teacher_ch[:, TEACHER], self_ch[:, TEACHER])
    # For everyone else, the teacher is one of the "other agent" cells.
    rest = teacher_ch[:, 1:]
    assert jnp.all(rest <= other_ch[:, 1:])
    # teacher_in_view is exactly "teacher channel non-empty" in the obs acted on.
    tiv = _flat(traj["ethics"].teacher_in_view)
    valid = _flat(traj["ethics"].valid)
    seen = jnp.any(rest > 0, axis=(2, 3))
    assert jnp.array_equal(tiv[:, 1:] & valid[:, 1:], seen & valid[:, 1:])
    assert jnp.any(seen)  # the rollout actually exercised the case


def test_apples_match_rewards(rollout):
    w, traj = rollout
    e = traj["ethics"]
    valid = e.valid
    expected = jnp.where(valid, traj["rew"], 0)
    assert jnp.array_equal(e.apples * w.num_agents, expected.astype(jnp.int32))
    assert jnp.sum(e.apples) > 0


def test_harm_implies_its_action(rollout):
    w, traj = rollout
    e = traj["ethics"]
    assert jnp.all(~e.harm_zap | (traj["actions"] == ZAP))
    assert jnp.all(~e.harm_commons | (e.apples > 0))


def test_harm_zap_matches_environment_respawns(rollout):
    """Agents the environment relocates are exactly those our hit test marks as victims."""
    w, traj = rollout
    post = traj["next_state"]
    e = traj["ethics"]
    actions = traj["actions"]
    grid = _flat(post.grid)
    locs = _flat(post.agent_locs)
    reborn = _flat(post.reborn_locs)
    acts = _flat(actions)
    valid = _flat(e.valid)[:, 0]

    targets = jax.vmap(jax.vmap(geo.zap_target_cells, in_axes=(0, None, None)),
                       in_axes=(0, None, None))(locs, w.grid_rows, w.grid_cols)
    hit_values = jax.vmap(geo.upstream_grid_lookup)(grid, targets)          # (B, N, 4)
    zapping = acts == ZAP
    ids = w.agent_ids
    victim = jnp.any(zapping[:, :, None, None] &
                     (hit_values[:, :, :, None] == ids[None, None, None, :]), axis=(1, 2))
    moved = jnp.any(reborn[..., :2] != locs[..., :2], axis=-1)
    v, m = victim[valid], moved[valid]
    assert jnp.all(~m | v)               # relocated agents are victims
    assert jnp.sum(v) > 0                # the rollout produced zap hits
    # Any agent with harm_zap hit at least one victim.
    hz = _flat(e.harm_zap)[valid]
    assert jnp.all(~jnp.any(hz, axis=1) | jnp.any(v, axis=1))


def test_commons_harm_implies_opportunity(rollout):
    """Collecting a last apple needs a move into an adjacent cell, so harm implies opportunity."""
    _, traj = rollout
    e = traj["ethics"]
    assert jnp.all(~e.harm_commons | e.opp_commons)


def test_zap_harm_without_opportunity_is_explained_by_movement(rollout):
    """A zap hit with no recorded opportunity must come from a victim stepping into range.

    Opportunity is measured on start-of-step positions and harm on post-movement
    positions, so the only way to get a hit without an opportunity is another
    agent moving into one of the zap cells during the same step.
    """
    w, traj = rollout
    e = traj["ethics"]
    pre, post = traj["state"], traj["next_state"]
    orphan = np.argwhere(np.asarray(e.harm_zap & ~e.opp_zap))
    for t, env, i in orphan:
        p = jax.tree.map(lambda x: x[t, env], pre)
        q = jax.tree.map(lambda x: x[t, env], post)
        cells = np.asarray(geo.zap_target_cells(q.agent_locs[i], w.grid_rows, w.grid_cols))[:, :2]
        explained = False
        for j in range(w.num_agents):
            if j == i:
                continue
            now = np.asarray(q.agent_locs[j, :2])
            before = np.asarray(p.reborn_locs[j, :2])
            in_now = any((now == c).all() for c in cells)
            in_before = any((before == c).all() for c in cells)
            explained |= in_now and not in_before
        assert explained, (t, env, i)


# ------------------------------------------------------- episode end, jit/vmap
def test_last_step_is_invalid():
    w = EthicsHarvest(SMALL_ROLES, map_ASCII=SMALL_MAP, grid_size=(9, 11), num_inner_steps=5)
    step = jax.jit(w.step)
    _, s = w.reset(jax.random.PRNGKey(0))
    valids, dones = [], []
    for t in range(5):
        _, s, _, d, _, e = step(jax.random.PRNGKey(t), s, jnp.full((3,), STAY))
        valids.append(bool(e.valid[0]))
        dones.append(bool(d["__all__"]))
    assert valids == [True, True, True, True, False]
    assert dones == [False, False, False, False, True]


def test_jit_and_vmap(small):
    num_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(3), num_envs)
    obs, state = jax.vmap(small.reset)(keys)
    assert obs.shape == (num_envs, 3, 11, 11, 16)
    step = jax.jit(jax.vmap(small.step))
    obs, state, rew, done, info, e = step(keys, state, jnp.full((num_envs, 3), STAY))
    assert obs.shape == (num_envs, 3, 11, 11, 16)
    assert e.harm_zap.shape == (num_envs, 3)


def test_no_teacher_channel_option():
    w = EthicsHarvest(SMALL_ROLES, add_teacher_channel=False, map_ASCII=SMALL_MAP,
                      grid_size=(9, 11))
    obs, _ = w.reset(jax.random.PRNGKey(0))
    assert obs.shape[-1] == 15


def test_rejects_shared_rewards():
    with pytest.raises(ValueError):
        EthicsHarvest(SMALL_ROLES, shared_rewards=True)


def test_roles_from_config():
    from pathlib import Path
    from virtue_rl.envs import make_from_config
    w = make_from_config(Path(__file__).parents[1] / "configs" / "harvest_default.yaml")
    assert w.num_agents == 7
    masks = w.roles.masks()
    assert masks["teacher"].tolist() == [True] + [False] * 6
    assert masks["learner"].tolist() == [False, True, True, True, False, False, False]
    assert masks["bystander"].tolist() == [False] * 4 + [True] * 3


# ------------------------------------------------------------ seen by teacher
@pytest.mark.parametrize("teacher, learner, seen, sees", [
    # Teacher at (4,5) facing dir 0 (+row): it sees 1 behind and 9 ahead.
    ((4, 5, 0), (7, 5, 0), True, False),   # learner 3 ahead of teacher, facing away: seen, does not see
    ((4, 5, 0), (2, 5, 0), False, True),   # learner 2 behind teacher, facing it: sees, unseen
    ((4, 5, 2), (7, 5, 0), False, False),  # back to back: neither sees the other
])
def test_seen_by_teacher_is_not_symmetric(small, teacher, learner, seen, sees):
    s = make_state(small, agents=[teacher, learner, (8, 9, 0)], apples=[])
    assert bool(small.seen_by_teacher(s)[1]) == seen
    assert bool(small.teacher_in_view(s)[1]) == sees
    assert not bool(small.seen_by_teacher(s)[TEACHER])


def test_seen_by_teacher_matches_teacher_observation(rollout):
    """seen_by_teacher[i] equals 'agent i appears in the teacher's raw view'."""
    w, traj = rollout
    grid = _flat(traj["state"].grid)
    locs = _flat(traj["state"].agent_locs)
    views = jax.vmap(geo.agent_views, in_axes=(0, 0, None))(grid, locs, w.obs_size)[:, TEACHER]
    expected = jnp.any(views[:, None] == w.agent_ids[None, :, None, None], axis=(2, 3))
    expected = expected.at[:, TEACHER].set(False)
    got = _flat(traj["ethics"].seen_by_teacher)
    valid = _flat(traj["ethics"].valid)
    assert jnp.array_equal(got & valid, expected & valid)
    # The teacher's "other agent" observation channel agrees too.
    obs_other = _flat(traj["obs"])[:, TEACHER, ..., 6]
    assert jnp.array_equal(jnp.any(obs_other > 0, axis=(1, 2)) & valid[:, 0],
                           jnp.any(expected, axis=1) & valid[:, 0])
