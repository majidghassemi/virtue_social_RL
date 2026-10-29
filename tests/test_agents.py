"""Tests for the scripted known-groups agents."""

from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtue_rl.agents import ScriptedPolicy, distance_field, make_params
from virtue_rl.agents.scripted import LAPSE_ACTIONS
from virtue_rl.envs import RoleConfig
from virtue_rl.runner import MapSuite, rollout

ROLES = RoleConfig(num_learners=3, num_bystanders=3)
LEARNERS = ROLES.masks()["learner"]
SUITE = Path(__file__).parents[1] / "maps" / "suite_v1.json"


def bfs(passable, targets):
    h, w = passable.shape
    d = np.full((h, w), 1_000_000, np.int64)
    q = deque()
    for r, c in np.argwhere(targets & passable):
        d[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and d[nr, nc] > d[r, c] + 1:
                d[nr, nc] = d[r, c] + 1
                q.append((nr, nc))
    return d


@pytest.mark.parametrize("seed", range(5))
def test_distance_field_matches_bfs(seed):
    rng = np.random.default_rng(seed)
    passable = rng.random((12, 17)) > 0.3
    targets = rng.random((12, 17)) > 0.95
    got = np.asarray(distance_field(jnp.asarray(passable), jnp.asarray(targets)))
    assert np.array_equal(got, bfs(passable, targets))


def test_make_params_assigns_roles():
    p = make_params(ROLES, "mixture", w=0.5, eps=0.2)
    assert p.w.tolist() == [1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]      # teacher virtuous, bystanders indifferent
    assert np.allclose(p.eps, [0, 0.2, 0.2, 0.2, 0, 0, 0])            # noise only on learners
    assert make_params(ROLES, "performative").performative.tolist() == [False, True, True, True, False, False, False]
    assert make_params(ROLES, "aggressive").aggressive.tolist() == [False, True, True, True, False, False, False]


@pytest.fixture(scope="module")
def suite():
    return MapSuite(str(SUITE), ROLES, num_inner_steps=400)


def run(suite, name, present, params, key=0, num_envs=8, num_steps=400):
    env = suite.env(name, present)
    policy = _policy(env)
    return rollout(env, policy, jax.random.PRNGKey(key), num_envs=num_envs, num_steps=num_steps,
                   policy_params=params)


_POLICIES = {}


def _policy(env):
    if env not in _POLICIES:
        _POLICIES[env] = ScriptedPolicy(env)
    return _POLICIES[env]


HIGH_COST = "seen_03_open"


def learner_sum(x):
    return int(jnp.sum(x[..., LEARNERS]))


def test_virtuous_never_harms(suite):
    for eps in (0.0, 0.5):
        e = run(suite, HIGH_COST, True, make_params(ROLES, "mixture", 1.0, eps))["ethics"]
        assert learner_sum(e.harm_commons) == 0
        assert learner_sum(e.harm_zap) == 0
        assert learner_sum(e.apples) > 0


def test_indifferent_harms_but_never_zaps(suite):
    e = run(suite, HIGH_COST, False, make_params(ROLES, "mixture", 0.0))["ethics"]
    assert learner_sum(e.harm_commons) > 0
    assert learner_sum(e.harm_zap) == 0


def test_teacher_and_bystanders_follow_their_roles(suite):
    e = run(suite, HIGH_COST, True, make_params(ROLES, "mixture", 0.5))["ethics"]
    assert int(jnp.sum(e.harm_commons[..., 0])) == 0          # teacher is virtuous
    assert int(jnp.sum(e.harm_zap)) == 0                       # nobody zaps


def test_aggressive_zaps(suite):
    e = run(suite, HIGH_COST, False, make_params(ROLES, "aggressive"))["ethics"]
    assert learner_sum(e.harm_zap) > 0


def test_performative_only_harms_unwatched(suite):
    e = run(suite, HIGH_COST, True, make_params(ROLES, "performative"))["ethics"]
    harm_in_view = e.harm_commons & e.teacher_in_view
    assert learner_sum(harm_in_view) == 0
    assert learner_sum(e.teacher_in_view) > 0


def test_performative_equals_indifferent_without_teacher(suite):
    a = run(suite, HIGH_COST, False, make_params(ROLES, "performative"), key=3)
    b = run(suite, HIGH_COST, False, make_params(ROLES, "mixture", 0.0), key=3)
    assert jnp.array_equal(a["actions"], b["actions"])


def test_lapses_never_zap_or_turn(suite):
    t = run(suite, HIGH_COST, False, make_params(ROLES, "mixture", 0.0, eps=1.0), num_steps=100)
    acts = t["actions"][..., LEARNERS]
    assert bool(jnp.all(jnp.isin(acts, LAPSE_ACTIONS)))


def test_mixture_is_graded(suite):
    vals = []
    for w in (0.0, 0.5, 1.0):
        e = run(suite, HIGH_COST, False, make_params(ROLES, "mixture", w), num_envs=16)["ethics"]
        vals.append(learner_sum(e.harm_commons) / max(1, learner_sum(e.opp_commons)))
    assert vals[0] > vals[1] > vals[2] == 0
