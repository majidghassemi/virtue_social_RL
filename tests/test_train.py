"""Trainer, disposition rewards, checkpoints and the learned-slot policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtue_rl.agents import ScriptedPolicy, make_params
from virtue_rl.agents.learned import SlotParams, SlotPolicy
from virtue_rl.envs import RoleConfig
from virtue_rl.envs.maps import load_suite
from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.train import PPOConfig, RewardConfig, Trainer
from virtue_rl.train import checkpoint as ckpt
from virtue_rl.train import rewards as R

ROLES = RoleConfig(num_learners=1, num_bystanders=1)
T = 20


@pytest.fixture(scope="module")
def envs():
    maps = [e for e in load_suite("maps/suite_v2.json") if e.split == "seen"][:2]
    return [EthicsHarvest(ROLES, num_inner_steps=T, map_ASCII=m.ascii(True), grid_size=m.grid_size)
            for m in maps]


def tiny_cfg(**kw):
    base = dict(envs_per_map=2, num_steps=T, total_timesteps=2 * 2 * T * 4, num_minibatches=2)
    base.update(kw)
    return PPOConfig(**base)


# ------------------------------------------------------------- rewards
def test_prosocial_reward():
    r = jnp.array([[1.0, 0.0, 2.0]])
    assert np.allclose(R.prosocial(r, jnp.zeros(3)), r)
    # mean of others: agent0 (0+2)/2=1, agent1 (1+2)/2=1.5, agent2 (1+0)/2=0.5
    assert np.allclose(R.prosocial(r, jnp.full(3, 0.5)), [[1.5, 0.75, 2.25]])
    assert np.allclose(R.prosocial(r, jnp.array([1.0, 0.0, 0.0])), [[2.0, 0.0, 2.0]])


def test_svo_reward_matches_mckee():
    r = jnp.array([1.0, 1.0, 0.0])
    w, target = jnp.full(3, 0.2), jnp.full(3, 45.0)
    u = R.svo(r, w, target)
    # agent 0: mean others 0.5, theta = atan2(0.5, 1)
    expect0 = 1.0 - 0.2 * abs(np.deg2rad(45) - np.arctan2(0.5, 1.0))
    expect2 = 0.0 - 0.2 * abs(np.deg2rad(45) - np.arctan2(1.0, 0.0))
    assert np.isclose(u[0], expect0) and np.isclose(u[2], expect2)


# ------------------------------------------------------------- trainer
def test_trainer_runs_and_learns_something(envs):
    tr = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (1.0, 1.0, 1.0)), train_slots=(0, 1, 2))
    r0 = tr.init(jax.random.PRNGKey(0))
    r1, m = tr.run(r0, 2)
    assert int(r1.update) == 2
    assert all(np.isfinite(np.asarray(v)).all() for v in m.values())
    assert m["apples"].shape == (2, 3)
    p0, p1 = jax.tree.leaves(r0.train_state.params), jax.tree.leaves(r1.train_state.params)
    assert any(not np.allclose(a, b) for a, b in zip(p0, p1))


def test_shaped_return_is_prosocial_transform_of_apples(envs):
    tr = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (0.5, 0.5, 0.5)), train_slots=(0, 1, 2))
    _, m = tr.run(tr.init(jax.random.PRNGKey(1)), 1)
    a = np.asarray(m["apples"][0])
    expect = a + 0.5 * (a.sum() - a) / 2
    assert np.allclose(m["shaped_return"][0], expect, atol=1e-4)


def test_frozen_slot_uses_frozen_params_and_is_not_trained(envs):
    base = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (0.0,) * 3), train_slots=(0, 1, 2))
    frozen = base.init(jax.random.PRNGKey(2)).train_state.params
    tr = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (0.0,) * 3), train_slots=(1,),
                 frozen={0: frozen, 2: frozen})
    assert tr.num_actors == tr.num_envs                        # only slot 1 enters the batch
    r1, m = tr.run(tr.init(jax.random.PRNGKey(3)), 1)
    assert np.isfinite(np.asarray(m["loss_total"])).all()
    with pytest.raises(ValueError):
        Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (0.0,) * 3), train_slots=(1,))


def test_runner_checkpoint_roundtrip(envs, tmp_path):
    tr = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (1.0,) * 3), train_slots=(0, 1, 2))
    r1, _ = tr.run(tr.init(jax.random.PRNGKey(4)), 1)
    ckpt.save_runner(tmp_path / "runner.msgpack", r1)
    template = tr.init(jax.random.PRNGKey(99))
    r1b = ckpt.load_runner(tmp_path / "runner.msgpack", template)
    for a, b in zip(jax.tree.leaves(r1), jax.tree.leaves(r1b)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # Resuming gives exactly the same next update as continuing.
    _, ma = tr.run(r1, 1)
    _, mb = tr.run(r1b, 1)
    assert np.allclose(ma["loss_total"], mb["loss_total"])


def test_params_checkpoint_and_meta(envs, tmp_path):
    tr = Trainer(envs, tiny_cfg(), RewardConfig("prosocial", (1.0,) * 3), train_slots=(0, 1, 2))
    params = tr.init(jax.random.PRNGKey(5)).train_state.params
    ckpt.save_params(tmp_path / "p.msgpack", params, {"lam": 1.0})
    loaded = ckpt.load_params(tmp_path / "p.msgpack", tr.init(jax.random.PRNGKey(6)).train_state.params)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(loaded)))
    assert ckpt.read_meta(tmp_path / "p.msgpack") == {"lam": 1.0}


# ------------------------------------------------------------- slot policy
def test_slot_policy_keeps_scripted_slots(envs):
    env = envs[0]
    pol = SlotPolicy(env, learned_slots=(1,))
    net = pol.init_params(jax.random.PRNGKey(7))
    sp = make_params(ROLES, "mixture", w=0.0)
    obs, state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(8), 4))
    key = jax.random.PRNGKey(9)
    a = pol(key, obs, state, SlotParams(scripted=sp, networks=(net,)))
    k_s, _ = jax.random.split(key)
    a_scripted = ScriptedPolicy(env)(k_s, obs, state, sp)
    assert np.array_equal(a[:, [0, 2]], a_scripted[:, [0, 2]])
    assert ((a[:, 1] >= 0) & (a[:, 1] < env.action_space().n)).all()
