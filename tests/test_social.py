"""Social learner: network, sequence batching, auxiliary loss, frozen teacher."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from virtue_rl.envs import RoleConfig
from virtue_rl.envs.maps import load_suite
from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.train import PPOConfig, RewardConfig, Trainer
from virtue_rl.train.social import SocialConfig, SocialNet, SocialTrainer, zero_carry

ROLES = RoleConfig(num_learners=1, num_bystanders=1)
T = 20


@pytest.fixture(scope="module")
def envs():
    maps = [e for e in load_suite("maps/suite_v2.json") if e.split == "seen"][:2]
    return [EthicsHarvest(ROLES, num_inner_steps=T, map_ASCII=m.ascii(True), grid_size=m.grid_size)
            for m in maps]


def cfg():
    return PPOConfig(envs_per_map=2, num_steps=T, total_timesteps=2 * 2 * T * 4, num_minibatches=2)


@pytest.fixture(scope="module")
def frozen(envs):
    t = Trainer(envs, cfg(), RewardConfig("prosocial", (0.0,) * 3), train_slots=(0, 1, 2))
    return t.init(jax.random.PRNGKey(0)).train_state.params


def learner(envs, frozen, aux):
    return SocialTrainer(envs, cfg(), SocialConfig(aux_coef=aux, seq_len=5),
                         RewardConfig("prosocial", (0.0,) * 3), train_slots=(1,),
                         frozen={0: frozen, 2: frozen})


def test_network_shapes_and_reset():
    net = SocialNet(8, (11, 11, 16), hidden=16)
    obs = jnp.ones((3, 11, 11, 16))
    p = net.init(jax.random.PRNGKey(0), zero_carry(3, 16), obs, jnp.zeros(3, bool), jnp.zeros(3, jnp.int32))
    carry, pi, v, pred = net.apply(p, zero_carry(3, 16), obs, jnp.zeros(3, bool), jnp.zeros(3, jnp.int32))
    assert pred.shape == obs.shape and v.shape == (3,) and pi.logits.shape == (3, 8)
    assert float(pred.min()) >= 0 and float(pred.max()) <= 1
    # A reset step ignores the incoming state.
    junk = (jnp.full((3, 16), 5.0), jnp.full((3, 16), -5.0))
    a = net.apply(p, junk, obs, jnp.ones(3, bool), method=SocialNet.step)[0]
    b = net.apply(p, zero_carry(3, 16), obs, jnp.zeros(3, bool), method=SocialNet.step)[0]
    assert all(np.allclose(x, y) for x, y in zip(a, b))


def test_sequences_are_consecutive_steps_of_one_actor(envs, frozen):
    tr = learner(envs, frozen, 3.0)
    t_len, actors = 20, 4
    x = jnp.arange(t_len)[:, None] * 100 + jnp.arange(actors)[None]      # value = 100 t + actor
    seq = np.asarray(tr._seq(x))                                          # (T/L * A, L)
    assert seq.shape == (t_len // 5 * actors, 5)
    for s in seq:
        assert len(set(s % 100)) == 1                                     # one actor
        assert np.all(np.diff(s // 100) == 1)                             # consecutive steps


def test_learner_trains_with_frozen_teacher(envs, frozen):
    tr = learner(envs, frozen, 3.0)
    assert tr.num_actors == tr.num_envs
    r0 = tr.init(jax.random.PRNGKey(1))
    r1, m = tr.run(r0, 2)
    assert int(r1.update) == 2
    assert all(np.isfinite(np.asarray(v)).all() for v in m.values())
    assert float(m["loss_aux"][-1]) > 0


@pytest.mark.parametrize("aux, changes", [(0.0, False), (3.0, True)])
def test_aux_coef_controls_decoder_training(envs, frozen, aux, changes):
    tr = learner(envs, frozen, aux)
    r0 = tr.init(jax.random.PRNGKey(2))
    r1, _ = tr.run(r0, 1)
    p0, p1 = r0.train_state.params["params"], r1.train_state.params["params"]
    moved = not np.allclose(p0["aux_fc"]["kernel"], p1["aux_fc"]["kernel"])
    assert moved == changes
    assert not np.allclose(p0["fc"]["kernel"], p1["fc"]["kernel"])        # the rest always trains
