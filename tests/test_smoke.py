"""Phase 0 checks that the pinned SocialJax environment imports, resets and steps."""

import jax
import jax.numpy as jnp
import pytest

import socialjax

NUM_AGENTS = 7


@pytest.fixture(scope="module")
def env():
    return socialjax.make("harvest_common_open", num_agents=NUM_AGENTS, shared_rewards=False)


def test_reset_shapes(env):
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape[0] == NUM_AGENTS
    assert obs.shape[1:3] == (11, 11)  # 11x11 egocentric view
    assert env.action_space(0).n == 8  # 7 = zap_forward
    assert state.agent_locs.shape[0] == NUM_AGENTS


def test_jit_vmap_step(env):
    num_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
    obs, state = jax.vmap(env.reset)(keys)
    actions = jnp.full((num_envs, NUM_AGENTS), 6)  # everyone stays
    step = jax.jit(jax.vmap(env.step))
    obs, state, reward, done, info = step(keys, state, actions)
    assert obs.shape[:2] == (num_envs, NUM_AGENTS)
    assert jnp.all(jnp.isfinite(reward))
    assert "__all__" in done
