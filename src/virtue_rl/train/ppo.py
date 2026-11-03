"""Independent PPO on several Commons Harvest maps with disposition rewards.

Follows SocialJax's IPPO baseline (algorithms/IPPO/ippo_cnn_harvest_open.py,
itself based on PureJaxRL): same network, same PPO loss, same GAE, same
learning-rate schedule. Differences, all needed for this project:

* Several maps in one run. SocialJax compiles a map's spawn cells into the
  environment, and maps differ in the number of apple cells, so each map is a
  separate group of `envs_per_map` parallel environments. One policy forward
  pass serves all groups.
* Ethics wrapper (`EthicsHarvest`): teacher channel in the observation, and
  per-step ethics records that are logged during training.
* Disposition rewards (`virtue_rl.train.rewards`) with a per-agent parameter.
* Slots. `train_slots` are the agents controlled by the trainable network
  (parameter sharing among them, as in SocialJax's default). Every other slot
  is played by a frozen network passed in `frozen`. Only transitions of
  trainable slots enter the PPO loss.
* Training runs in chunks of updates, so it can be checkpointed and resumed.

Observations are binary, so the rollout buffer stores them as uint8.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from virtue_rl.envs.wrapper import EthicsHarvest
from virtue_rl.train import rewards as R
from virtue_rl.train.networks import ActorCritic

ZAP = 7  # Actions.zap_forward in SocialJax Harvest Open


@dataclass(frozen=True)
class PPOConfig:
    """Defaults are SocialJax's IPPO settings for Harvest Open (ippo_base.yaml)."""
    lr: float = 5e-4
    envs_per_map: int = 24
    num_steps: int = 1000
    total_timesteps: float = 1e8          # environment steps over all parallel envs
    update_epochs: int = 2
    num_minibatches: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    activation: str = "relu"


@dataclass(frozen=True)
class RewardConfig:
    kind: str                    # "prosocial" or "svo"
    lam: tuple[float, ...]       # per agent: weight on others (prosocial) or target angle (svo)
    svo_w: tuple[float, ...] | None = None


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray             # uint8


class RunnerState(NamedTuple):
    train_state: TrainState
    env_states: tuple
    obs: tuple
    rng: jnp.ndarray
    update: jnp.ndarray


class Trainer:
    def __init__(self, envs: Sequence[EthicsHarvest], cfg: PPOConfig, reward: RewardConfig,
                 train_slots: Sequence[int], frozen: dict | None = None):
        self.envs = tuple(envs)
        self.cfg = cfg
        self.reward = reward
        n = envs[0].num_agents
        if any(e.num_agents != n for e in envs):
            raise ValueError("all maps need the same number of agents")
        if any(e.num_inner_steps != cfg.num_steps for e in envs):
            raise ValueError("num_steps must equal the episode length, so that each rollout is "
                             "exactly one episode per environment (as in SocialJax's IPPO)")
        self.n = n
        self.train_slots = tuple(int(s) for s in train_slots)
        self.frozen = dict(frozen or {})
        others = sorted(set(range(n)) - set(self.train_slots))
        if sorted(self.frozen) != others:
            raise ValueError(f"slots {others} need frozen params, got {sorted(self.frozen)}")
        if len(reward.lam) != n:
            raise ValueError("reward.lam needs one value per agent")
        self.lam = jnp.asarray(reward.lam, jnp.float32)
        self.svo_w = None if reward.svo_w is None else jnp.asarray(reward.svo_w, jnp.float32)

        self.k = len(self.envs)
        self.e = cfg.envs_per_map
        self.num_envs = self.k * self.e
        self.num_actors = self.num_envs * len(self.train_slots)
        self.num_updates = int(cfg.total_timesteps // cfg.num_steps // self.num_envs)
        self.batch_size = self.num_actors * cfg.num_steps
        if self.batch_size % cfg.num_minibatches:
            raise ValueError(f"batch {self.batch_size} not divisible by {cfg.num_minibatches} minibatches")
        obs, _ = jax.eval_shape(envs[0].reset, jax.random.PRNGKey(0))
        self.obs_shape = tuple(obs.shape[1:])
        self.slots = np.asarray(self.train_slots)        # array index (a tuple would index axes)
        self.network = ActorCritic(envs[0].action_space().n, activation=cfg.activation)

    # ------------------------------------------------------------ setup
    def _lr(self, count):
        c = self.cfg
        frac = 1.0 - (count // (c.num_minibatches * c.update_epochs)) / self.num_updates
        return c.lr * frac

    def init(self, rng: jax.Array) -> RunnerState:
        c = self.cfg
        rng, k_net, k_env = jax.random.split(rng, 3)
        obs, states = [], []
        for i, env in enumerate(self.envs):
            o, s = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(k_env, i), self.e))
            obs.append(o.astype(jnp.uint8))
            states.append(s)
        params = self.network.init(k_net, jnp.zeros((1, *self.obs_shape)))
        tx = optax.chain(optax.clip_by_global_norm(c.max_grad_norm),
                         optax.adam(learning_rate=self._lr if c.anneal_lr else c.lr, eps=1e-5))
        ts = TrainState.create(apply_fn=self.network.apply, params=params, tx=tx)
        return RunnerState(ts, tuple(states), tuple(obs), rng, jnp.int32(0))

    # ------------------------------------------------------------ rollout
    def _act(self, params, obs_all, key):
        """obs_all (B, N, ...) uint8 -> actions (B, N), plus logp and value of trainable slots."""
        b = obs_all.shape[0]
        k_train, k_frozen = jax.random.split(key)
        x = obs_all[:, self.slots].reshape(-1, *self.obs_shape).astype(jnp.float32)
        pi, value = self.network.apply(params, x)
        a_train = pi.sample(seed=k_train)
        logp = pi.log_prob(a_train)
        actions = jnp.zeros((b, self.n), jnp.int32)
        actions = actions.at[:, self.slots].set(a_train.reshape(b, -1).astype(jnp.int32))
        for j, (slot, fp) in enumerate(sorted(self.frozen.items())):
            pj, _ = self.network.apply(fp, obs_all[:, slot].astype(jnp.float32))
            actions = actions.at[:, slot].set(
                pj.sample(seed=jax.random.fold_in(k_frozen, j)).astype(jnp.int32))
        return actions, a_train, logp, value

    def _env_step(self, carry, _):
        ts, states, obs, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        obs_all = jnp.concatenate(obs, axis=0)
        actions, a_train, logp, value = self._act(ts.params, obs_all, k_act)
        new_obs, new_states, rews, dones, eths = [], [], [], [], []
        for i, env in enumerate(self.envs):
            a = actions[i * self.e:(i + 1) * self.e]
            o, s, r, d, _, eth = jax.vmap(env.step)(
                jax.random.split(jax.random.fold_in(k_step, i), self.e), states[i], a)
            new_obs.append(o.astype(jnp.uint8))
            new_states.append(s)
            rews.append(r)
            dones.append(d["__all__"])
            eths.append(eth)
        r_env = jnp.concatenate(rews, 0)                               # (B, N)
        done = jnp.concatenate(dones, 0)                               # (B,)
        eth = jax.tree.map(lambda *x: jnp.concatenate(x, 0), *eths)
        shaped = R.shape(r_env, self.reward.kind, self.lam, self.svo_w)
        s = len(self.train_slots)
        tr = Transition(
            done=jnp.repeat(done, s).astype(jnp.float32),
            action=a_train,
            value=value,
            reward=shaped[:, self.slots].reshape(-1),
            log_prob=logp,
            obs=obs_all[:, self.slots].reshape(-1, *self.obs_shape),
        )
        log = {"r_env": r_env, "shaped": shaped, "zap": (actions == ZAP),
               "harm_commons": eth.harm_commons, "opp_commons": eth.opp_commons,
               "harm_zap": eth.harm_zap, "regrown": eth.regrown[:, 0], "stock": eth.stock[:, 0]}
        return (ts, tuple(new_states), tuple(new_obs), rng), (tr, log)

    # ------------------------------------------------------------ update
    def _loss(self, params, batch, gae, targets):
        c = self.cfg
        pi, value = self.network.apply(params, batch.obs.astype(jnp.float32))
        log_prob = pi.log_prob(batch.action)
        v_clip = batch.value + (value - batch.value).clip(-c.clip_eps, c.clip_eps)
        v_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(v_clip - targets)).mean()
        ratio = jnp.exp(log_prob - batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        a_loss = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * gae).mean()
        ent = pi.entropy().mean()
        total = a_loss + c.vf_coef * v_loss - c.ent_coef * ent
        return total, (v_loss, a_loss, ent)

    def _update(self, runner: RunnerState, _):
        c = self.cfg
        ts, states, obs, rng, upd = runner
        (ts, states, obs, rng), (traj, log) = jax.lax.scan(
            self._env_step, (ts, states, obs, rng), None, c.num_steps)

        last_x = jnp.concatenate(obs, 0)[:, self.slots].reshape(-1, *self.obs_shape)
        _, last_val = self.network.apply(ts.params, last_x.astype(jnp.float32))

        def gae_step(carry, t):
            gae, next_value = carry
            delta = t.reward + c.gamma * next_value * (1 - t.done) - t.value
            gae = delta + c.gamma * c.gae_lambda * (1 - t.done) * gae
            return (gae, t.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val), traj,
                              reverse=True, unroll=16)
        targets = adv + traj.value

        def epoch(carry, _):
            ts, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, self.batch_size)
            flat = jax.tree.map(lambda x: x.reshape((self.batch_size,) + x.shape[2:]),
                                (traj, adv, targets))
            mbs = jax.tree.map(lambda x: jnp.take(x, perm, axis=0).reshape(
                (c.num_minibatches, -1) + x.shape[1:]), flat)

            def mb_step(ts, mb):
                b, g, tg = mb
                (loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(ts.params, b, g, tg)
                return ts.apply_gradients(grads=grads), (loss, *aux)

            ts, losses = jax.lax.scan(mb_step, ts, mbs)
            return (ts, rng), losses

        (ts, rng), losses = jax.lax.scan(epoch, (ts, rng), None, c.update_epochs)

        # One rollout is one episode per environment, so sums over time are per episode.
        ep = lambda x: x.astype(jnp.float32).sum(0).mean(0)            # (T, B, N) -> (N,)
        metrics = {
            "apples": ep(log["r_env"]),
            "shaped_return": ep(log["shaped"]),
            "harm_commons": ep(log["harm_commons"]),
            "opp_commons": ep(log["opp_commons"]),
            "harm_zap": ep(log["harm_zap"]),
            "zaps": ep(log["zap"]),
            "regrown": log["regrown"].astype(jnp.float32).sum(0).mean(),
            "stock": log["stock"].astype(jnp.float32).mean(),
            "loss_total": losses[0].mean(), "loss_value": losses[1].mean(),
            "loss_actor": losses[2].mean(), "entropy": losses[3].mean(),
        }
        return RunnerState(ts, states, obs, rng, upd + 1), metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def run(self, runner: RunnerState, num_updates: int):
        """Run `num_updates` PPO updates. Returns the new state and stacked metrics."""
        return jax.lax.scan(self._update, runner, None, num_updates)


def metrics_to_rows(metrics: dict, start_update: int, env_steps_per_update: int) -> list[dict]:
    """Stacked metrics from `Trainer.run` -> one JSON-ready dict per update."""
    m = jax.device_get(metrics)
    u = len(m["entropy"])
    rows = []
    for i in range(u):
        row = {"update": start_update + i + 1, "env_steps": (start_update + i + 1) * env_steps_per_update}
        for k, v in m.items():
            v = np.asarray(v[i])
            row[k] = v.tolist() if v.ndim else float(v)
        rows.append(row)
    return rows
