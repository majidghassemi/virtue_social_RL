"""Social learner: recurrent PPO with the auxiliary predictive loss of Ndousse et al. (2021).

Ndousse, Eck, Levine and Jaques, "Emergent Social Learning via Multi-agent
Reinforcement Learning", ICML 2021 (arXiv 2010.00581), Section 3.2 and the
appendix. What is taken from the paper:

* PPO with GAE on the agent's own reward only. The learner never receives
  another agent's state, action or reward. Other agents are visible only in
  its own observation (here also through the teacher channel).
* Network: three conv layers, a fully connected layer to 192 units, an LSTM
  with 192 units, and two-layer policy and value heads.
* Auxiliary branch: from the recurrent state and the action taken, a fully
  connected layer followed by transposed convolutions predicts the next
  observation, trained with mean absolute error. Total loss
  L_pi + c_V L_V + c_aux L_aux - c_ent L_ent with c_aux = 3.
* Truncated backpropagation through time on short trajectory segments (the
  paper uses length 16. Here 20, the closest divisor of the 1,000-step
  episode).

Choices the paper leaves open or that differ here, all stated so they can be
reported:

* The auxiliary input is the LSTM output concatenated with a one-hot action.
  The paper says the branch is appended to the policy's encoding of the
  current state and lists the input size as 192 plus the action.
* The decoder ends in a sigmoid because observations are binary channels.
* The auxiliary loss is masked on the last step of an episode, whose next
  observation belongs to the next episode.
* Conv layer sizes are those of SocialJax's network (32 filters, 5x5, 3x3,
  3x3), and c_V, c_ent and the other PPO settings are SocialJax's Harvest
  defaults, not the paper's (c_V 0.1, c_ent 1e-5, tuned for its own
  environment), so that learners and teachers share one PPO setup.
* aux_coef = 0 gives the plain recurrent PPO learner with the same network,
  the paper's "vanilla PPO" comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from virtue_rl.train import rewards as R
from virtue_rl.train.ppo import Trainer


@dataclass(frozen=True)
class SocialConfig:
    aux_coef: float = 3.0
    hidden: int = 192
    seq_len: int = 20


def _dense(n, scale=np.sqrt(2)):
    return nn.Dense(n, kernel_init=orthogonal(scale), bias_init=constant(0.0))


class SocialNet(nn.Module):
    action_dim: int
    obs_shape: tuple[int, int, int]
    hidden: int = 192
    activation: str = "relu"

    def setup(self):
        conv = lambda k: nn.Conv(32, (k, k), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.convs = [conv(5), conv(3), conv(3)]
        self.fc = _dense(self.hidden)
        self.lstm = nn.OptimizedLSTMCell(self.hidden)
        self.pi1, self.pi2 = _dense(64), _dense(self.action_dim, 0.01)
        self.v1, self.v2 = _dense(64), _dense(1, 1.0)
        h, w, c = self.obs_shape
        self.aux_fc = _dense(h * w * 32)
        deconv = lambda f, k: nn.ConvTranspose(f, (k, k), kernel_init=orthogonal(np.sqrt(2)),
                                               bias_init=constant(0.0))
        self.deconvs = [deconv(32, 3), deconv(32, 3), deconv(c, 5)]

    def _a(self, x):
        return nn.relu(x) if self.activation == "relu" else nn.tanh(x)

    def encode(self, obs):
        x = obs
        for cv in self.convs:
            x = self._a(cv(x))
        return self._a(self.fc(x.reshape((x.shape[0], -1))))

    def core(self, carry, emb, reset):
        carry = jax.tree.map(lambda c: jnp.where(reset[:, None], 0.0, c), carry)
        return self.lstm(carry, emb)

    def heads(self, h):
        pi = distrax.Categorical(logits=self.pi2(self._a(self.pi1(h))))
        return pi, jnp.squeeze(self.v2(self._a(self.v1(h))), -1)

    def predict(self, h, action):
        x = jnp.concatenate([h, jax.nn.one_hot(action, self.action_dim)], -1)
        x = self._a(self.aux_fc(x)).reshape((x.shape[0], *self.obs_shape[:2], 32))
        for i, dc in enumerate(self.deconvs):
            x = dc(x)
            x = self._a(x) if i < len(self.deconvs) - 1 else jax.nn.sigmoid(x)
        return x

    def step(self, carry, obs, reset):
        carry, h = self.core(carry, self.encode(obs), reset)
        pi, v = self.heads(h)
        return carry, h, pi, v

    def __call__(self, carry, obs, reset, action):
        carry, h, pi, v = self.step(carry, obs, reset)
        return carry, pi, v, self.predict(h, action)


def zero_carry(batch: int, hidden: int):
    return (jnp.zeros((batch, hidden)), jnp.zeros((batch, hidden)))


class SocialTransition(NamedTuple):
    done: jnp.ndarray
    reset: jnp.ndarray            # episode starts at this step (hidden state is zeroed)
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray              # uint8
    carry_c: jnp.ndarray          # LSTM state entering this step
    carry_h: jnp.ndarray


class SocialRunnerState(NamedTuple):
    train_state: TrainState
    env_states: tuple
    obs: tuple
    carry: tuple
    last_done: jnp.ndarray
    rng: jnp.ndarray
    update: jnp.ndarray


class SocialTrainer(Trainer):
    """Recurrent PPO with the auxiliary loss for the trainable slots, frozen networks elsewhere."""

    def __init__(self, envs, cfg, social: SocialConfig, reward, train_slots, frozen=None):
        super().__init__(envs, cfg, reward, train_slots, frozen)
        self.social = social
        L = social.seq_len
        if cfg.num_steps % L:
            raise ValueError(f"num_steps {cfg.num_steps} not divisible by seq_len {L}")
        self.num_seqs = self.num_actors * cfg.num_steps // L
        if self.num_seqs % cfg.num_minibatches:
            raise ValueError(f"{self.num_seqs} sequences not divisible by {cfg.num_minibatches} minibatches")
        self.network = SocialNet(envs[0].action_space().n, self.obs_shape, social.hidden, cfg.activation)

    def init(self, rng):
        c = self.cfg
        rng, k_net, k_env = jax.random.split(rng, 3)
        obs, states = [], []
        for i, env in enumerate(self.envs):
            o, s = jax.vmap(env.reset)(jax.random.split(jax.random.fold_in(k_env, i), self.e))
            obs.append(o.astype(jnp.uint8))
            states.append(s)
        params = self.network.init(k_net, zero_carry(1, self.social.hidden),
                                   jnp.zeros((1, *self.obs_shape)), jnp.zeros(1, bool),
                                   jnp.zeros(1, jnp.int32))
        tx = optax.chain(optax.clip_by_global_norm(c.max_grad_norm),
                         optax.adam(learning_rate=self._lr if c.anneal_lr else c.lr, eps=1e-5))
        ts = TrainState.create(apply_fn=self.network.apply, params=params, tx=tx)
        return SocialRunnerState(ts, tuple(states), tuple(obs), zero_carry(self.num_actors, self.social.hidden),
                                 jnp.ones(self.num_actors, bool), rng, jnp.int32(0))

    # ------------------------------------------------------------ rollout
    def _env_step(self, carry, _):
        ts, states, obs, lstm, last_done, rng = carry
        rng, k_act, k_frozen, k_step = jax.random.split(rng, 4)
        obs_all = jnp.concatenate(obs, axis=0)
        b = obs_all.shape[0]
        x = obs_all[:, self.slots].reshape(-1, *self.obs_shape)
        new_lstm, _, pi, value = self.network.apply(ts.params, lstm, x.astype(jnp.float32), last_done,
                                                    method=SocialNet.step)
        a_train = pi.sample(seed=k_act)
        logp = pi.log_prob(a_train)
        actions = jnp.zeros((b, self.n), jnp.int32).at[:, self.slots].set(
            a_train.reshape(b, -1).astype(jnp.int32))
        actions = self._frozen_actions(obs_all, actions, k_frozen)
        new_obs, new_states, r_env, done, eth = self._step_groups(states, actions, k_step)
        shaped = R.shape(r_env, self.reward.kind, self.lam, self.svo_w)
        done_a = jnp.repeat(done, len(self.train_slots))
        tr = SocialTransition(done=done_a.astype(jnp.float32), reset=last_done, action=a_train,
                              value=value, reward=shaped[:, self.slots].reshape(-1), log_prob=logp,
                              obs=x, carry_c=lstm[0], carry_h=lstm[1])
        return ((ts, new_states, new_obs, new_lstm, done_a, rng),
                (tr, self._log(r_env, shaped, actions, eth)))

    # ------------------------------------------------------------ update
    def _seq(self, x):
        """(T, A, ...) -> (T/L * A, L, ...): consecutive segments of length L per actor."""
        L = self.social.seq_len
        T, A = x.shape[:2]
        x = x.reshape((T // L, L, A) + x.shape[2:])
        return jnp.moveaxis(x, 2, 1).reshape((T // L * A, L) + x.shape[3:])

    def _loss(self, params, mb):
        c, net = self.cfg, self.network
        b, gae, targets, next_obs, aux_mask = mb                       # leading axes (M, L)
        m, L = b.action.shape
        flat = lambda x: x.reshape((m * L,) + x.shape[2:])
        emb = net.apply(params, flat(b.obs).astype(jnp.float32), method=SocialNet.encode)
        emb = emb.reshape(m, L, -1)

        def core_step(carry, inp):
            e, r = inp
            carry, h = net.apply(params, carry, e, r, method=SocialNet.core)
            return carry, h

        carry0 = (b.carry_c[:, 0], b.carry_h[:, 0])
        _, h = jax.lax.scan(core_step, carry0, (jnp.swapaxes(emb, 0, 1), b.reset.T))
        h = jnp.swapaxes(h, 0, 1).reshape(m * L, -1)
        pi, value = net.apply(params, h, method=SocialNet.heads)

        action, old_v, old_lp = flat(b.action), flat(b.value), flat(b.log_prob)
        gae, targets = flat(gae), flat(targets)
        log_prob = pi.log_prob(action)
        v_clip = old_v + (value - old_v).clip(-c.clip_eps, c.clip_eps)
        v_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(v_clip - targets)).mean()
        ratio = jnp.exp(log_prob - old_lp)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        a_loss = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * gae).mean()
        ent = pi.entropy().mean()

        pred = net.apply(params, h, action, method=SocialNet.predict)
        err = jnp.abs(pred - flat(next_obs).astype(jnp.float32)).mean(axis=(1, 2, 3))
        mask = flat(aux_mask)
        aux = (err * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        total = a_loss + c.vf_coef * v_loss - c.ent_coef * ent + self.social.aux_coef * aux
        return total, (v_loss, a_loss, ent, aux)

    def _update(self, runner, _):
        c = self.cfg
        ts, states, obs, lstm, last_done, rng, upd = runner
        (ts, states, obs, lstm, last_done, rng), (traj, log) = jax.lax.scan(
            self._env_step, (ts, states, obs, lstm, last_done, rng), None, c.num_steps)

        last_x = jnp.concatenate(obs, 0)[:, self.slots].reshape(-1, *self.obs_shape)
        _, _, _, last_val = self.network.apply(ts.params, lstm, last_x.astype(jnp.float32), last_done,
                                               method=SocialNet.step)

        def gae_step(carry, t):
            gae, next_value = carry
            delta = t.reward + c.gamma * next_value * (1 - t.done) - t.value
            gae = delta + c.gamma * c.gae_lambda * (1 - t.done) * gae
            return (gae, t.value), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val), traj,
                              reverse=True, unroll=16)
        targets = adv + traj.value
        next_obs = jnp.concatenate([traj.obs[1:], last_x[None]], 0)
        aux_mask = 1.0 - traj.done

        data = jax.tree.map(self._seq, (traj, adv, targets, next_obs, aux_mask))

        def epoch(carry, _):
            ts, rng = carry
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, self.num_seqs)
            mbs = jax.tree.map(lambda x: jnp.take(x, perm, axis=0).reshape(
                (c.num_minibatches, -1) + x.shape[1:]), data)

            def mb_step(ts, mb):
                (loss, aux), grads = jax.value_and_grad(self._loss, has_aux=True)(ts.params, mb)
                return ts.apply_gradients(grads=grads), (loss, *aux)

            ts, losses = jax.lax.scan(mb_step, ts, mbs, unroll=self.unroll_mb)
            return (ts, rng), losses

        (ts, rng), losses = jax.lax.scan(epoch, (ts, rng), None, c.update_epochs,
                                          unroll=self.unroll_ep)
        metrics = self._episode_metrics(log)
        metrics.update({"loss_total": losses[0].mean(), "loss_value": losses[1].mean(),
                        "loss_actor": losses[2].mean(), "entropy": losses[3].mean(),
                        "loss_aux": losses[4].mean()})
        return SocialRunnerState(ts, states, obs, lstm, last_done, rng, upd + 1), metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def _run_scan(self, runner, num_updates: int):
        return jax.lax.scan(self._update, runner, None, num_updates)

    @partial(jax.jit, static_argnums=(0,))
    def _update_once(self, runner):
        return self._update(runner, None)
