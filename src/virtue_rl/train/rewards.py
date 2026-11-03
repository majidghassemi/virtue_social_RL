"""Training rewards with a disposition parameter.

Both functions take the environment's per-agent rewards for one step, shape
(..., N) (apples collected), and per-agent parameters of shape (N,).

prosocial (main):  u_i = r_i + lam_i * mean_{j != i} r_j
    lam_i is the weight agent i puts on the other agents' apples relative to
    its own. lam = 0 is the selfish baseline. The agent's own apples always
    keep weight 1, so competence is rewarded the same way for every lam.

svo (McKee et al. 2020, "Social diversity and social preferences in
mixed-motive reinforcement learning", Eq. 2 and 3):
    theta_i = atan2(mean_{j != i} r_j, r_i)
    u_i = r_i - w_i * |theta_target_i - theta_i|
    Implemented as in the paper. SocialJax's own `get_svo_rewards` multiplies
    the penalty by the number of agents, which the paper does not.
    atan2(0, 0) = 0, so on steps where nobody collects an apple the penalty is
    w * theta_target.
"""

from __future__ import annotations

import jax.numpy as jnp


def mean_others(r: jnp.ndarray) -> jnp.ndarray:
    n = r.shape[-1]
    return (r.sum(-1, keepdims=True) - r) / (n - 1)


def prosocial(r: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    return r + lam * mean_others(r)


def svo(r: jnp.ndarray, w: jnp.ndarray, theta_target_deg: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.arctan2(mean_others(r), r)
    return r - w * jnp.abs(jnp.deg2rad(theta_target_deg) - theta)


def shape(r: jnp.ndarray, kind: str, lam: jnp.ndarray, svo_w: jnp.ndarray | None = None
          ) -> jnp.ndarray:
    """kind "prosocial" uses lam. kind "svo" uses lam as the target angle in degrees."""
    if kind == "prosocial":
        return prosocial(r, lam)
    if kind == "svo":
        return svo(r, svo_w, lam)
    raise ValueError(kind)
