"""Tests for behavioural and validity metrics."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from virtue_rl.envs import RoleConfig
from virtue_rl.metrics import validity as V
from virtue_rl.metrics.behaviour import COUNT_FIELDS, episode_counts, metrics_from_counts
from virtue_rl.runner import MapSuite, random_policy, rollout

SUITE = Path(__file__).parents[1] / "maps" / "suite_v1.json"


def test_episode_counts_match_manual_sums():
    suite = MapSuite(str(SUITE), RoleConfig(3, 3), num_inner_steps=150)
    env = suite.env(suite.entries[0].name, True)
    e = rollout(env, random_policy, jax.random.PRNGKey(0), num_envs=4, num_steps=150)["ethics"]
    c = jax.device_get(episode_counts(e))
    v = np.asarray(e.valid)
    tiv = np.asarray(e.teacher_in_view)
    assert np.array_equal(c["steps"], v.sum(0))
    assert np.array_equal(c["harm_zap"], np.asarray(e.harm_zap).sum(0))
    assert np.array_equal(c["opp_commons_tiv"], (np.asarray(e.opp_commons) & tiv).sum(0))
    assert int(c["steps"][0, 0]) == 149  # last step of the episode is invalid


def _rows(**kw):
    base = {k: 0 for k in COUNT_FIELDS}
    base.update(kw)
    return base


def test_metrics_are_pooled_ratios():
    df = pd.DataFrame([_rows(harm_commons=1, opp_commons=2, apples=4),
                       _rows(harm_commons=0, opp_commons=0, apples=0)])
    m = metrics_from_counts(df)
    assert m["commons_harm_per_opp"] == pytest.approx(0.5)      # 1 / 2, empty episode ignored
    assert m["commons_harm_per_episode"] == pytest.approx(0.5)  # 1 / 2 episodes
    assert m["commons_harm_share"] == pytest.approx(0.25)


def synthetic(ladder=lambda w: 1 - w, maps=4, units=6, eps_levels=(0.0,), conditions=("absent",),
              gap_disp=None, rng=None):
    """Synthetic episode counts with a known harm probability per rung."""
    rng = rng or np.random.default_rng(0)
    rows = []
    jobs = [("mixture", w) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    if gap_disp:
        jobs.append((gap_disp, 0.0))
    for m in range(maps):
        for cond in conditions:
            for disp, w in jobs:
                for eps in eps_levels:
                    for u in range(units):
                        for _ in range(10):
                            opp = rng.integers(5, 15)
                            p = ladder(w)
                            if disp == gap_disp:
                                p = 1.0 if cond == "absent" else 0.1
                            harm = rng.binomial(opp, p)
                            apples = int(rng.integers(20, 30) * (1 - eps))
                            rows.append({"map": f"m{m}", "split": "seen", "condition": cond,
                                         "disposition": disp, "w": w, "eps": eps, "unit": u,
                                         **_rows(harm_commons=harm, opp_commons=opp,
                                                 apples=max(apples, harm), steps=999)})
    return pd.DataFrame(rows)


def test_rank_validity_perfect_and_reversed():
    good = V.rank_validity(synthetic(), "commons_harm_per_opp")
    assert good.tau_b.iloc[0] == pytest.approx(1.0)
    bad = V.rank_validity(synthetic(ladder=lambda w: w), "commons_harm_per_opp")
    assert bad.tau_b.iloc[0] == pytest.approx(-1.0)


def test_competence_independence_small_for_rate_metric():
    df = synthetic(eps_levels=(0.0, 0.1, 0.2))
    ci = V.competence_independence(df, "commons_harm_per_opp")
    assert (ci.relative_change < 0.1).all()
    assert (ci.competence_change < 0).all()   # noise lowered apples


def test_teacher_gap_detects_and_nulls():
    df = synthetic(conditions=("absent", "present"), gap_disp="performative")
    tg = V.teacher_gap(df, n_boot=200)
    perf = tg[tg.disposition == "performative"].iloc[0]
    assert perf.excludes_zero and perf.gap > 0.5
    for w in (0.0, 0.5):
        row = tg[(tg.disposition == "mixture") & (tg.w == w)].iloc[0]
        assert not row.excludes_zero


def test_reliability_shrinks_with_sample_size():
    rl = V.reliability(synthetic(), sizes=(8, 64))
    assert rl.mean_se.iloc[1] < rl.mean_se.iloc[0]


def test_empirical_virtue_cost():
    df = synthetic()
    df.loc[df.w == 1.0, "apples"] = df.loc[df.w == 1.0, "apples"] // 2
    cost = V.empirical_virtue_cost(df)
    assert (cost.empirical_virtue_cost > 0.3).all()


def test_opportunity_onsets_count_each_temptation_once():
    from virtue_rl.envs import EthicsInfo
    T = 6
    opp = jnp.array([1, 1, 1, 0, 1, 1], bool).reshape(T, 1, 1)   # two separate temptations
    tiv = jnp.array([1, 1, 1, 1, 0, 0], bool).reshape(T, 1, 1)
    z = jnp.zeros((T, 1, 1), bool)
    e = EthicsInfo(harm_zap=z, opp_zap=z, harm_commons=z, opp_commons=opp,
                   apples=jnp.zeros((T, 1, 1), jnp.int32), teacher_in_view=tiv,
                   seen_by_teacher=z, valid=jnp.ones((T, 1, 1), bool))
    c = episode_counts(e)
    assert int(c["opp_commons"][0, 0]) == 5
    assert int(c["opp_commons_onset"][0, 0]) == 2
    assert int(c["opp_commons_onset_tiv"][0, 0]) == 1
