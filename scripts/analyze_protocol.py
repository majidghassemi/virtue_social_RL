"""Analyse Phase 3 episode counts and check the pre-registered criteria.

Usage:
    python scripts/analyze_protocol.py                                  # config paths
    python scripts/analyze_protocol.py --runs runs/protocol_v1_pilot \\
        --summary results/protocol_v1_pilot_summary.json

Every number is computed from the raw episode counts in the run directory.
Writes a JSON summary and a short Markdown report next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from virtue_rl.metrics import validity as V


def load_counts(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("counts_*.csv.gz"))
    if not files:
        raise SystemExit(f"no counts files in {run_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/protocol_v1.yaml"))
    p.add_argument("--runs", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    crit = cfg["criteria"]
    run_dir = args.runs or Path(cfg["out_dir"])
    out = args.summary or Path(cfg["summary"])

    df = load_counts(run_dir)
    learners = df[df.role == "learner"].copy()
    summary: dict = {"run_dir": str(run_dir), "rows": int(len(df)),
                     "maps": sorted(df["map"].unique().tolist())}
    checks = {}

    # 1. Rank validity, every metric, every split.
    rv = pd.concat([V.rank_validity(learners, m, crit["rank_validity"]["condition"])
                    for m in cfg["metrics"]])
    rv_present = pd.concat([V.rank_validity(learners, m, "present") for m in cfg["metrics"]])
    summary["rank_validity"] = rv.to_dict("records")
    summary["rank_validity_present"] = rv_present.to_dict("records")
    for m, g in rv.groupby("metric"):
        checks[f"rank_validity::{m}"] = bool((g.tau_b >= crit["rank_validity"]["min_tau_b"] - 1e-9).all())

    # 2. Competence independence.
    ci = pd.concat([V.competence_independence(learners, m) for m in cfg["metrics"]])
    summary["competence_independence"] = ci.to_dict("records")
    if len(ci):
        for m, g in ci.groupby("metric"):
            checks[f"competence_independence::{m}"] = bool(
                (g.relative_change <= crit["competence_independence"]["max_relative_change"] + 1e-9).all())

    # 3. Teacher gap (condition level) and within-episode gaps.
    tg_cfg = crit["teacher_gap"]
    tg = V.teacher_gap(learners, tg_cfg["metric"], n_boot=int(tg_cfg["n_boot"]))
    summary["teacher_gap"] = tg.to_dict("records")
    summary["in_view_gap"] = V.in_view_gap(learners).to_dict("records")
    detect = tg[tg.disposition.isin(tg_cfg["detect"])]
    nulls = tg[(tg.disposition == "mixture") & tg.w.isin(tg_cfg["no_gap_w"])]
    checks["teacher_gap::detects_performative"] = bool(
        len(detect) and (detect.excludes_zero & (detect.gap > 0)).all())
    checks["teacher_gap::null_for_virtuous_and_indifferent"] = bool(
        len(nulls) and (~nulls.excludes_zero).all())

    # 4. Reliability.
    rl_cfg = crit["reliability"]
    rl = V.reliability(learners, rl_cfg["metric"], rl_cfg["w"])
    summary["reliability"] = rl.to_dict("records")
    ok = rl[rl.mean_se <= rl_cfg["target_se"]]
    summary["episodes_for_target_se"] = int(ok.agent_episodes.min()) if len(ok) else None

    # 5. Zap known groups, empirical cost of virtue.
    summary["zap_validity"] = V.zap_validity(learners).to_dict("records")
    cost = V.empirical_virtue_cost(learners)
    maps_meta = df.groupby("map")[["band", "split"]].first()
    from virtue_rl.envs.maps import load_suite
    structural = {e.name: e.stats["structural_virtue_cost"] for e in load_suite(cfg["suite"])}
    cost["structural_virtue_cost"] = cost["map"].map(structural)
    cost["band_structural"] = cost["map"].map(maps_meta["band"])
    seen_cost = cost[cost.split == "seen"]["empirical_virtue_cost"]
    if len(seen_cost) >= 3:
        cuts = np.quantile(seen_cost, [1 / 3, 2 / 3])
        cost["band_empirical"] = np.where(cost.empirical_virtue_cost <= cuts[0], "low",
                                          np.where(cost.empirical_virtue_cost <= cuts[1],
                                                   "medium", "high"))
        summary["empirical_band_cuts"] = [float(c) for c in cuts]
    summary["virtue_cost"] = cost.to_dict("records")
    if cost[["empirical_virtue_cost", "structural_virtue_cost"]].dropna().shape[0] >= 3:
        summary["virtue_cost_spearman"] = float(
            cost[["empirical_virtue_cost", "structural_virtue_cost"]].corr("spearman").iloc[0, 1])

    # 6. Exploratory metrics (added after the pilot, not part of the checks).
    xm = cfg.get("exploratory_metrics", [])
    have = [m for m in xm if all(f in df.columns for f in ("opp_commons_onset",))]
    if have:
        summary["exploratory"] = {
            "rank_validity": pd.concat([V.rank_validity(learners, m, "absent") for m in have]).to_dict("records"),
            "competence_independence": pd.concat(
                [V.competence_independence(learners, m) for m in have]).to_dict("records"),
            "teacher_gap": pd.concat([V.teacher_gap(learners, m, n_boot=int(tg_cfg["n_boot"]))
                                      for m in have]).to_dict("records"),
            "reliability": pd.concat([V.reliability(learners, m, rl_cfg["w"]) for m in have]).to_dict("records"),
        }

    summary["checks"] = checks
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float) + "\n")

    # Short Markdown report.
    lines = [f"# Protocol summary ({run_dir})", "",
             f"Agent-episode rows: {len(df)}, learner rows: {len(learners)}, maps: {len(summary['maps'])}", "",
             "## Pre-registered checks", "", "| Check | Passed |", "|---|---|"]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in checks.items()]
    lines += ["", "## Rank validity (Kendall tau-b vs ladder, teacher absent)", "",
              "| Metric | Split | tau-b | Median per-map tau | Rung values (w: metric) |", "|---|---|---|---|---|"]
    for r in summary["rank_validity"]:
        rungs = ", ".join(f"{k:g}: {fmt(v)}" for k, v in r["rung_values"].items())
        lines.append(f"| {r['metric']} | {r['split']} | {fmt(r['tau_b'], 2)} | "
                     f"{fmt(r['per_map_tau_median'], 2)} | {rungs} |")
    if summary["competence_independence"]:
        lines += ["", "## Competence independence (seen maps, teacher absent)", "",
                  "| Metric | w | Relative change | Competence change |", "|---|---|---|---|"]
        for r in summary["competence_independence"]:
            lines.append(f"| {r['metric']} | {r['w']:g} | {fmt(r['relative_change'])} | "
                         f"{fmt(r['competence_change'])} |")
    lines += ["", "## Teacher gap (absent minus present, commons harm per opportunity)", "",
              "| Disposition | w | Gap | 95% CI | Excludes zero |", "|---|---|---|---|---|"]
    for r in summary["teacher_gap"]:
        lines.append(f"| {r['disposition']} | {r['w']:g} | {fmt(r['gap'])} | "
                     f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {r['excludes_zero']} |")
    lines += ["", "## Within-episode gaps (teacher present)", "",
              "| Disposition | w | Gap, teacher in view | Gap, seen by teacher | Share in view | Share seen |",
              "|---|---|---|---|---|---|"]
    for r in summary["in_view_gap"]:
        lines.append(f"| {r['disposition']} | {r['w']:g} | {fmt(r['gap_teacher_in_view'])} | "
                     f"{fmt(r['gap_seen_by_teacher'])} | {fmt(r['tiv_share'])} | {fmt(r['sbt_share'])} |")
    lines += ["", "## Reliability", "", "| Agent-episodes | Mean SE |", "|---|---|"]
    lines += [f"| {r['agent_episodes']} | {fmt(r['mean_se'])} |" for r in summary["reliability"]]
    lines += ["", f"Agent-episodes needed for SE <= {rl_cfg['target_se']}: "
                  f"{summary['episodes_for_target_se']}"]
    lines += ["", "## Zap known groups", "", "| Metric | Aggressive mean | Max of others | Separates |",
              "|---|---|---|---|"]
    lines += [f"| {r['metric']} | {fmt(r['aggressive_mean'])} | {fmt(r['others_max'])} | {r['separates']} |"
              for r in summary["zap_validity"]]
    if "exploratory" in summary:
        x = summary["exploratory"]
        lines += ["", "## Exploratory metric (added after the pilot, not pre-registered)", "",
                  "| Metric | Split | tau-b | Rung values |", "|---|---|---|---|"]
        for r in x["rank_validity"]:
            rungs = ", ".join(f"{k:g}: {fmt(v)}" for k, v in r["rung_values"].items())
            lines.append(f"| {r['metric']} | {r['split']} | {fmt(r['tau_b'], 2)} | {rungs} |")
        if x["competence_independence"]:
            lines += ["", "| Metric | w | Relative change under noise |", "|---|---|---|"]
            lines += [f"| {r['metric']} | {r['w']:g} | {fmt(r['relative_change'])} |"
                      for r in x["competence_independence"]]
        lines += ["", "| Metric | Disposition | w | Teacher gap | 95% CI |", "|---|---|---|---|---|"]
        lines += [f"| {r['metric']} | {r['disposition']} | {r['w']:g} | {fmt(r['gap'])} | "
                  f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |" for r in x["teacher_gap"]]
        lines += ["", "| Metric | Agent-episodes | Mean SE |", "|---|---|---|"]
        lines += [f"| {r['metric']} | {r['agent_episodes']} | {fmt(r['mean_se'])} |"
                  for r in x["reliability"]]
    if "virtue_cost_spearman" in summary:
        lines += ["", f"Empirical vs structural cost of virtue, Spearman rho: "
                      f"{fmt(summary['virtue_cost_spearman'], 2)}"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
