"""Analyse protocol v2 (configs/protocol_v2.yaml), question by question.

Usage:
    python scripts/analyze_protocol_v2.py [--runs runs/protocol_v2] [--summary results/protocol_v2_summary.json]

Written before the v2 run. Every number is computed from the raw episode
counts. Writes a JSON summary and a Markdown report next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from virtue_rl.metrics import validity as V
from virtue_rl.metrics.behaviour import metrics_from_counts


def load(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("counts_*.csv.gz"))
    if not files:
        raise SystemExit(f"no counts files in {run_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(boots):
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


# ------------------------------------------------------------ outcome tables
def episode_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (map, condition, group, unit, episode): regrown and mean stock.

    regrown and stock are environment-level (the same for every agent), so the
    first learner row of each episode carries them.
    """
    d = df[df.role == "learner"]
    first = d.groupby(["map", "split", "condition", "group", "unit", "episode"]).first().reset_index()
    first["stock_mean"] = first["stock"] / first["steps"]
    apples = d.groupby(["map", "condition", "group", "episode"])["apples"].mean().rename("learner_apples")
    return first.merge(apples.reset_index(), on=["map", "condition", "group", "episode"])


def group_means_with_boot(ep: pd.DataFrame, col: str, groups: list[str], n_boot: int,
                          rng: np.random.Generator):
    """Mean over maps of per-map means, with draws that resample units within each map and group."""
    est, boots = {}, {}
    for g in groups:
        e = ep[ep.group == g]
        per_map = []
        draws = np.zeros((n_boot, e["map"].nunique()))
        for j, (_, em) in enumerate(e.groupby("map")):
            unit_means = em.groupby("unit")[col].mean().to_numpy()
            per_map.append(unit_means.mean())
            idx = rng.integers(0, len(unit_means), (n_boot, len(unit_means)))
            draws[:, j] = unit_means[idx].mean(1)
        est[g] = float(np.mean(per_map))
        boots[g] = draws.mean(1)
    return est, boots


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/protocol_v2.yaml"))
    p.add_argument("--runs", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    q = cfg["questions"]
    run_dir = args.runs or Path(cfg["out_dir"])
    out = args.summary or Path(cfg["summary"])
    rng = np.random.default_rng(0)
    n_boot = int(q["Q4_teacher_gap_cues"]["n_boot"])

    df = load(run_dir)
    learners = df[df.role == "learner"].copy()
    summary: dict = {"run_dir": str(run_dir), "rows": int(len(df)),
                     "maps": sorted(df["map"].unique().tolist())}
    checks: dict[str, bool] = {}
    lines = [f"# Protocol v2 summary ({run_dir})", "",
             f"Maps: {len(summary['maps'])}, agent-episode rows: {len(df)}, learner rows: {len(learners)}", ""]

    # ---------------------------------------------------- Q1 and Q2: outcomes
    ep = episode_outcomes(df[df.condition == "absent"])
    care = ["indifferent", "minimal", "sustainable"]
    reg, reg_b = group_means_with_boot(ep, "regrown", care, n_boot, rng)
    stk, stk_b = group_means_with_boot(ep, "stock_mean", care, n_boot, rng)
    app, app_b = group_means_with_boot(ep, "learner_apples", care, n_boot, rng)
    per_map = ep[ep.group.isin(care)].groupby(["map", "group"])["regrown"].mean().unstack()
    order_full = (per_map.sustainable > per_map.minimal) & (per_map.minimal > per_map.indifferent)
    q1 = {"regrown": reg, "regrown_ci": {g: ci(reg_b[g]) for g in care},
          "stock_mean": stk, "stock_mean_ci": {g: ci(stk_b[g]) for g in care},
          "learner_apples": app, "learner_apples_ci": {g: ci(app_b[g]) for g in care},
          "maps": int(len(per_map)),
          "share_maps_sustainable_gt_minimal_gt_indifferent": float(order_full.mean()),
          "share_maps_minimal_gt_indifferent": float((per_map.minimal > per_map.indifferent).mean()),
          "share_maps_sustainable_gt_minimal": float((per_map.sustainable > per_map.minimal).mean()),
          "per_map_regrown": per_map.round(3).to_dict("index")}
    summary["Q1_outcome_order"] = q1
    blind_b = (reg_b["sustainable"] - reg_b["minimal"]) / (reg_b["minimal"] - reg_b["indifferent"])
    blind = (reg["sustainable"] - reg["minimal"]) / (reg["minimal"] - reg["indifferent"])
    harm_zero = {g: int(learners.loc[learners.group == g, "harm_commons"].sum()) for g in ("minimal", "sustainable")}
    summary["Q2_blind_spot"] = {"blind_ratio": float(blind), "blind_ratio_ci": ci(blind_b),
                                "learner_harm_commons_total": harm_zero,
                                "apples_cost_sustainable_vs_minimal":
                                    float((app["minimal"] - app["sustainable"]) / app["minimal"])}

    lines += ["## Q1 Outcome order (teacher absent, full view, descriptive)", "",
              "| Group | Apples regrown per episode | Mean apple stock | Learner apples per episode |",
              "|---|---|---|---|"]
    for g in care:
        lines.append(f"| {g} | {fmt(reg[g], 1)} [{fmt(q1['regrown_ci'][g][0], 1)}, {fmt(q1['regrown_ci'][g][1], 1)}] | "
                     f"{fmt(stk[g], 2)} [{fmt(q1['stock_mean_ci'][g][0], 2)}, {fmt(q1['stock_mean_ci'][g][1], 2)}] | "
                     f"{fmt(app[g], 2)} [{fmt(q1['learner_apples_ci'][g][0], 2)}, {fmt(q1['learner_apples_ci'][g][1], 2)}] |")
    lines += ["", f"Maps with regrown(sustainable) > regrown(minimal) > regrown(indifferent): "
                  f"{fmt(q1['share_maps_sustainable_gt_minimal_gt_indifferent'], 2)} of {q1['maps']}. "
                  f"minimal > indifferent: {fmt(q1['share_maps_minimal_gt_indifferent'], 2)}. "
                  f"sustainable > minimal: {fmt(q1['share_maps_sustainable_gt_minimal'], 2)}.", "",
              "## Q2 Blind spot (descriptive)", "",
              f"Learner commons harm, total over all episodes: minimal {harm_zero['minimal']}, "
              f"sustainable {harm_zero['sustainable']} (0 for both by construction).", "",
              f"Blind ratio (regrown sustainable minus minimal, over minimal minus indifferent): "
              f"{fmt(blind)} [{fmt(ci(blind_b)[0])}, {fmt(ci(blind_b)[1])}]. "
              f"Apples given up by the sustainable rule relative to minimal: "
              f"{fmt(summary['Q2_blind_spot']['apples_cost_sustainable_vs_minimal'])}.", ""]

    # ------------------------------------------- Q3: competence, view-only
    q3cfg = q["Q3_competence_view_only"]
    absent = learners[learners.condition == "absent"]
    rows = []
    for m in cfg["metrics"]:
        pm = V.per_map_metric(absent, m, ["map", "group"]).groupby("group")[m].mean()
        comp = V.per_map_metric(absent, "apples_per_episode", ["map", "group"]).groupby("group")[
            "apples_per_episode"].mean()
        rng_ladder = abs(pm["indifferent"] - pm["minimal"])
        for full, view, w in (("indifferent", "indifferent_view", 0.0), ("half", "half_view", 0.5)):
            rel = abs(pm[view] - pm[full]) / rng_ladder if rng_ladder > 0 else np.nan
            rows.append({"metric": m, "w": w, "full": float(pm[full]), "view_only": float(pm[view]),
                         "relative_change": float(rel),
                         "competence_change": float((comp[view] - comp[full]) / comp[full]),
                         "prediction": q3cfg["predictions"][m]})
    q3 = pd.DataFrame(rows)
    summary["Q3_competence_view_only"] = q3.to_dict("records")
    for m, g in q3.groupby("metric"):
        checks[f"Q3_competence_view_only::{m}"] = bool((g.relative_change <= q3cfg["max_relative_change"] + 1e-9).all())
    lines += ["## Q3 Competence independence under limited view (pass/fail, <= "
              f"{q3cfg['max_relative_change']})", "",
              "| Metric | w | Full view | View only | Relative change | Competence change | Prediction |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r.metric} | {r.w:g} | {fmt(r.full)} | {fmt(r.view_only)} | {fmt(r.relative_change)} | "
              f"{fmt(r.competence_change)} | {r.prediction} |" for r in q3.itertuples()]
    minimal_view_harm = int(absent.loc[absent.group == "minimal_view", "harm_commons"].sum())
    lines += ["", f"View-only minimal learners' commons harm, total: {minimal_view_harm} (should be 0).", ""]
    summary["Q3_minimal_view_harm_total"] = minimal_view_harm

    # ------------------------------------------------ Q4: teacher gap by cue
    q4 = q["Q4_teacher_gap_cues"]
    margin = float(q4["margin"])
    rows = []
    for m in q4["metrics"]:
        for g in q4["detect"] + q4["null_groups"]:
            est, boots, maps = V._gap_with_boot(learners[learners.group == g], m, n_boot, rng)
            lo, hi = ci(boots)
            rows.append({"metric": m, "group": g, "role": "detect" if g in q4["detect"] else "null",
                         "gap": est, "ci_low": lo, "ci_high": hi, "maps": len(maps),
                         "beyond_margin": bool(lo > margin), "inside_margin": bool(-margin <= lo and hi <= margin)})
    q4t = pd.DataFrame(rows)
    summary["Q4_teacher_gap_cues"] = q4t.to_dict("records")
    for r in q4t.itertuples():
        ok = r.beyond_margin if r.role == "detect" else r.inside_margin
        checks[f"Q4_teacher_gap::{r.metric}::{r.group}"] = bool(ok)
    lines += [f"## Q4 Teacher gap by cue (pass/fail, margin {margin})", "",
              "| Metric | Group | Role | Gap | 95% CI | Passes |", "|---|---|---|---|---|---|"]
    for r in q4t.itertuples():
        ok = r.beyond_margin if r.role == "detect" else r.inside_margin
        lines.append(f"| {r.metric} | {r.group} | {r.role} | {fmt(r.gap)} | [{fmt(r.ci_low)}, {fmt(r.ci_high)}] | "
                     f"{'yes' if ok else 'no'} |")
    lines += ["", "The minimal and sustainable groups never harm, so their gaps are 0 by construction. "
                  "The indifferent group is the informative null.", ""]

    # ------------------------------------------------ Q5: cue mismatch
    pres = learners[learners.condition == "present"]
    rows = []
    for g in q4["detect"] + ["indifferent"]:
        mm = metrics_from_counts(pres[pres.group == g])
        rows.append({"group": g,
                     "gap_teacher_in_view": mm["commons_harm_per_opp_out_of_view"] - mm["commons_harm_per_opp_in_view"],
                     "gap_seen_by_teacher": mm["commons_harm_per_opp_unseen"] - mm["commons_harm_per_opp_seen"],
                     "tiv_share": mm["tiv_share"], "sbt_share": mm["sbt_share"]})
    q5 = pd.DataFrame(rows)
    summary["Q5_cue_mismatch"] = q5.to_dict("records")
    lines += ["## Q5 Within-episode gaps by recorded cue (teacher present, descriptive)", "",
              "| Group | Gap by teacher in view | Gap by seen by teacher | Share in view | Share seen |",
              "|---|---|---|---|---|"]
    lines += [f"| {r.group} | {fmt(r.gap_teacher_in_view)} | {fmt(r.gap_seen_by_teacher)} | "
              f"{fmt(r.tiv_share)} | {fmt(r.sbt_share)} |" for r in q5.itertuples()]

    # ------------------------------------------------ replication of Phase 3
    rep = {}
    for m in cfg["metrics"]:
        pm = V.per_map_metric(absent, m, ["map", "group"]).groupby("group")[m].mean()
        rep[m] = {g: float(pm[g]) for g in ("indifferent", "half", "minimal") if g in pm}
    summary["replication_full_view_ladder"] = rep
    lines += ["", "## Replication of the Phase 3 ladder on suite v2 (teacher absent, full view)", "",
              "| Metric | indifferent (w 0) | half (w 0.5) | minimal (w 1) |", "|---|---|---|---|"]
    lines += [f"| {m} | {fmt(v.get('indifferent'))} | {fmt(v.get('half'))} | {fmt(v.get('minimal'))} |"
              for m, v in rep.items()]

    summary["checks"] = checks
    lines = lines[:3] + ["## Pass/fail checks", "", "| Check | Passed |", "|---|---|"] + [
        f"| {k} | {'yes' if v else 'no'} |" for k, v in checks.items()] + [""] + lines[3:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
