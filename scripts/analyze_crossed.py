"""Analyse the crossed virtue x competence ladder.

Usage:
    python scripts/analyze_crossed.py            # paths from configs/protocol_crossed_v1.yaml

Combines the crossed run (high eps) with the protocol_v1 run (eps 0, 0.1,
0.2) on the same maps, checks the pre-registered gamma criterion and writes
a JSON summary plus a Markdown report.
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


def load(run_dir: Path, maps: list[str] | None = None) -> pd.DataFrame:
    files = sorted(run_dir.glob("counts_*.csv.gz"))
    if maps is not None:
        files = [f for f in files if f.name[len("counts_"):-len(".csv.gz")] in maps]
    if not files:
        raise SystemExit(f"no counts files in {run_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/protocol_crossed_v1.yaml"))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    new = load(Path(cfg["out_dir"]))
    maps = sorted(new["map"].unique().tolist())
    base = load(Path(cfg["combine_with"]), maps)
    base = base[(base.condition == "absent") & (base.disposition == "mixture")]
    df = pd.concat([base, new], ignore_index=True)
    learners = df[df.role == "learner"]
    eps_levels = sorted(learners.eps.unique().tolist())

    metrics = cfg["metrics"] + cfg.get("exploratory_metrics", [])
    res = pd.concat([V.crossed_rank_validity(learners, m) for m in metrics], ignore_index=True)
    res["pre_registered"] = res.metric.isin(cfg["metrics"])
    min_gamma = cfg["criteria"]["crossed_rank_validity"]["min_gamma"]
    checks = {f"crossed_rank_validity::{r.metric}": bool(r.gamma >= min_gamma - 1e-9)
              for r in res.itertuples() if r.pre_registered}

    comp = (V.per_map_metric(learners, "apples_per_episode", ["map", "w", "eps"])
            .groupby(["w", "eps"])["apples_per_episode"].mean())
    summary = {"maps": maps, "eps_levels": eps_levels, "rows": int(len(df)),
               "crossed_rank_validity": res.to_dict("records"), "checks": checks,
               "apples_per_episode": {f"w={k[0]:g},eps={k[1]:g}": float(v) for k, v in comp.items()}}
    out = Path(cfg["summary"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=float) + "\n")

    lines = [f"# Crossed virtue x competence ladder ({cfg['out_dir']} + {cfg['combine_with']})", "",
             f"Maps: {len(maps)} (seen split), teacher absent, eps levels {eps_levels}, "
             f"learner agent-episodes: {len(learners)}", "",
             "## Pre-registered check (gamma >= " + f"{min_gamma})", "",
             "| Check | Passed |", "|---|---|"]
    lines += [f"| {k} | {'yes' if v else 'no'} |" for k, v in checks.items()]
    lines += ["", "## Results", "",
              "| Metric | Pre-registered | Gamma | Misordered pairs | tau-b | Per-map gamma median (min) | Worst misorder |",
              "|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        wm = r.worst_misorder
        worst = "none" if wm is None else (
            f"(w={wm['less_virtuous_cell'][0]:g}, eps={wm['less_virtuous_cell'][1]:g}) below "
            f"(w={wm['more_virtuous_cell'][0]:g}, eps={wm['more_virtuous_cell'][1]:g}) by "
            f"{wm['excess_harm']:.3f}")
        lines.append(f"| {r.metric} | {'yes' if r.pre_registered else 'no'} | {fmt(r.gamma)} | "
                     f"{fmt(r.misordered_share)} | {fmt(r.tau_b)} | "
                     f"{fmt(r.per_map_gamma_median)} ({fmt(r.per_map_gamma_min)}) | {worst} |")

    lines += ["", "## Cell values (mean over maps)", ""]
    for r in res.itertuples():
        lines += [f"### {r.metric}", "", "| w \\ eps | " + " | ".join(f"{e:g}" for e in eps_levels) + " |",
                  "|---" * (len(eps_levels) + 1) + "|"]
        for w in sorted(learners.w.unique()):
            vals = [r.cell_values.get(f"w={w:g},eps={e:g}", np.nan) for e in eps_levels]
            lines.append(f"| {w:g} | " + " | ".join(fmt(v) for v in vals) + " |")
        lines.append("")
    lines += ["### apples_per_episode (competence)", "",
              "| w \\ eps | " + " | ".join(f"{e:g}" for e in eps_levels) + " |",
              "|---" * (len(eps_levels) + 1) + "|"]
    for w in sorted(learners.w.unique()):
        lines.append(f"| {w:g} | " + " | ".join(fmt(comp.get((w, e), np.nan), 2) for e in eps_levels) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
