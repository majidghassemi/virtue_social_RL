"""Learning curves from training runs, and the pilot sanity checks.

Usage:
    python scripts/plot_training.py runs/train/prosocial_v1
    python scripts/plot_training.py runs/train/prosocial_v1 --out results/pilot_prosocial_v1

Reads every <run>/metrics.jsonl below the folder, plots the per-update
episode metrics against environment steps, and prints the checks in
docs/pilot.md (training numerics and learning, not ethics results).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PANELS = [
    ("apples", "Apples per episode, all agents", sum),
    ("shaped_return", "Training return per episode, mean over agents", lambda v: sum(v) / len(v)),
    ("harm_commons", "Last apples taken per episode, all agents", sum),
    ("zaps", "Zaps per episode, all agents", sum),
    ("harm_zap", "Zaps that hit an agent, per episode, all agents", sum),
    ("regrown", "Apples regrown per episode", None),
    ("stock", "Mean apple stock", None),
    ("entropy", "Policy entropy (ln 8 = 2.08 is uniform)", None),
    ("loss_value", "Value loss", None),
    ("loss_aux", "Auxiliary loss (social learner)", None),
]


def load(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines() if x.strip()]


def series(rows, key, agg):
    if key not in rows[0]:
        return None
    v = [r[key] for r in rows]
    return np.asarray([agg(x) if agg else x for x in v], float)


def window_mean(x, frac=0.1):
    k = max(1, int(math.ceil(len(x) * frac)))
    return float(np.mean(x[:k])), float(np.mean(x[-k:]))


def checks(name, rows) -> list[str]:
    out = []
    finite = all(np.isfinite(np.asarray(v, float)).all() for r in rows for k, v in r.items()
                 if isinstance(v, (int, float, list)))
    out.append(f"{'pass' if finite else 'FAIL'}  all logged values finite")
    a0, a1 = window_mean(series(rows, "apples", sum))
    out.append(f"{'pass' if a1 > a0 else 'FAIL'}  apples per episode rose: first 10% {a0:.1f}, last 10% {a1:.1f}")
    e0, e1 = window_mean(series(rows, "entropy", None))
    out.append(f"{'pass' if e1 < e0 - 0.05 else 'FAIL'}  entropy fell by more than 0.05: {e0:.3f} to {e1:.3f}")
    out.append(f"info  updates {rows[-1]['update']}, env steps {rows[-1]['env_steps']:,}")
    return [f"{name}: {c}" for c in out]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=None, help="output prefix (default: <folder>/curves)")
    args = p.parse_args()
    runs = sorted(d for d in args.folder.iterdir() if (d / "metrics.jsonl").exists())
    if not runs:
        raise SystemExit(f"no metrics.jsonl below {args.folder}")
    data = {r.name: load(r) for r in runs}
    data = {k: v for k, v in data.items() if v}

    panels = [pn for pn in PANELS if any(pn[0] in rows[0] for rows in data.values())]
    cols = 3
    nrow = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(nrow, cols, figsize=(5 * cols, 3.2 * nrow), squeeze=False)
    for ax, (key, title, agg) in zip(axes.flat, panels):
        for name, rows in data.items():
            y = series(rows, key, agg)
            if y is None:
                continue
            x = np.asarray([r["env_steps"] for r in rows], float) / 1e6
            ax.plot(x, y, label=name, lw=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("environment steps (millions)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    prefix = args.out or (args.folder / "curves")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=120)

    lines = []
    for name, rows in data.items():
        lines += checks(name, rows)
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {prefix.with_suffix('.png')} and {prefix.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
