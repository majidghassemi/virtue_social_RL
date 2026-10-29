"""Generate the map suite (seen, held-out position, held-out structure).

Usage:
    python scripts/make_maps.py                       # uses configs/maps.yaml
    python scripts/make_maps.py --config configs/maps.yaml --preview results/maps_v1.png

Writes one JSON file with every map in both teacher conditions, its stats and
a provisional detour-cost band (tertiles of the structural virtue cost over the
seen maps). Phase 3 replaces the band with an empirical measurement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from virtue_rl.envs import maps as M


def apple_cells(ascii_map):
    return {tuple(p) for p in np.argwhere(M.to_array(ascii_map) == "A")}


def overlap(a, b):
    return len(a & b) / max(1, min(len(a), len(b)))


BANDS = ("low", "medium", "high")


def band_of(cost: float, cuts: list[float]) -> str:
    return "low" if cost <= cuts[0] else ("medium" if cost <= cuts[1] else "high")


def make_entry(split, idx, family, seed, patch, n):
    spec = M.MapSpec(family=family, seed=seed, num_patches=tuple(patch["num_patches"]),
                     patch_radii=tuple(patch["radii"]))
    present, absent, info = M.generate(spec)
    M.validate_map(present, n, teacher_present=True)
    M.validate_map(absent, n, teacher_present=False)
    return M.MapEntry(name=f"{split}_{idx:02d}_{family}", split=split, family=family,
                      seed=seed, ascii_present=present, ascii_absent=absent,
                      stats=M.map_stats(present) | info)


def build(cfg: dict) -> tuple[list[M.MapEntry], dict]:
    """Seen maps first (they define the band cuts), then band-stratified held-out splits.

    Held-out splits get an equal number of maps per band (and per family), so
    a held-out result reflects the layout shift, not a shift in how costly
    virtue is.
    """
    n = int(cfg["num_agents"])
    patch = cfg["patch"]
    splits = cfg["splits"]

    sc = splits["seen"]
    seen = [make_entry("seen", i, sc["families"][i % len(sc["families"])],
                       int(sc["first_seed"]) + i, patch, n) for i in range(int(sc["count"]))]
    costs = [e.stats["structural_virtue_cost"] for e in seen]
    cuts = [float(np.quantile(costs, 1 / 3)), float(np.quantile(costs, 2 / 3))]
    for e in seen:
        e.band = band_of(e.stats["structural_virtue_cost"], cuts)
    seen_apples = [apple_cells(e.ascii_present) for e in seen]
    entries = list(seen)

    for split in ("heldout_position", "heldout_structure"):
        sc = splits[split]
        families = sc["families"]
        cells_per_quota = len(families) * len(BANDS)
        if int(sc["count"]) % cells_per_quota:
            raise ValueError(f"{split}: count must be a multiple of families x bands = {cells_per_quota}")
        quota = {(f, b): int(sc["count"]) // cells_per_quota for f in families for b in BANDS}
        seed = int(sc["first_seed"])
        made: list[M.MapEntry] = []
        tries = 0
        while any(v > 0 for v in quota.values()):
            tries += 1
            if tries > 20000:
                raise RuntimeError(f"{split}: could not fill band quotas {quota}")
            family = families[tries % len(families)]
            e = make_entry(split, len(made), family, seed, patch, n)
            seed += 1
            e.band = band_of(e.stats["structural_virtue_cost"], cuts)
            if quota[(family, e.band)] == 0:
                continue
            if split == "heldout_position":
                cells = apple_cells(e.ascii_present)
                if any(overlap(cells, s) > sc["max_apple_overlap"] for s in seen_apples):
                    continue
            quota[(family, e.band)] -= 1
            made.append(e)
        entries.extend(made)

    meta = {"config": cfg, "band_cuts_structural_cost": cuts,
            "band_note": "provisional, from structural cost; Phase 3 measures it empirically",
            "grid_size": list(entries[0].grid_size)}
    return entries, meta


def preview(entries: list[M.MapEntry], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    code = {" ": 0, "W": 1, "A": 2, "P": 3, "Q": 4}
    cmap = ListedColormap(["#f4f1ea", "#4a4a48", "#3b8f3b", "#9ab7d6", "#d9822b"])
    cols = 6
    rows = int(np.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 1.9))
    for ax, e in zip(axes.flat, entries):
        g = np.vectorize(code.get)(M.to_array(e.ascii_present))
        ax.imshow(g, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(f"{e.name}\ncost {e.stats['structural_virtue_cost']:.2f} ({e.band})",
                     fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(entries):]:
        ax.axis("off")
    fig.suptitle("Map suite (teacher-present version). Green apples, blue spawns, "
                 "orange teacher start, right edge = sealed teacher bay", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/maps.yaml"))
    p.add_argument("--preview", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    entries, meta = build(cfg)
    M.save_suite(entries, cfg["out"], meta)
    print(f"wrote {len(entries)} maps to {cfg['out']}")
    for e in entries:
        s = e.stats
        print(f"  {e.name:<32} apples {s['apples']:>3} components {s['components']:>2} "
              f"cost {s['structural_virtue_cost']:.3f} band {e.band}")
    print(f"band cuts (structural cost): {meta['band_cuts_structural_cost']}")
    if args.preview:
        preview(entries, args.preview)
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
