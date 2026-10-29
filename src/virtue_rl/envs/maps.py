"""Seeded map generation for Commons Harvest, with a teacher bay.

A map is a list of equal-length strings using SocialJax's characters:
    'W' wall, ' ' empty, 'A' apple spawn, 'P' agent spawn, 'Q' inside spawn.

Layout of every generated map (columns left to right):

    [ arena (ARENA_ROWS x ARENA_COLS) ][ wall band ][ teacher bay ]

The arena is where agents play: apple patches in the top rows, an open
middle, and spawn cells in the bottom rows. The wall band and the one-cell
teacher bay to its right exist in every map, so both teacher conditions see
exactly the same grid. The only difference between conditions is where the
single 'Q' cell sits:

    teacher present: 'Q' in the arena, so the teacher starts there and plays
    teacher absent:  'Q' in the bay, so the teacher is walled in, far beyond
                     every agent's view window and zap reach

This works because SocialJax places agent 0 (the teacher) on an inside spawn
cell 'Q' at reset when there is exactly one (harvest_open.py, _reset_state),
and respawns zapped agents only on 'P' cells.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from virtue_rl.envs.geometry import APPLE_NEIGHBOURHOOD

ARENA_ROWS = 16
ARENA_COLS = 22
APPLE_ROWS = 10          # patches live in arena rows 0..APPLE_ROWS-1
SPAWN_ROWS = (13, 14, 15)
OBS_SIZE = 11            # SocialJax default view size

# The view reaches obs_size - 2 = 9 cells ahead, so the bay must sit at least
# 10 columns past the last arena column. validate_map checks this exactly.
BAND_COLS = OBS_SIZE - 2

# Every map is surrounded by a one-cell wall ring. Upstream does not
# bounds-check the one-ahead and two-ahead zap cells, so at the map edge JAX
# indexing wraps negative indices to the opposite edge and clamps too-large
# ones onto the last row or column (NOTES.md, "Zapping"). Without the ring,
# an agent on row 0 zapping upward hits agents on the bottom row, and an agent
# zapping outward at the right or bottom edge hits itself and respawns. With
# the ring, every wrapped or clamped target is a wall.
RING = 1
_INNER_COLS = ARENA_COLS + BAND_COLS + 1          # arena + band + bay column

# Public coordinates, in the final map including the ring.
GRID_ROWS = ARENA_ROWS + 2 * RING
GRID_COLS = _INNER_COLS + 2 * RING
ARENA_Q = (RING + 12, RING + ARENA_COLS // 2)     # teacher start when present
BAY_ROW = RING + ARENA_ROWS // 2
BAY_COL = RING + ARENA_COLS + BAND_COLS
ARENA_COL_RANGE = (RING, RING + ARENA_COLS)        # [start, stop)

FAMILIES = ("open", "divided", "rooms")
NEIGHBOURS = [tuple(map(int, o)) for o in np.asarray(APPLE_NEIGHBOURHOOD)]


@dataclass
class MapSpec:
    family: str
    seed: int
    num_patches: tuple[int, int] = (5, 8)          # inclusive range
    patch_radii: tuple[int, ...] = (0, 1, 2)       # diamond radius per patch
    max_tries: int = 500


@dataclass
class MapEntry:
    name: str
    split: str
    family: str
    seed: int
    ascii_present: list[str]
    ascii_absent: list[str]
    stats: dict = field(default_factory=dict)
    band: str | None = None

    @property
    def grid_size(self) -> tuple[int, int]:
        return (len(self.ascii_present), len(self.ascii_present[0]))

    def ascii(self, teacher_present: bool) -> list[str]:
        return self.ascii_present if teacher_present else self.ascii_absent


# --------------------------------------------------------------- generation
def _diamond(radius: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) <= radius]


def _walls(family: str, rng: np.random.Generator) -> np.ndarray:
    """Boolean wall mask for the arena."""
    w = np.zeros((ARENA_ROWS, ARENA_COLS), bool)
    if family == "open":
        return w
    sep = APPLE_ROWS + 1  # horizontal wall between apples and spawns
    if family == "divided":
        w[sep, :] = True
        for g in rng.choice(np.arange(1, ARENA_COLS - 2), size=2, replace=False):
            w[sep, g:g + 2] = False
        mid = ARENA_COLS // 2
        w[:sep, mid] = True
        gap = int(rng.integers(2, APPLE_ROWS - 2))
        w[gap:gap + 2, mid] = False
        return w
    if family == "rooms":
        w[sep, :] = True
        thirds = [ARENA_COLS // 3, 2 * ARENA_COLS // 3]
        for lo, hi in ((0, thirds[0]), (thirds[0] + 1, thirds[1]), (thirds[1] + 1, ARENA_COLS)):
            g = int(rng.integers(lo, max(lo + 1, hi - 1)))
            w[sep, g:g + 2] = False
        for c in thirds:
            w[:sep, c] = True
            gap = int(rng.integers(1, APPLE_ROWS - 2))
            w[gap:gap + 2, c] = False
        return w
    raise ValueError(f"unknown family {family!r}")


def _place_patches(walls: np.ndarray, spec: MapSpec, rng: np.random.Generator):
    """Non-touching diamond patches: no two patches share a regrowth neighbourhood."""
    apples = np.zeros_like(walls)
    blocked = walls.copy()  # cells an apple may not occupy or neighbour
    patches = []
    target = int(rng.integers(spec.num_patches[0], spec.num_patches[1] + 1))
    for _ in range(spec.max_tries):
        if len(patches) == target:
            break
        r = int(rng.choice(spec.patch_radii))
        cr = int(rng.integers(r, APPLE_ROWS - r))
        cc = int(rng.integers(r, ARENA_COLS - r))
        cells = [(cr + dr, cc + dc) for dr, dc in _diamond(r)]
        if any(walls[p] or apples[p] for p in cells):
            continue
        # Reject if any new apple lies in the regrowth neighbourhood of an
        # existing apple, so every patch stays its own component.
        clash = False
        for (pr, pc) in cells:
            for dr, dc in NEIGHBOURS:
                q = (pr + dr, pc + dc)
                if 0 <= q[0] < ARENA_ROWS and 0 <= q[1] < ARENA_COLS and apples[q]:
                    clash = True
                    break
            if clash:
                break
        if clash:
            continue
        for p in cells:
            apples[p] = True
        patches.append({"center": [cr, cc], "radius": r, "size": len(cells)})
    return apples, patches


def _assemble(walls: np.ndarray, apples: np.ndarray, teacher_present: bool) -> list[str]:
    inner = np.full((ARENA_ROWS, _INNER_COLS), " ", dtype="<U1")
    inner[:, :ARENA_COLS][walls] = "W"
    inner[:, :ARENA_COLS][apples] = "A"
    for r in SPAWN_ROWS:
        inner[r, 1:ARENA_COLS - 1:2] = "P"
    # Wall band, then the bay column (walls except the bay cell).
    inner[:, ARENA_COLS:] = "W"
    grid = np.pad(inner, RING, constant_values="W")
    grid[BAY_ROW, BAY_COL] = " "
    if teacher_present:
        grid[ARENA_Q] = "Q"
    else:
        grid[BAY_ROW, BAY_COL] = "Q"
    return ["".join(row) for row in grid]


def generate(spec: MapSpec) -> tuple[list[str], list[str], dict]:
    """Return (ascii_present, ascii_absent, info) for one seeded map."""
    if spec.family not in FAMILIES:
        raise ValueError(spec.family)
    rng = np.random.default_rng(spec.seed)
    for _ in range(spec.max_tries):
        walls = _walls(spec.family, rng)
        apples, patches = _place_patches(walls, spec, rng)
        if len(patches) < spec.num_patches[0]:
            continue
        present = _assemble(walls, apples, True)
        if not _all_reachable(present):
            continue
        absent = _assemble(walls, apples, False)
        return present, absent, {"patches": patches}
    raise RuntimeError(f"could not generate a valid {spec.family} map for seed {spec.seed}")


# --------------------------------------------------------------- analysis
def to_array(ascii_map: list[str]) -> np.ndarray:
    return np.array([list(row) for row in ascii_map])


def _reachable(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), bool)
    q = deque([start])
    seen[start] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != "W":
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _all_reachable(ascii_map: list[str]) -> bool:
    g = to_array(ascii_map)
    spawns = np.argwhere(g == "P")
    reach = _reachable(g, tuple(spawns[0]))
    targets = (g == "A") | (g == "P") | (g == "Q")
    targets[BAY_ROW, BAY_COL] = False
    return bool(np.all(reach[targets]))


def components(ascii_map: list[str]) -> int:
    """Apple components under the regrowth (radius-2) neighbourhood."""
    g = to_array(ascii_map)
    apples = {tuple(p) for p in np.argwhere(g == "A")}
    seen, count = set(), 0
    for a in apples:
        if a in seen:
            continue
        count += 1
        stack = [a]
        seen.add(a)
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBOURS:
                q = (r + dr, c + dc)
                if q in apples and q not in seen:
                    seen.add(q)
                    stack.append(q)
    return count


def structural_virtue_cost(ascii_map: list[str]) -> float:
    """Share of the initial apple stock a virtuous agent must leave standing.

    Within one component an agent can always take an apple that still has a
    neighbour (remove leaves of a spanning tree), until one apple is left. That
    last apple is the only harmful take, so a virtuous agent forgoes exactly
    one apple per component. This ignores regrowth, so it is a starting-state
    measure. Phase 3 measures the empirical cost with scripted agents.
    """
    n = int((to_array(ascii_map) == "A").sum())
    return components(ascii_map) / n if n else float("nan")


def _window_cells(r: int, c: int, d: int, obs_size: int) -> tuple[int, int, int, int]:
    """Rows and columns (in map coordinates) covered by a view, inclusive."""
    pad = obs_size - 1
    half = obs_size // 2
    x, y = r + pad - half, c + pad - half
    if d == 0:
        x += half - 1
    elif d == 1:
        y += half - 1
    elif d == 2:
        x -= half - 1
    elif d == 3:
        y -= half - 1
    return x - pad, x - pad + obs_size - 1, y - pad, y - pad + obs_size - 1


def validate_map(ascii_map: list[str], num_agents: int, teacher_present: bool,
                 obs_size: int = OBS_SIZE) -> None:
    """Raise ValueError if the map breaks any assumption the study relies on."""
    g = to_array(ascii_map)
    if len({len(row) for row in ascii_map}) != 1:
        raise ValueError("rows have different lengths")
    if (g == "P").sum() < num_agents:
        raise ValueError("fewer 'P' spawn cells than agents")
    q = np.argwhere(g == "Q")
    if len(q) != 1:
        raise ValueError("need exactly one 'Q' cell (the teacher's start)")
    q = tuple(q[0])
    in_bay = q == (BAY_ROW, BAY_COL)
    if in_bay == teacher_present:
        raise ValueError("'Q' is in the wrong place for this teacher condition")
    if not (np.all(g[0] == "W") and np.all(g[-1] == "W") and np.all(g[:, 0] == "W")
            and np.all(g[:, -1] == "W")):
        raise ValueError("map needs a wall ring (edge zaps wrap around otherwise)")
    if not _all_reachable(ascii_map):
        raise ValueError("some apple or spawn cell is unreachable from the spawn area")
    # The bay must be sealed and invisible from every reachable arena cell.
    arena_reach = _reachable(g, tuple(np.argwhere(g == "P")[0]))
    if arena_reach[BAY_ROW, BAY_COL]:
        raise ValueError("teacher bay is reachable from the arena")
    for r, c in np.argwhere(arena_reach):
        for d in range(4):
            r0, r1, c0, c1 = _window_cells(int(r), int(c), d, obs_size)
            if r0 <= BAY_ROW <= r1 and c0 <= BAY_COL <= c1:
                raise ValueError(f"teacher bay visible from arena cell {(r, c)} facing {d}")
            # Zap reach is at most 2 cells, far inside the view window.


def map_stats(ascii_map: list[str]) -> dict:
    g = to_array(ascii_map)
    n_apples = int((g == "A").sum())
    n_comp = components(ascii_map)
    return {
        "apples": n_apples,
        "components": n_comp,
        "structural_virtue_cost": round(n_comp / n_apples, 4) if n_apples else None,
        "arena_walls": int((g[RING:-RING, ARENA_COL_RANGE[0]:ARENA_COL_RANGE[1]] == "W").sum()),
    }


# --------------------------------------------------------------- persistence
def save_suite(entries: list[MapEntry], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "maps": [asdict(e) for e in entries]}
    path.write_text(json.dumps(payload, indent=1) + "\n")


def load_suite(path: str | Path) -> list[MapEntry]:
    payload = json.loads(Path(path).read_text())
    return [MapEntry(**m) for m in payload["maps"]]
