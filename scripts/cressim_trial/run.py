"""CRESSim GPU trials on the project's reference cucumber. Run at repository root."""

import argparse
import ctypes as ct
import json
from pathlib import Path
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

root = Path(__file__).resolve().parents[2]
cache = root / ".cache/cressim-run"
sample = root / "examples/cucumber/2026_0000"
meta = json.loads((sample / "sample.json").read_text())
with np.load(sample / "mpm.npz") as data:
    rest, tissue, vol = (data[n] for n in ("x", "tissue", "volume_m3"))
plane = float((rest[:, 1].min() + rest[:, 1].max()) / 2)
n = len(rest)
properties = [meta["materials"][name] for name in meta["material_order"]]
rho, e, nu = np.array(
    [[p["density_kg_m3"], p["E_MPa"] * 1e6, p["nu"]] for p in properties]
)[tissue - 1].T
mass = vol * rho
xyz_mass = np.ascontiguousarray(np.c_[rest * 100, mass * 1000], np.float32)
elastic = np.ascontiguousarray(
    10 * np.c_[e * nu / (1 + nu) / (1 - 2 * nu), e / (2 * (1 + nu))], np.float32
)
dt = 1 / (65536 * int(os.getenv("CRESSIM_SUBDIVIDE", "1")))
block = 1 / 4096
start_tip, half_tip, full_tip, clear_tip = 0.12369, 0.06656, 0.02100, 0.13869
half_height = 0.0307
width = 0.004  # total contact width of the analytic plane
raw_edges = cKDTree(rest).query_pairs(0.0025, output_type="ndarray")
a, b = raw_edges.T
flipped = rest[a, 1] > rest[b, 1]
low, high = np.where(flipped, b, a), np.where(flipped, a, b)
cross = (rest[low, 1] < plane) & (rest[high, 1] >= plane)
low, high = low[cross], high[cross]
z0 = (rest[low, 2] + rest[high, 2]) * 0.5
upper = z0 > half_tip + 0.008
lower = z0 < half_tip - 0.008
# Fixed local tissue patches for the partial-cut opening test.
offset = rest[:, 1] - plane
grip = (abs(offset) > 0.012) & (abs(offset) < 0.035) & (rest[:, 2] > half_tip + 0.012)
grip_sign = np.ascontiguousarray(np.sign(offset) * grip, np.int8)
visual_ids = np.sort(np.random.default_rng(2026).choice(n, 24000, replace=False))
slice_ids = np.flatnonzero(abs(rest[:, 0] + 0.028) < 0.0035)
visual_ids = np.union1d(visual_ids, slice_ids)
if os.getenv("CRESSIM_ALL_FRAMES") == "1":
    visual_ids = np.arange(n)


def ptr(a):
    return ct.c_void_p(a.ctypes.data)


handle, f32 = ct.c_void_p, ct.c_float
lib = ct.CDLL(str(cache / "bridge.so"))
for name, ret, args in [
    ("start_trial", handle, [ct.c_int, handle, handle, f32, f32, f32, f32, f32]),
    ("advance_trial", None, [handle, f32, f32]),
    ("positions", None, [handle, handle]),
    ("add_y_velocity", None, [handle, handle]),
    ("retain_material", None, [handle, handle]),
    ("drive_grips", None, [handle, handle, handle, f32, f32]),
    ("finish_trial", None, [handle]),
]:
    fn = getattr(lib, name)
    fn.restype = ret
    fn.argtypes = args


def run(name):
    half = name.startswith(("half", "sham"))
    sham = name.startswith("sham")
    withdrawal_target = clear_tip if half else 0.245
    split = name.startswith("split")
    passive_only = name == "split_passive"
    loaded = name.endswith("_load")
    trial = lib.start_trial(
        n,
        ptr(xyz_mass),
        ptr(elastic),
        plane * 100,
        (0.20 if sham else start_tip) * 100,
        dt,
        0.4,
        width * 50,
    )
    if not trial:
        raise RuntimeError("scene allocation failed")
    curr = rest.copy()
    tip = start_tip
    t = 0
    sides = np.zeros(n, np.int8)
    keep = np.ones(n, bool)
    history = []
    rows = []
    crossings = {}
    min_abs_probe_y = float("inf")

    def record(phase):
        if not np.isfinite(curr[keep]).all():
            raise FloatingPointError(f"{name} nonfinite particle state at {t}")
        if (curr[keep].min(0) < np.array([-0.112, -0.292, -0.022])).any() or (
            curr[keep].max(0) > np.array([0.072, 0.292, 0.342])
        ).any():
            raise RuntimeError(
                f"{name} particles reached outer grid boundary at {t}: {curr[keep].min(0)} {curr[keep].max(0)}"
            )
        gap = (curr[high, 1] - curr[low, 1]) * 1000
        signed = np.where(sides != 0, sides, np.where(offset > 0, 1, -1))
        centers = []
        for side in (-1, 1):
            mask = (signed == side) & keep
            centers.append(
                float(np.average(curr[mask, 1], weights=mass[mask]) * 1000)
                if mask.any()
                else None
            )
        row = dict(
            time_s=t,
            phase=phase,
            tip_z_m=(0.20 if sham else tip),
            mean_y_mm=centers,
            sample_gap_mm=[float(np.median(gap[m])) for m in (upper, lower)],
            sample_gap_p95_mm=[
                float(np.percentile(gap[m], 95)) for m in (upper, lower)
            ],
        )
        history.append(curr[visual_ids].copy())
        rows.append(row)

    def advance(pos, count=1):
        nonlocal t, tip, min_abs_probe_y
        for _ in range(count):
            old = tip
            trace = not sham and abs(pos - old) > 0
            before = curr.copy() if trace else None
            tip = pos
            lib.advance_trial(trial, (0.20 if sham else tip) * 100, block)
            lib.positions(trial, ptr(curr))
            t += block
            if trace:
                # Linear inter-sample paths; count traversal of the finite mid-surface
                # inside its edge, away from the 1 mm uncertainty near its rim.
                indices = np.flatnonzero(
                    (before[:, 1] >= plane) != (curr[:, 1] >= plane)
                )
                alpha = (plane - before[indices, 1]) / (
                    curr[indices, 1] - before[indices, 1]
                )
                at = before[indices] + alpha[:, None] * (
                    curr[indices] - before[indices]
                )
                depth = at[:, 2] - (old + alpha * (pos - old))
                through = (
                    (depth > 0.001)
                    & (depth < 2 * half_height - 0.001)
                    & (abs(at[:, 0] + 0.025) < 0.079)
                )
                for ident in indices[through]:
                    crossings.setdefault(int(ident), [0, 0])[int(pos > old)] += 1
                probe = curr[71186]
                if pos < old and pos + 0.001 < probe[2] < pos + 2 * half_height - 0.001:
                    min_abs_probe_y = min(
                        min_abs_probe_y, float(probe[1] - plane) * 1000
                    )
            if pos < old:
                touch = (
                    (sides == 0)
                    & (curr[:, 2] >= pos)
                    & (abs(curr[:, 0] + 0.025) <= 0.08)
                )
                sides[touch] = np.where(curr[touch, 1] >= plane, 1, -1)

    def stroke(target, phase):
        direction = np.sign(target - tip)
        it = 0
        while abs(target - tip) > 1e-8:
            advance(tip + direction * min(abs(target - tip), 4 * block))
            it += 1
            if it % 8 == 0 and abs(target - tip) > 1e-8:
                record(phase)
        record(phase)

    record("intact")
    stroke(half_tip, "half_depth")
    half_snapshot = curr.copy()
    if not half:
        stroke(full_tip, "through_depth")
    at_depth = curr.copy()
    marked = int(np.count_nonzero(sides))
    validlinks = raw_edges[sides[a] * sides[b] != -1]
    components, labels = connected_components(
        coo_matrix((np.ones(len(validlinks), bool), validlinks.T), shape=(n, n)),
        directed=False,
    )
    if not half and (components != 2 or marked != n):
        raise ValueError(f"full cut tracker: {components=} {marked=} {n=}")
    if half and components != 1:
        raise ValueError(f"partial cut unexpected tracker {components=}")
    early = split and os.getenv("CRESSIM_SPLIT_AT_DEPTH") == "1"
    if not early:
        stroke(withdrawal_target, "withdraw")
    if split:
        # Load is applied outwards only. Each replay evolves one retained component
        # on its own grid, keeping that component's entire pre-cut particle history.
        keep = sides == (-1 if passive_only else 1)
        lib.retain_material(trial, ptr(np.ascontiguousarray(keep, np.uint8)))
    if early:
        stroke(withdrawal_target, "withdraw")
    withdrawn = curr.copy()

    def blade_clearance():
        band = (
            keep
            & (abs(curr[:, 1] - plane) <= width / 2 + 0.004)
            & (abs(curr[:, 0] + 0.025) <= 0.084)
        )
        return (
            float((0.20 if sham else tip) - curr[band, 2].max()) if band.any() else 1.0
        )

    clearance = blade_clearance()
    if clearance < 0.002:
        raise RuntimeError(f"{name} blade clearance {clearance} m at load onset")
    anchor_y = np.ascontiguousarray(withdrawn[:, 1])
    if not half and loaded:
        lib.add_y_velocity(
            trial, ptr(np.ascontiguousarray(0.6 * (sides == 1), np.float32))
        )
    for k in range(16):
        for j in range(8):
            if half:
                speed = 0.128 if loaded else 0.0
                distance = speed * (8 * k + j) * block
                lib.drive_grips(trial, ptr(grip_sign), ptr(anchor_y), distance, speed)
            advance(withdrawal_target)
            clearance = min(clearance, blade_clearance())
        record("opening" if half else "move_away")
    dest = Path(os.getenv("CRESSIM_RAW_OUT", str(cache)))
    dest.mkdir(parents=True, exist_ok=True)
    if clearance < 0.002:
        raise RuntimeError(f"{name} tool returned to test region: {clearance} m")
    out = dest / (name + ".npz")
    np.savez(
        out,
        rows=json.dumps(rows),
        visual_ids=visual_ids,
        frames=np.stack(history),
        full_depth=at_depth,
        half_depth=half_snapshot,
        withdrawn=withdrawn,
        end=curr,
        side=sides,
        keep=keep,
        min_tool_clearance_m=clearance,
        dt_s=dt,
        early_split=early,
        crossings=json.dumps(crossings),
        probe_min_signed_y_mm=min_abs_probe_y,
        components=components,
        marked=marked,
        edges=np.stack([low, high]),
        upper=upper,
        lower=lower,
    )
    print(name, "tracker", components, marked, "final rows", rows[-1], flush=True)
    lib.finish_trial(trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs",
        nargs="+",
        choices=[
            "half_load",
            "half_free",
            "sham_load",
            "sham_free",
            "native_load",
            "native_free",
            "split_load",
            "split_free",
            "split_passive",
        ],
    )
    for key in parser.parse_args().runs:
        run(key)
