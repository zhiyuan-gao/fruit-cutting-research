"""Replay the withdrawn reference-coupon state with and without a one-sided load."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cut_physics_trial import ReferenceCut, knife_geometry  # noqa: E402
from pieces import SingleCutPieces  # noqa: E402


def probe(folder):
    d = json.loads((folder / "result.json").read_text())
    if d.get("lateral_mm", 0.0) != 0.0:
        raise ValueError("This loading protocol uses the centered cut")
    src = np.load(folder / "states.npz")
    xyz = src["rest"]
    spacing, dx, dt = d["spacing_m"], d["grid_spacing_m"], d["dt_s"]
    # Both pieces are free and the knife is clear. Positive Y is outward.
    phases = [("out", 0.012, 16.0), ("back", 0.034, -16.0), ("reopen", 0.045, 24.0)]
    chunks = [np.tile([0.007, 0.0, 0.0, a], (round(t / dt), 1)) for _, t, a in phases]
    action = np.concatenate(chunks)
    ends = np.cumsum([len(c) for c in chunks])
    starts = np.r_[0, ends[:-1]]
    results = {}
    for loaded in (False, True):
        inputs = action.copy()
        if not loaded:
            inputs[:, 3] = 0.0
        sim = ReferenceCut(
            knife_geometry(ROOT),
            xyz,
            spacing,
            dx,
            d["material"],
            inputs,
            dt,
            d["keff_J_m2"],
            d["knife_friction"],
            True,
            d["wrist_local_m"],
            d["anchor_abs_y_m"],
        )
        for k in ("x", "v", "side", "F", "work"):
            getattr(sim.p, k).assign(src[k + "_withdrawn"])
        keyframes = np.unique(np.r_[np.linspace(1, len(inputs), 30).astype(int), ends])
        states = sim.execute(len(inputs), keyframes if loaded else ends)
        if loaded:
            np.savez_compressed(
                folder / "manipulation.npz",
                time_s=dt * keyframes,
                x=np.stack([states[k]["x"] for k in keyframes]),
                side=np.stack([states[k]["side"] for k in keyframes]),
                knife_z=np.full(len(keyframes), inputs[0, 0]),
            )
        impulse = sim.log.pair_impulse.numpy()
        tool = sim.log.force_impulse.numpy()
        stats = []
        lo, hi = xyz[:, 1] < 0.0, xyz[:, 1] > 0.0
        # Probe originally neighboring columns separately above and below the half-cut plane.
        near_left = np.flatnonzero(np.isclose(xyz[:, 1], -0.5 * spacing))
        near_right = np.flatnonzero(np.isclose(xyz[:, 1], 0.5 * spacing))
        assert np.allclose(xyz[near_left][:, [0, 2]], xyz[near_right][:, [0, 2]])
        for (name, _, _), start, end in zip(phases, starts, ends):
            state = states[end]
            if not np.isfinite(state["x"]).all():
                raise FloatingPointError("Nonfinite manipulation replay")
            dy = state["x"][:, 1] - src["x_withdrawn"][:, 1]
            gaps = state["x"][near_right, 1] - state["x"][near_left, 1]
            upper = xyz[near_left, 2] > 0.003
            stats.append(
                dict(
                    phase=name,
                    left_dy_mm=float(dy[lo].mean() * 1000),
                    right_dy_mm=float(dy[hi].mean() * 1000),
                    upper_gap_mm=float(np.median(gaps[upper]) * 1000),
                    lower_gap_mm=float(np.median(gaps[~upper]) * 1000),
                    pair_normal_impulse_Ns=float(impulse[start:end].sum(dtype=float)),
                    end_pair_impulse_Ns=float(impulse[end - 1]),
                    tool_impulse_Ns=float(abs(tool[start:end]).sum()),
                    stable_side=bool(
                        np.array_equal(state["side"], src["side_withdrawn"])
                    ),
                )
            )
        results["loaded" if loaded else "unloaded"] = stats
        if loaded and np.all(src["side_withdrawn"] != 0):
            # Start in an existing gap so no physical recontact can masquerade as grid drag.
            open_state = states[ends[0]]
            clear = []
            for strength in (0.0, 16.0):
                segment = np.tile([0.007, 0.0, 0.0, strength], (round(0.006 / dt), 1))
                branch = ReferenceCut(
                    knife_geometry(ROOT),
                    xyz,
                    spacing,
                    dx,
                    d["material"],
                    segment,
                    dt,
                    d["keff_J_m2"],
                    d["knife_friction"],
                    True,
                    d["wrist_local_m"],
                    0.005,
                )
                for k in ("x", "v", "side", "F", "work"):
                    getattr(branch.p, k).assign(open_state[k])
                final = branch.execute(len(segment), [])[len(segment)]
                clear.append(
                    dict(
                        dy=(final["x"][:, 1] - open_state["x"][:, 1]),
                        contact=float(branch.log.pair_impulse.numpy().sum()),
                    )
                )
            delta = clear[1]["dy"] - clear[0]["dy"]
            results["already_open_push"] = dict(
                left_extra_mm=float(delta[lo].mean() * 1000),
                right_extra_mm=float(delta[hi].mean() * 1000),
                normal_contact_Ns=[branch["contact"] for branch in clear],
            )
    values = src["side_withdrawn"]
    signed = np.choose(values, [0, -1, 1]).astype(np.int8)
    ids = SingleCutPieces(xyz, spacing).update(signed)
    results["components"] = len(np.unique(ids))
    results["clamp_and_gravity_enabled"] = False
    results["right_external_force_N"] = float(16 * sim.p.mass * hi.sum())
    results["delta_out_mm"] = {
        side: results["loaded"][0][side + "_dy_mm"]
        - results["unloaded"][0][side + "_dy_mm"]
        for side in ["left", "right"]
    }
    (folder / "piece_probe.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print(folder.name, results["components"], results["delta_out_mm"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cases", nargs="+", type=Path)
    for case in p.parse_args().cases:
        probe(case)
