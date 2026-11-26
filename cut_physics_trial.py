"""Reproducible straight-knife coupon trial in SI units with substep wrench traces."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src/mpm_cutting"))
from reference_cut import ReferenceCut, knife_geometry  # noqa: E402


def block(spacing, stagger=False):
    dims = np.rint(np.array([0.006, 0.012, 0.006]) / spacing).astype(int)
    indices = np.indices(dims).reshape(3, -1).T + 0.5
    if stagger:
        # Stratify within the same physical cells, preserving point count and total mass.
        parity = (np.floor(indices[:, 0]) + np.floor(indices[:, 2])).astype(int) % 2
        indices[:, 1] += 0.25 * (2 * parity - 1)
    return (indices - [dims[0] / 2, dims[1] / 2, 0]) * spacing


def protocol(case, speed, dt):
    height = 0.007
    chunks = []
    marks = {}
    steps = 0

    def move(name, target, velocity, anchor=1.0, accel=0.0, duration=None):
        nonlocal steps, height
        if duration is None:
            duration = abs(target - height) / velocity
        n = max(1, round(duration / dt))
        vel = (target - height) / (n * dt)
        chunk = np.tile([height, vel, anchor, accel], (n, 1))
        chunk[:, 0] += np.arange(n) * dt * vel
        height = target
        chunks.append(chunk)
        steps += n
        marks[name] = steps

    target = {"air": 0.0068, "indent": 0.0049, "half": 0.003, "full": -0.001}[case]
    move("down", target, speed)
    move("withdrawn", 0.007, speed)
    return np.row_stack(chunks), marks


def summaries(
    sim,
    states,
    path,
    dt,
    speed,
    action,
    marks,
    spacing,
    dx,
    case,
    keff,
    friction,
    material,
    xyz,
    wrist,
):
    data = {k: getattr(sim.log, k).numpy() for k in sim.log._cls.__annotations__}
    data["action"] = action
    mass = sim.p.mass
    if not all(np.isfinite(a).all() for a in data.values()) or not all(
        np.isfinite(a).all() for snap in states.values() for a in snap.values()
    ):
        raise FloatingPointError("Nonfinite state or trace")
    np.savez_compressed(path / "substeps.npz", **data)
    np.savez_compressed(
        path / "states.npz",
        rest=xyz,
        **{
            key + "_" + phase: state
            for phase, end in marks.items()
            for key, state in states[end].items()
        },
    )
    n = len(action)
    # Aggregate intervals without straddling command changes.
    starts = np.unique(
        np.r_[np.arange(0, n, max(1, round(0.001 / dt))), list(marks.values()), 0, n]
    )
    rows = []
    for a, b in zip(starts[:-1], starts[1:]):
        elapsed = dt * (b - a)
        f = data["force_impulse"][a:b].sum(0, dtype=np.float64) / elapsed
        m = data["moment_impulse"][a:b].sum(0, dtype=np.float64) / elapsed
        rows.append(
            [
                b * dt,
                (0.006 - action[b - 1, 0] - action[b - 1, 1] * dt) * 1000,
                *f,
                *m,
                data["tool_work"][a:b].sum(dtype=float),
                *data["energy"][b - 1],
                int(data["crossings"][a:b].sum()),
                int(data["inside"][a:b].sum()),
            ]
        )
    with (path / "wrench.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "time_s",
                "depth_mm",
                "Fx_N",
                "Fy_N",
                "Fz_N",
                "Mx_Nm",
                "My_Nm",
                "Mz_Nm",
                "window_work_J",
                "kinetic_J",
                "strain_J",
                "entries",
                "inside_particle_steps",
            ]
        )
        out.writerows(rows)
    down = marks["down"]
    load = np.array([row for row in rows if row[0] <= down * dt + 1.0e-12])
    impulse = data["force_impulse"].sum(0, dtype=float)
    loadwork = data["tool_work"][:down].sum(dtype=float)
    p_external = (
        -impulse
        + data["clamp_impulse"].sum(0, dtype=float)
        + data["load_impulse"].sum(0, dtype=float)
    )
    final = states[n]
    total_momentum = final["v"].sum(0, dtype=float) * mass
    energy = data["energy"][-1].sum()
    residual = data["tool_work"].sum(dtype=float) - energy
    if data["min_j"].min() <= 0.0:
        raise FloatingPointError("Inverted material element")
    summary = dict(
        case=case,
        speed_m_s=speed,
        dt_s=dt,
        spacing_m=spacing,
        grid_spacing_m=dx,
        keff_J_m2=keff,
        knife_friction=friction,
        material=material,
        particles=len(xyz),
        sample_size_m=[0.006, 0.012, 0.006],
        wrist_local_m=list(wrist),
        anchor_abs_y_m=0.005,
        gravity_m_s2=0.0,
        elastic_only=True,
        pic_blend_per_step=sim.run.pic,
        normal_impulse_scale=sim.run.normal_scale,
        face_friction=0.2,
        radius_m=sim.p.radius,
        peak_window_Fz_N=float(load[:, 4].max()),
        My_at_Fz_peak_Nm=float(load[np.argmax(load[:, 4]), 6]),
        downstroke_work_mJ=loadwork * 1000,
        cycle_work_mJ=float(data["tool_work"].sum(dtype=float) * 1000),
        cumulative_gate_work_mJ=float(final["work"].sum(dtype=float) * 1000),
        threshold_reached=int(
            np.sum(final["work"] >= keff * spacing**2 * (1 - 1.0e-5))
        ),
        min_J=float(data["min_j"].min()),
        segment_entries=int(data["crossings"].sum()),
        max_inside_particles=int(data["inside"].max()),
        inside_particle_steps=int(data["inside"].sum()),
        max_momentum_error_Ns=float(abs(total_momentum - p_external).max()),
        unallocated_cycle_work_mJ=float(residual * 1000),
        particles_by_side_down=[
            int(v) for v in np.bincount(states[down]["side"], minlength=3)
        ],
        max_down_displacement_mm=float(
            np.linalg.norm(states[down]["x"] - xyz, axis=1).max() * 1000
        ),
        steps=n,
        phase_end_steps=marks,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["air", "indent", "half", "full"])
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=5.0e-6)
    parser.add_argument("--spacing", type=float, default=0.001)
    parser.add_argument("--keff", type=float, default=80.0)
    parser.add_argument("--friction", type=float, default=0.03)
    parser.add_argument(
        "--lateral-mm",
        type=float,
        default=0.0,
        help="Knife translation across particle rows",
    )
    parser.add_argument(
        "--stagger",
        action="store_true",
        help="Perturb sampling within cells of the same specimen",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Export measured particle and tool samples",
    )
    parser.add_argument(
        "--fixed-contact-stiffness",
        action="store_true",
        help="Hold s_n/dt^2 at its reference value for the timestep sensitivity check",
    )
    parser.add_argument(
        "--contact-samples",
        action="store_true",
        help="Save and re-integrate sampled point reactions",
    )
    parser.add_argument("--wrist-x-mm", type=float, default=100.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not all(
        np.isfinite(v) and v > 0 for v in (args.speed, args.dt, args.spacing, args.keff)
    ):
        parser.error("speed, dt, spacing and keff must be finite and positive")
    if args.spacing not in (0.001, 0.0005) or not 0 <= args.friction <= 1:
        parser.error(
            "use 1 or 0.5 mm spacing and a friction coefficient between 0 and 1"
        )
    material = json.loads(
        (ROOT / "examples/cucumber/2026_0000/sample.json").read_text()
    )["materials"]["Pulp"]
    wave = np.sqrt(
        material["E_MPa"]
        * 1.0e6
        * (1 - material["nu"])
        / (material["density_kg_m3"] * (1 + material["nu"]) * (1 - 2 * material["nu"]))
    )
    dx = 2 * args.spacing
    if args.dt > 0.2 * dx / wave:
        parser.error("timestep exceeds the coupon CFL limit")
    action, marks = protocol(args.case, args.speed, args.dt)
    xyz = block(args.spacing, args.stagger)
    if not np.isfinite(args.wrist_x_mm):
        parser.error("wrist coordinates must be finite")
    offset = args.lateral_mm * 0.001
    if not np.isfinite(offset) or abs(offset) > 0.002:
        parser.error("keep the blade within 2 mm of the coupon center")
    mesh = knife_geometry(ROOT)
    mesh[0][:, 1] += offset
    mesh[2][:, 1] += offset
    wrist = [args.wrist_x_mm * 0.001, offset, 0.03]
    sim = ReferenceCut(
        mesh,
        xyz,
        args.spacing,
        dx,
        material,
        action,
        args.dt,
        args.keff,
        args.friction,
        args.case in ("half", "full"),
        wrist,
        0.005,
        normal_scale=0.25 * (args.dt / 5.0e-6) ** 2
        if args.fixed_contact_stiffness
        else 0.25,
    )
    frame_steps = np.unique(
        np.r_[np.linspace(0, len(action), 65).astype(int), list(marks.values())]
    )
    snapshots = (
        frame_steps[frame_steps > 0]
        if args.timeline or args.contact_samples
        else marks.values()
    )
    states = sim.execute(len(action), snapshots, args.contact_samples)
    args.out.mkdir(parents=True, exist_ok=True)
    report = summaries(
        sim,
        states,
        args.out,
        args.dt,
        args.speed,
        action,
        marks,
        args.spacing,
        dx,
        args.case,
        args.keff,
        args.friction,
        material,
        xyz,
        wrist,
    )
    if args.contact_samples:
        selected = np.array(sorted(states))
        pos = np.stack([states[k]["last_surface"] for k in selected])
        reactions = np.stack([states[k]["last_reaction"] for k in selected])
        origins = np.tile(wrist, (len(selected), 1))
        origins[:, 2] += action[selected - 1, 0]
        js = reactions.sum(1, dtype=float)
        ls = np.cross(pos - origins[:, None, :], reactions).sum(1, dtype=float)
        np.testing.assert_allclose(
            js, sim.log.force_impulse.numpy()[selected - 1], rtol=2.0e-5, atol=1.0e-11
        )
        np.testing.assert_allclose(
            ls, sim.log.moment_impulse.numpy()[selected - 1], rtol=2.0e-5, atol=1.0e-12
        )
        np.savez_compressed(
            args.out / "point_contacts.npz",
            step=selected,
            wrist_m=origins,
            surface_m=pos,
            reaction_Ns=reactions,
        )
        report["reintegration_samples"] = len(selected)
        report["nonzero_point_impulses"] = int(
            np.count_nonzero(np.linalg.norm(reactions, axis=2))
        )
    if args.timeline:
        z = np.r_[action[:, 0], action[-1, 0] + args.dt * action[-1, 1]]
        np.savez_compressed(
            args.out / "timeline.npz",
            time_s=frame_steps * args.dt,
            knife_z=z[frame_steps],
            x=np.stack(
                [xyz.astype("f4") if n == 0 else states[n]["x"] for n in frame_steps]
            ),
            side=np.stack(
                [
                    np.zeros(len(xyz), "i4") if n == 0 else states[n]["side"]
                    for n in frame_steps
                ]
            ),
        )
    report["sample_staggered"] = args.stagger
    report["lateral_mm"] = args.lateral_mm
    (args.out / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
