"""Single transverse cucumber cut, unilateral loading, recontact and release in MPM."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from pxr import Gf, Sdf, Usd

from simulate import FruitSimulation, ROOT
from sample_view import SampleView


class Trial:
    def __init__(self, env, sample, rate):
        self.env = env
        self.spacing = env.record["spacing_m"]
        rows = json.loads((ROOT / "assets/tissues/cucumber.json").read_text())[
            "materials"
        ]
        palette = {
            row["name"]: ((np.array(row["srgb"]) / 255) ** 2.2).tolist() for row in rows
        }
        self.view = SampleView(Usd.Stage.CreateInMemory(), env, sample, palette)
        self.view.stage.GetRootLayer().customLayerData = {
            "cameraSettings": {
                "Perspective": {
                    "position": Gf.Vec3d(0.36, -0.43, 0.32),
                    "target": Gf.Vec3d(-0.025, 0, 0.07),
                }
            }
        }
        self.phase = self.view.stage.GetDefaultPrim().CreateAttribute(
            "mpm:phase", Sdf.ValueTypeNames.String
        )
        self.masses = env.sim.particles.mass.to_numpy().astype(float)
        self.ids = None
        self.report = dict(
            fruit="cucumber",
            seed=env.record["seed"],
            index=env.record["index"],
            particles=len(self.masses),
            mass_kg=float(self.masses.sum()),
            spacing_m=self.spacing,
            dt_s=env.sim.dt,
            knife_speed_m_s=4.0,
            velocity_decay_per_s=rate,
            frames=[],
            checks={},
        )

    def record(self, phase):
        state = self.env.state()
        x = state["positions"]
        ids = state["piece_ids"]
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite particle position")
        frame = self.view.frame
        grip_clearance = state["knife"]["position"][0] + self.env.grip_guard_x - x[:, 0].max()
        self.report["minimum_recorded_grip_clearance_m"] = min(
            float(grip_clearance), self.report.get("minimum_recorded_grip_clearance_m", float("inf"))
        )
        row = dict(
            frame=frame,
            phase=phase,
            time_s=state["time_s"],
            piece_count=len(np.unique(ids)),
            knife_z_m=float(state["knife"]["position"][2] + self.env.tip[2]),
            compressive_nodes=int(self.env.sim.fracture.contact_nodes[None]),
        )
        if row["piece_count"] == 2:
            if self.ids is None:
                self.ids = ids.copy()
                groups = [np.flatnonzero(ids == i) for i in (1, 2)]
                self.passive, self.loaded = sorted(groups, key=lambda g: x[g, 1].mean())
                a, b = self.env.pieces.edges.T
                forward = (ids[a] == ids[self.loaded[0]]) & (
                    ids[b] == ids[self.passive[0]]
                )
                reverse = (ids[b] == ids[self.loaded[0]]) & (
                    ids[a] == ids[self.passive[0]]
                )
                self.face_loaded = np.r_[a[forward], b[reverse]]
                self.face_passive = np.r_[b[forward], a[reverse]]
                self.report.update(
                    loaded_piece=int(ids[self.loaded[0]]),
                    cut_neighbor_pairs=len(self.face_loaded),
                    particles_per_piece=[len(group) for group in groups],
                    mass_per_piece_kg=[
                        float(self.masses[group].sum()) for group in groups
                    ],
                )
            if not np.array_equal(ids, self.ids):
                raise RuntimeError("Particle membership changed after separation")
            row["passive_y_m"], row["loaded_y_m"] = [
                float(np.average(x[g, 1], weights=self.masses[g]))
                for g in (self.passive, self.loaded)
            ]
            distances = x[self.face_loaded, 1] - x[self.face_passive, 1]
            row["face_pair_y_p10_m"] = float(np.percentile(distances, 10))
        self.phase.Set(phase, frame)
        self.view.update(state)
        self.report["frames"].append(row)
        return row

    def advance(self, ticks, phase, frames):
        for chunk in np.diff(np.linspace(0, ticks, frames + 1, dtype=int)):
            if chunk:
                self.env.step(int(chunk))
                last = self.record(phase)
        return last

    def load(self, acceleration):
        # A distributed diagnostic body force, only on the positive-Y piece.
        force = acceleration * self.masses[self.loaded].sum()
        self.env.set_piece_force(self.report["loaded_piece"], [0, force, 0])

    def run(self):
        env = self.env
        x = env.rest_usd
        plane_y = (x[:, 1].min() + x[:, 1].max()) / 2
        section = x[np.abs(x[:, 1] - plane_y) <= self.spacing]
        pos = env.control.position.copy()
        pos[1] = plane_y - env.tip[1]
        pos[2] = x[:, 2].max() + 0.0003 - env.tip[2]
        speed = self.report["knife_speed_m_s"]
        env.command(pos, [0, 0, -speed])
        checks = self.report["checks"]
        checks["intact"] = self.record("intact")["piece_count"] == 1
        center, size = env.support_boxes[0]
        board = center[2] + size[2] / 2
        self.report.update(
            support_slot_width_m=env.support_boxes[1][0][1]
            - env.support_boxes[1][1][1] / 2
            - (center[1] + size[1] / 2),
            blade_overtravel_in_slot_m=0.010,
        )
        bottom = float(env.knife_mesh.vertices[:, 1].min())
        end = int((pos[2] + bottom - (board - 0.010)) / (speed * env.sim.dt))
        halfway_z = (section[:, 2].max() + section[:, 2].min()) / 2
        half = int((pos[2] + env.tip[2] - halfway_z) / (speed * env.sim.dt))
        midway = self.advance(half, "partial_cut", 8)
        marked = int(np.count_nonzero(env.sim.fracture.side.to_numpy()))
        self.report["particles_marked_at_half_depth"] = marked
        checks["partial_cut_connected"] = midway["piece_count"] == 1 and marked > 0
        full = self.advance(end - half, "finish_cut", 12)
        checks["through_cut_disconnected"] = full["piece_count"] == 2
        print("Cut checks:", checks, flush=True)
        if not checks["through_cut_disconnected"]:
            return
        env.command(env.control.position.copy(), [0, 0, speed])
        lift = int(
            np.ceil((pos[2] + 0.015 - env.control.position[2]) / (speed * env.sim.dt))
        )
        self.advance(lift, "withdraw", 10)
        env.command(env.control.position.copy())
        env.step()
        start = self.record("withdrawn")
        # Branch from identical particle state for the unloaded control. The knife is
        # stationary and clear; the single cut labels cannot change in either branch.
        duration = int(np.ceil(0.032 / env.sim.dt))
        checkpoint = env.sim.particles.to_numpy()
        clock, frame, impulse = env.sim.sim_time, env.frame, env.tool_contact
        env.step(duration)
        baseline = env.state()["positions"]
        free_y = [
            float(np.average(baseline[g, 1], weights=self.masses[g]))
            for g in (self.passive, self.loaded)
        ]
        self.report["unloaded_endpoint_y_m"] = dict(passive=free_y[0], loaded=free_y[1])
        env.sim.particles.from_numpy(checkpoint)
        env.sim.sim_time, env.frame, env.tool_contact = clock, frame, impulse
        acceleration = 180.0
        self.report["body_acceleration_m_s2"] = acceleration
        self.report["push_force_N"] = acceleration * self.masses[self.loaded].sum()
        self.load(acceleration)
        opened = self.advance(duration, "push_out", 12)
        dy = np.array([opened["passive_y_m"], opened["loaded_y_m"]]) - free_y
        self.report["push_displacement_minus_unloaded_mm"] = dict(
            passive=1000 * dy[0], loaded=1000 * dy[1]
        )
        checks["independent_push"] = bool(
            dy[1] > 2 * self.spacing and abs(dy[0]) < 0.2 * dy[1]
        )
        checks["faces_open"] = (
            opened["face_pair_y_p10_m"] > start["face_pair_y_p10_m"] + 2 * self.spacing
            and opened["compressive_nodes"] == 0
        )
        print("Push:", self.report["push_displacement_minus_unloaded_mm"], flush=True)
        # Contact requires the solver to actually exchange compressive impulses.
        self.load(-acceleration)
        near = 1.5 * self.spacing
        ticks = int(np.ceil(0.0015 / env.sim.dt))
        touched = False
        for _ in range(70):
            env.step(ticks)
            row = self.record("return_to_contact")
            if row["compressive_nodes"] > 0 and row["face_pair_y_p10_m"] <= near:
                touched = True
                break
        checks["recontact"] = touched
        print("Recontact:", row, flush=True)
        if touched:
            self.load(acceleration)
            for _ in range(60):
                env.step(ticks)
                row = self.record("reopen")
                if (
                    row["compressive_nodes"] == 0
                    and row["face_pair_y_p10_m"] >= near + 2 * self.spacing
                ):
                    break
            checks["reopened"] = (
                row["compressive_nodes"] == 0
                and row["face_pair_y_p10_m"] >= near + 2 * self.spacing
            )
            checks["stable_ids"] = np.array_equal(env.state()["piece_ids"], self.ids)
        self.load(0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if json.loads((args.sample / "sample.json").read_text())["fruit"] != "cucumber":
        p.error("The transverse cut protocol is defined for cucumber")
    rate = 300.0
    with np.load(args.sample / "mpm.npz") as data:
        y = data["x"][:, 1]
        center_y = float((y.min() + y.max()) / 2)
    env = FruitSimulation(
        args.sample,
        track_pieces=True,
        velocity_decay_per_s=rate,
        slot_center_y=center_y,
    )
    try:
        trial = Trial(env, args.sample, rate)
        trial.run()
        passed = all(trial.report["checks"].values()) and trial.report["checks"].get(
            "reopened", False
        )
        trial.report["passed"] = bool(passed)
        args.out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            stage = Path(tmp) / "cucumber.usdc"
            report = Path(tmp) / "trial.json"
            if not trial.view.stage.GetRootLayer().Export(str(stage)):
                raise OSError("Could not export stage")
            report.write_text(
                json.dumps(trial.report, indent=2, allow_nan=False) + "\n"
            )
            stage.replace(args.out / stage.name)
            report.replace(args.out / report.name)
        print(json.dumps(trial.report["checks"], indent=2), flush=True)
        return 0 if passed else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
