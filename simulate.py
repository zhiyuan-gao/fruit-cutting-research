"""CulinaryCut soft MPM with swept fixed-pit contact and movable convex seeds (SI, USD Z-up)."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import taichi as ti
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent
SOLVER = ROOT / "src/mpm_cutting"
UPSTREAM = ROOT / "third_party/EEF-Cutting-Simulation"
sys.path[:0] = [str(SOLVER), str(UPSTREAM), str(ROOT / ".cache/bullet")]
from adapter import TO_WORLD as AXES

TO_WORLD = AXES.astype(float)
from fracture import SweptFracture, load_fracture_core
from rigid_knife import RigidKnifeCollider, RigidKnifeControl
from rigid_pit import RigidPit
from seed_contact import SeedContact


def pack(name):
    with np.load(ROOT / ".cache/tools" / f"{name}.npz") as f:
        return dict(f)


class MovingSeeds:
    def __init__(self, sim, clouds, masses, knife_mesh, spacing):
        try:
            import pybullet as pb
        except ImportError as exc:
            raise RuntimeError("Run python3 prepare_bullet.py to install the rigid solver.") from exc
        self.pb = pb
        self.client = pb.connect(pb.DIRECT)
        self.sim = sim
        pb.setTimeStep(sim.dt, physicsClientId=self.client)
        pb.setGravity(*list(sim.gravity[None]), physicsClientId=self.client)
        self.bodies = []
        local_clouds = []
        self.rest_centers = []
        for pts, mass in zip(clouds, masses):
            hull = trimesh.convex.convex_hull(pts)
            # Align the reference body frame with its principal axes and install the true hull inertia.
            center = hull.center_mass
            inertia, axes = np.linalg.eigh(hull.moment_inertia * mass / hull.volume)
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            local = (pts - center) @ axes
            shape = pb.createCollisionShape(
                pb.GEOM_MESH, vertices=local.tolist(), physicsClientId=self.client
            )
            b = pb.createMultiBody(
                mass,
                shape,
                basePosition=center,
                baseOrientation=Rotation.from_matrix(axes).as_quat(),
                physicsClientId=self.client,
            )
            pb.changeDynamics(
                b,
                -1,
                localInertiaDiagonal=inertia,
                collisionMargin=0.00002,
                lateralFriction=0.3,
                linearDamping=0,
                angularDamping=0,
                physicsClientId=self.client,
            )
            self.bodies.append(b)
            local_clouds.append(local)
            self.rest_centers.append(center)
        self.contact = SeedContact(sim, local_clouds, masses, spacing)
        shape = pb.createCollisionShape(
            pb.GEOM_MESH,
            vertices=knife_mesh.vertices.tolist(),
            indices=knife_mesh.faces.ravel().tolist(),
            flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=self.client,
        )
        self.knife = pb.createMultiBody(0, shape, basePosition=[0, 2, 0], physicsClientId=self.client)
        pb.changeDynamics(self.knife, -1, collisionMargin=0.00002, physicsClientId=self.client)
        shape = pb.createCollisionShape(pb.GEOM_PLANE, planeNormal=[0, 1, 0], physicsClientId=self.client)
        pb.createMultiBody(
            0, shape, basePosition=[0, float(sim.board_top_y[None]), 0], physicsClientId=self.client
        )
        self.soft_impulse = np.zeros(len(masses))
        self.knife_contacts = 0

    def poses(self):
        pairs = [self.pb.getBasePositionAndOrientation(b, physicsClientId=self.client) for b in self.bodies]
        return np.array([p for p, q in pairs]), Rotation.from_quat([q for p, q in pairs]).as_matrix()

    def step(self, control):
        pb = self.pb
        pos, rot = self.poses()
        velocities = np.asarray([pb.getBaseVelocity(b, physicsClientId=self.client) for b in self.bodies])
        js, jt, activity = self.contact.exchange(pos, rot, velocities[:, 0], velocities[:, 1])
        self.soft_impulse += activity
        for i, b in enumerate(self.bodies):
            pb.applyExternalForce(
                b, -1, js[i] / self.sim.dt, pos[i], pb.WORLD_FRAME, physicsClientId=self.client
            )
            pb.applyExternalTorque(b, -1, jt[i] / self.sim.dt, pb.WORLD_FRAME, physicsClientId=self.client)
        state = control.state()
        # The solid uses the same baked knife mesh and the actual constrained pose as MPM.
        kpos = TO_WORLD.T @ state["position"]
        krot = TO_WORLD.T @ control.rotation.as_matrix()
        pb.resetBasePositionAndOrientation(
            self.knife, kpos, Rotation.from_matrix(krot).as_quat(), physicsClientId=self.client
        )
        pb.resetBaseVelocity(
            self.knife,
            TO_WORLD.T @ state["linear_velocity"],
            TO_WORLD.T @ state["angular_velocity"],
            physicsClientId=self.client,
        )
        pb.stepSimulation(physicsClientId=self.client)
        self.knife_contacts += len(pb.getContactPoints(bodyA=self.knife, physicsClientId=self.client))

    def close(self):
        self.pb.disconnect(self.client)


class FruitSimulation:
    """One active CUDA session per process. Knife positions and seed state use USD metres, Z-up."""

    def __init__(self, folder, fracture=True):
        folder = Path(folder)
        self.record = json.loads((folder / "sample.json").read_text())
        self.temp = tempfile.TemporaryDirectory()
        import mpmcore.sim as native

        core = load_fracture_core(self.temp.name) if fracture else native
        ti.init(arch=ti.cuda, enable_fallback=False, random_seed=self.record["seed"] % (2**31))
        core._TI_INIT = True
        core.KnifeCollider = RigidKnifeCollider
        record = self.record
        names = record["material_order"]
        mats = record["materials"]
        cfg = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
        cfg["knife"]["mesh_path"] = str(UPSTREAM / "assets/knife.obj")
        dx = 0.5 / cfg["world"]["grid_resolution"]
        for mat in mats.values():
            if mat["mode"] != "soft_mpm":
                continue
            e = mat["E_MPa"] * 1e6
            nu = mat["nu"]
            rho = mat["density_kg_m3"]
            wave = (e * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho)) ** 0.5
            cfg["world"]["dt"] = min(cfg["world"]["dt"], 0.2 * dx / wave)
        # Template spacing and poses are already registered to the USD/board.
        spacing = record["spacing_m"]
        cfg["cutting_mesh"]["particle_spacing"] = spacing
        proxy = dict(voxel_size=spacing, T_asset_to_world=np.eye(4))
        cfg["cutting_mesh_pack"] = proxy
        blade = pack("knife")
        board = pack("board")
        ys = np.flatnonzero((board["sdf"] < 0).any(axis=(0, 2)))
        board["render_offset_y"] = (
            1.15 * dx - board["origin"][1] - (ys.min() + 0.5) * float(board["voxel_size"])
        )
        sim = self.sim = core.MPMCuttingSim(cfg, None, blade, board, viewer=False)
        self.seeds = None
        with np.load(folder / "mpm.npz") as data:
            points = data["x"] @ TO_WORLD
            ids = data["tissue"]
            vol = data["volume_m3"]
            self.tissue_ids = ids
            self.segment_ids = data["segment"]
            self.rest_usd = data["x"]
            properties = np.zeros((len(names) + 1, 3))
            for i, k in enumerate(names, 1):
                mat = mats[k]
                if mat["mode"] == "soft_mpm":
                    properties[i] = [mat["density_kg_m3"], mat["E_MPa"] * 1.0e6, mat["nu"]]
            rho, e, nu = properties[ids].T
            if not (rho > 0).all():
                raise ValueError("Rigid material seeded into the soft domain")
            num = len(vol)
            sim.pcount[None] = num
            sim.particles = core.Particle.field(shape=(num,))
            sim.particles.x.from_numpy(points.astype("f4"))
            sim.particles.F.from_numpy(np.tile(np.eye(3, dtype="f4"), (num, 1, 1)))
            sim.particles.Jp.fill(1.0)
            for field, values in [
                (sim.particles.vol, vol),
                (sim.particles.mass, rho * vol),
                (sim.particles.mu0, e / (2 * (1 + nu))),
                (sim.particles.la0, e * nu / ((1 + nu) * (1 - 2 * nu) + 1e-4)),
            ]:
                field.from_numpy(values.astype("f4"))
            clouds = [data[f"body_{i}"] @ TO_WORLD for i in range(len(record["rigid_bodies"]))]
        self.knife_mesh = trimesh.load(cfg["knife"]["mesh_path"], force="mesh")
        self.knife_mesh.apply_transform(blade["T_asset_to_world"])
        edge = self.knife_mesh.vertices
        self.tip = edge[edge[:, 1] < edge[:, 1].min() + 0.001].mean(0) @ TO_WORLD.T
        # The blade is 5 mm thick. The bolster is the first widened part of this
        # baked mesh. Reserve an extra 10 mm on its blade side as part of the grip.
        tool_usd = edge @ TO_WORLD.T
        half_width = np.abs(tool_usd[:, 1] - self.tip[1])
        self.grip_guard_x = float(tool_usd[half_width > 0.003, 0].min() - 0.010)
        initial = self.rest_usd.mean(0).astype(float)
        initial[2] = self.rest_usd[:, 2].max() + 0.015
        initial = self.vertical_cut_pose(initial)
        dynamic = []
        masses = []
        self.pit = None
        for item, vertices in zip(record["rigid_bodies"], clouds):
            if item["mode"] == "fixed_rigid":
                hull = trimesh.convex.convex_hull(vertices)
                self.pit = RigidPit(
                    sim,
                    {"pit_vertices": hull.vertices.copy(), "pit_faces": hull.faces.copy()},
                    0.0,
                    TO_WORLD.T @ initial,
                )
                sim.pit = self.pit
            else:
                dynamic.append(vertices)
                masses.append(item["volume_m3"] * mats[item["material"]]["density_kg_m3"])
        if fracture:
            sim.fracture = SweptFracture(sim)
        self.quat = np.array([2**-0.5, 2**-0.5, 0.0, 0.0])
        self.control = RigidKnifeControl(sim, dict(position=initial, quaternion=self.quat))
        if dynamic:
            self.seeds = MovingSeeds(sim, dynamic, masses, self.knife_mesh, spacing)
        self.initial_seed_poses = self.seeds.poses() if self.seeds else None
        self.tool_contact = 0.0
        self.frame = 0

    def vertical_cut_pose(self, edge_point):
        """Put the usable edge across the fruit and the entire grip beyond its +X side."""
        position = np.asarray(edge_point, dtype=float) - self.tip
        position[0] = self.rest_usd[:, 0].max() + 0.050 - self.grip_guard_x
        return position

    def command(self, position, velocity=(0.0, 0.0, 0.0), quaternion=None, angular=(0.0, 0.0, 0.0)):
        self.control.command(
            np.asarray(position, float),
            self.quat if quaternion is None else quaternion,
            np.asarray(velocity, float),
            np.asarray(angular, float),
        )

    def step(self, ticks=1):
        for _ in range(ticks):
            self.control.step()
            if self.pit:
                self.pit.exclude_pulp()
            if self.seeds:
                self.seeds.step(self.control)
            self.tool_contact += np.linalg.norm(self.sim.knife_impulse_g[None])
            self.frame += 1

    def state(self):
        """MPM positions/velocities and rest-to-current transforms for each USD rigid prim."""
        rigid = {}
        if self.seeds:
            p, r = self.seeds.poses()
            p0, r0 = self.initial_seed_poses
            delta = r @ r0.transpose(0, 2, 1)
            paths = [b["path"] for b in self.record["rigid_bodies"] if b["mode"] == "dynamic_rigid"]
            for path, center, initial, rotation in zip(paths, p, p0, delta):
                transform = np.eye(4)
                transform[:3, :3] = TO_WORLD @ rotation @ TO_WORLD.T
                transform[:3, 3] = TO_WORLD @ (center - rotation @ initial)
                rigid[path] = transform
        for b in self.record["rigid_bodies"]:
            if b["mode"] == "fixed_rigid":
                rigid[b["path"]] = np.eye(4)
        return dict(
            time_s=self.sim.sim_time,
            positions=self.sim.particles.x.to_numpy() @ TO_WORLD.T,
            velocities=self.sim.particles.v.to_numpy() @ TO_WORLD.T,
            tissue=self.tissue_ids,
            segment=self.segment_ids,
            rigid_rest_to_world=rigid,
            knife=self.control.state(),
        )

    def summary(self):
        x = self.sim.particles.x.to_numpy() @ TO_WORLD.T
        if not np.isfinite(x).all():
            raise FloatingPointError("Nonfinite MPM state")
        result = dict(
            fruit=self.record["fruit"],
            ticks=self.frame,
            time_s=self.sim.sim_time,
            dt_s=self.sim.dt,
            soft_particles=len(x),
            maximum_soft_displacement_m=float(np.linalg.norm(x - self.rest_usd, axis=1).max()),
            soft_knife_accumulated_impulse_Ns=self.tool_contact,
        )
        if self.pit:
            result.update(self.pit.state())
        if self.seeds:
            p, r = self.seeds.poses()
            result.update(
                seed_count=len(p),
                seed_centers_usd_m=(p @ TO_WORLD.T).tolist(),
                max_seed_displacement_m=float(np.linalg.norm(p - self.seeds.rest_centers, axis=1).max()),
                accumulated_soft_seed_impulse_Ns=float(self.seeds.soft_impulse.sum()),
                knife_seed_contact_points_over_ticks=self.seeds.knife_contacts,
            )
        if hasattr(self.sim, "fracture"):
            side = self.sim.fracture.side.to_numpy()
            result["cut_side_particles"] = [int((side == i).sum()) for i in (-1, 1)]
        return result

    def close(self):
        if self.seeds:
            self.seeds.close()
        self.temp.cleanup()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample", type=Path)
    ap.add_argument(
        "--ticks", type=int, help="Stop after this many ticks. Default: cut until the board or a fixed pit."
    )
    ap.add_argument("--speed", type=float, default=0.5, help="downward speed in metres per second")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--stage", type=Path, help="Save up to 121 sampled states as an Isaac-viewable .usdc stage")
    args = ap.parse_args()
    if (args.ticks is not None and args.ticks < 1) or not np.isfinite(args.speed) or args.speed < 0:
        ap.error("Positive ticks and finite nonnegative speed required")
    if args.ticks is None and args.speed == 0:
        ap.error("Specify --ticks for a stationary knife")
    if args.stage and args.stage.suffix != ".usdc":
        ap.error("--stage must end in .usdc")
    env = FruitSimulation(args.sample)
    try:
        pos = env.control.position.copy()
        pos[2] = env.rest_usd[:, 2].max() - env.tip[2] + 0.0003
        env.command(pos, [0.0, 0.0, -args.speed])
        board_top = float(env.sim.board_top_y[None])
        start_tip_z = pos[2] + env.tip[2]
        limit = args.ticks
        if args.speed > 0:
            board_ticks = max(0, int((start_tip_z - board_top) / (args.speed * env.sim.dt)))
            limit = board_ticks if limit is None else min(limit, board_ticks)
        view = None
        if args.stage:
            from pxr import Usd
            from sample_view import SampleView

            tissue_spec = ROOT / "assets/tissues" / (env.record["fruit"] + ".json")
            rows = json.loads(tissue_spec.read_text())["materials"]
            palette = {row["name"]: ((np.array(row["srgb"])/255.)**2.2).tolist() for row in rows}
            palette["GreenRind"] = [.045, .17, .025]
            view = SampleView(Usd.Stage.CreateInMemory(), env, args.sample, palette)
            view.update(env.state())
            interval = max(1, (limit + 119) // 120)
        for tick in range(1, limit + 1):
            env.step()
            blocked = bool(env.pit and env.pit.blocked)
            if view and (tick % interval == 0 or tick == limit or blocked):
                view.update(env.state())
            if blocked:
                break
        if view:
            args.stage.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=args.stage.parent) as tmp:
                pending = Path(tmp) / args.stage.name
                if not view.stage.GetRootLayer().Export(str(pending)):
                    raise OSError(f"Could not write {args.stage}")
                pending.replace(args.stage)
        result = env.summary()
        result["blade_travel_m"] = start_tip_z - (env.control.position[2] + env.tip[2])
        report = json.dumps(result, indent=2, allow_nan=False)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report + "\n")
        print(report)
    finally:
        env.close()
