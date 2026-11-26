"""Matched homogeneous coupon with the pinned CulinaryCut MLS solver and six-axis telemetry."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import taichi as ti
import warp as wp
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src/mpm_cutting"),
    str(ROOT / "third_party/EEF-Cutting-Simulation"),
]
from adapter import TO_WORLD  # noqa: E402
from cut_physics_trial import block, protocol  # noqa: E402
from reference_cut import knife_geometry, crossed_into_mesh, inside_mesh  # noqa: E402
from rigid_knife import RigidKnifeCollider  # noqa: E402


@ti.data_oriented
class NativeProbe:
    def __init__(self, sim, n, clock, base_translation, wrist, anchor):
        self.sim = sim
        self.i = ti.field(ti.i32, shape=())
        self.pose = ti.Vector.field(2, ti.f32, shape=n)
        self.pose.from_numpy(clock[:, :2].astype("f4").copy())
        self.base = ti.Vector(list(base_translation))
        self.wrist = ti.Vector(list(wrist))
        self.anchor = anchor
        self.J = ti.Vector.field(3, ti.f32, shape=n)
        self.L = ti.Vector.field(3, ti.f32, shape=n)
        self.shift = ti.field(ti.f32, shape=n)
        self.path = ti.Vector.field(3, ti.f32, shape=(257, int(sim.pcount[None])))

    @ti.kernel
    def begin(self, s: ti.i32):
        self.i[None] = s
        self.sim.knife.contact_accum_f[None] = 0.0
        h, v = self.pose[s][0], self.pose[s][1]
        self.sim.knife.translation[None] = self.base + ti.Vector(
            [0.0, h + v * self.sim.dt, 0.0]
        )
        self.sim.knife.linear[None] = ti.Vector([0.0, v, 0.0])
        self.sim.knife._down[None] = ti.cast(v < 0.0, ti.i32)

    @ti.func
    def contact(self, point, j):
        k = self.i[None]
        wrist = self.sim.knife.translation[None] + self.wrist
        self.J[k] += -j
        self.L[k] += (point - wrist).cross(-j)

    @ti.func
    def project(self, shift):
        ti.atomic_max(self.shift[self.i[None]], shift.norm())

    @ti.kernel
    def store(self, k: ti.i32):
        for i in self.sim.particles:
            self.path[k, i] = self.sim.particles[i].x


@wp.kernel
def audit(
    mesh: wp.uint64,
    x: wp.array2d(dtype=wp.vec3),
    height: wp.array(dtype=float),
    entries: wp.array(dtype=int),
    inside: wp.array(dtype=int),
):
    k, i = wp.tid()
    a = x[k, i] - wp.vec3(0.0, 0.0, height[k])
    b = x[k + 1, i] - wp.vec3(0.0, 0.0, height[k + 1])
    if crossed_into_mesh(mesh, a, b):
        wp.atomic_add(entries, k, 1)
    if inside_mesh(mesh, b):
        wp.atomic_add(inside, k, 1)


def run(case, speed, dt, out, tmp):
    import trimesh
    from blade_geometry import mesh_lower_edge

    upstream = ROOT / "third_party/EEF-Cutting-Simulation"
    patched = tmp / "native_coupon.py"
    subprocess.run(
        [
            "patch",
            "--batch",
            "--fuzz=0",
            "-o",
            str(patched),
            str(upstream / "mpmcore/sim.py"),
            "-i",
            str(Path(__file__).with_name("native_wrench.patch").resolve()),
        ],
        check=True,
    )
    spec = importlib.util.spec_from_file_location("native_coupon", patched)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    core.KnifeCollider = RigidKnifeCollider
    ti.init(arch=ti.cuda, enable_fallback=False, random_seed=2026)
    core._TI_INIT = True
    xyz = block(0.001)
    action, marks = protocol(case, speed, dt)
    material = json.loads(
        (ROOT / "examples/cucumber/2026_0000/sample.json").read_text()
    )["materials"]["Pulp"]
    cfg = yaml.safe_load((ROOT / "configs/simulation.yaml").read_text())
    cfg["world"].update(
        bounds_min=[-0.024] * 3,
        bounds_max=[0.024] * 3,
        grid_resolution=24,
        dt=dt,
        gravity=[0.0] * 3,
    )
    cfg["cutting_mesh"].update(
        density=material["density_kg_m3"], grid_damping_ratio=0.0, particle_damping=0.0
    )
    cfg["cutting_mesh"]["elasticity"].update(
        youngs_modulus=material["E_MPa"] * 1.0e6, poisson_ratio=material["nu"]
    )
    cfg["cutting"].update(tip_force_gain=0.0, tip_post_gain=0.0)
    cfg["plasticity"]["enabled"] = False
    cfg["knife"]["mesh_path"] = str(upstream / "assets/knife.obj")
    with np.load(ROOT / ".cache/tools/knife.npz") as bundle:
        pack = dict(bundle)
    cfg["cutting_mesh_pack"] = dict(voxel_size=0.001, T_asset_to_world=np.eye(4))
    sim = core.MPMCuttingSim(cfg, None, pack, None, viewer=False)
    if not np.isclose(sim.dt, dt, rtol=1.0e-7, atol=0.0):
        raise ValueError("CFL changed requested timing")
    sim.pcount[None] = len(xyz)
    sim.particles = core.Particle.field(shape=len(xyz))
    sim.particles.x.from_numpy((xyz @ TO_WORLD).astype("f4"))
    sim.particles.F.from_numpy(np.tile(np.eye(3, dtype="f4"), (len(xyz), 1, 1)))
    sim.particles.Jp.fill(1.0)
    sim.particles.mass.fill(material["density_kg_m3"] * 1.0e-9)
    sim.particles.vol.fill(1.0e-9)
    sim.particles.mu0.fill(sim.mu0)
    sim.particles.la0.fill(sim.la0)
    knife = trimesh.load(cfg["knife"]["mesh_path"], force="mesh")
    knife.apply_transform(pack["T_asset_to_world"])
    center, _ = mesh_lower_edge(knife, np.array([0.040]), 0.070)
    wrist_asset = center[0] + np.array([0.10, 0, 0.03]) @ TO_WORLD
    probe = sim.probe = NativeProbe(
        sim, len(action), action, -center[0], wrist_asset, 0.005
    )
    sim.knife.set_boundary(-center[0], np.eye(3), np.zeros(3), np.zeros(3), 1)
    sim.kmin_f[None] = [-1.0, -1.0, -1.0]
    sim.kmax_f[None] = [1.0, 1.0, 1.0]
    verts, faces, _ = knife_geometry(ROOT)
    mesh = wp.Mesh(
        wp.array(verts, dtype=wp.vec3, device="cuda"),
        wp.array(faces.ravel(), dtype=int, device="cuda"),
    )
    entries, inside = [], []
    z = np.r_[action[:, 0], action[-1, 0] + dt * action[-1, 1]]
    for start in range(0, len(action), 256):
        end = min(start + 256, len(action))
        probe.store(0)
        for j in range(start, end):
            sim._reset_contact_accumulators()
            probe.begin(j)
            sim.p2g()
            sim.g2p()
            probe.store(j - start + 1)
        paths = np.ascontiguousarray(
            probe.path.to_numpy()[: end - start + 1] @ TO_WORLD.T
        )
        if not np.isfinite(paths).all():
            raise FloatingPointError("Nonfinite native coupon state")
        count = end - start
        hits, vol = [wp.zeros(count, dtype=int, device="cuda") for _ in range(2)]
        wp.launch(
            audit,
            (count, len(xyz)),
            [
                mesh.id,
                wp.array(paths, dtype=wp.vec3, device="cuda"),
                wp.array(z[start : end + 1], dtype=float, device="cuda"),
                hits,
                vol,
            ],
            device="cuda",
        )
        entries.extend(hits.numpy())
        inside.extend(vol.numpy())
    j = probe.J.to_numpy() @ TO_WORLD.T
    moments = probe.L.to_numpy() @ TO_WORLD.T
    projection = probe.shift.to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "substeps.npz",
        force_impulse=j,
        moment_impulse=moments,
        action=action,
        crossings=entries,
        inside=inside,
        max_position_correction_m=projection,
    )
    rows = []
    boundaries = np.unique(
        np.r_[0, np.arange(0, len(j), round(0.001 / dt)), list(marks.values()), len(j)]
    )
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        length = (b - a) * dt
        rows.append(
            [
                b * dt,
                (0.006 - z[b]) * 1000,
                *(j[a:b].sum(0, dtype=float) / length),
                *(moments[a:b].sum(0, dtype=float) / length),
                sum(entries[a:b]),
                sum(inside[a:b]),
            ]
        )
    with (out / "wrench.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_s",
                "depth_mm",
                "Fx_N",
                "Fy_N",
                "Fz_N",
                "Mx_Nm",
                "My_Nm",
                "Mz_Nm",
                "entries",
                "inside_particle_steps",
            ]
        )
        w.writerows(rows)
    down = marks["down"]
    result = dict(
        case=case,
        backend="CulinaryCut MLS-MPM, fixed end grid nodes",
        particles=len(xyz),
        dt_s=dt,
        speed_m_s=speed,
        segment_entries=int(sum(entries)),
        inside_particle_steps=int(sum(inside)),
        max_inside_particles=int(max(inside)),
        peak_window_Fz_N=float(max(row[4] for row in rows if row[0] <= down * dt)),
        downstroke_work_mJ=float(
            -np.dot(j[:down, 2].astype(float), action[:down, 1]) * 1000
        ),
        max_position_correction_mm=float(projection.max() * 1000),
    )
    (out / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["air", "indent", "full"])
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        run(a.case, a.speed, 5.0e-6, a.out, Path(tmp))
