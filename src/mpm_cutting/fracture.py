"""Single advancing crack, recorded on material particles when the blade sweeps them."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import taichi as ti

from adapter import TO_WORLD, UPSTREAM


def load_fracture_core(directory):
    """Apply the small grid-transfer diff to a private copy of the pinned source."""
    target = Path(directory) / "fracture_mpm.py"
    patch = Path(__file__).with_name("fracture_transfers.patch")
    subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(target),
                    str(UPSTREAM / "mpmcore/sim.py"), "-i", str(patch)], check=True)
    spec = importlib.util.spec_from_file_location("fracture_mpm", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@ti.data_oriented
class SweptFracture:
    def __init__(self, sim):
        self.sim = sim
        pack = sim.knife_pack
        sdf = pack["sdf"]
        solid = (sdf < 0) & (np.asarray(pack.get("blade_mask", 1)) > .5)
        self.voxel = float(pack["voxel_size"])
        origin = np.asarray(pack["origin"])
        self.x0 = float(origin[0])
        self.columns = sdf.shape[2]
        edge = np.zeros((self.columns, 3), np.float32)
        valid = np.zeros(self.columns, np.int32)
        for i in range(self.columns):
            z, y = np.where(solid[:, :, i])
            if y.size:
                bottom = y.min()
                edge[i] = origin + self.voxel * np.array([i, bottom, z[y == bottom].mean()])
                valid[i] = 1
        self.edge = ti.Vector.field(3, ti.f32, shape=self.columns)
        self.edge.from_numpy(edge)
        self.valid = ti.field(ti.i32, shape=self.columns)
        self.valid.from_numpy(valid)
        self.engaged = ti.field(ti.i32, shape=self.columns)
        count = int(sim.pcount[None])
        self.side = ti.field(ti.i32, shape=count)
        self.reference_normal = ti.Vector.field(3, ti.f32, shape=count)
        self.reach = 4.5 * float(sim.dx_s[None])
        seed_voxel = float(sim.cfg["cutting_mesh_pack"]["voxel_size"])
        requested = float(sim.cfg["cutting_mesh"].get("particle_spacing", .0033))
        self.spacing = max(1, round(requested / seed_voxel)) * seed_voxel
        self.friction = float(sim.cfg.get("fracture", {}).get("face_friction", .2))
        shape = (sim.n_grid,) * 3
        self.mass = ti.field(ti.f32, shape=shape)
        self.velocity = ti.Vector.field(3, ti.f32, shape=shape)
        self.normal_sum = ti.Vector.field(3, ti.f32, shape=shape)
        self.uncut_mass = ti.field(ti.f32, shape=shape)
        self.plus_min = ti.field(ti.f32, shape=shape)
        self.minus_max = ti.field(ti.f32, shape=shape)
        self.active_nodes = ti.field(ti.i32, shape=())
        self.contact_nodes = ti.field(ti.i32, shape=())

    @ti.func
    def edge_column(self, p):
        x = (self.sim.knife.local_point(p)[0] - self.x0) / self.voxel
        i = ti.max(0, ti.min(self.columns-2, ti.cast(ti.floor(x), ti.i32)))
        valid = x >= 0 and x < self.columns-1 and self.valid[i] and self.valid[i+1]
        return i, x-i, valid

    @ti.kernel
    def find_engaged_edge(self):
        for i in self.engaged:
            self.engaged[i] = 0
        for p in self.side:
            x = self.sim.particles[p].x
            i, f, valid = self.edge_column(x)
            if valid:
                local = self.sim.knife.local_point(x)
                edge = (1-f)*self.edge[i] + f*self.edge[i+1]
                if (local-edge).norm() < 1.5*self.voxel and self.sim.knife.is_blade(x):
                    if self.sim.knife.sample(x) < self.voxel:
                        ti.atomic_max(self.engaged[i], 1)

    @ti.kernel
    def record_sweep(self, previous_p: ti.math.vec3, previous_r: ti.math.mat3):
        k = self.sim.knife
        for p in self.side:
            if self.side[p] == 0:
                part = self.sim.particles[p]
                i, f, valid = self.edge_column(part.x)
                if valid and self.engaged[i]:
                    a, b = self.edge[i], self.edge[i+1]
                    local = (1-f)*a + f*b
                    q = k.orientation[None] @ local + k.translation[None]
                    prev = previous_r @ local + previous_p
                    tangent = (k.orientation[None] @ (b-a)).normalized()
                    advance = q-prev
                    advance -= tangent * advance.dot(tangent)
                    length = advance.norm()
                    if length > 1e-9:
                        along = advance/length
                        normal = tangent.cross(along)
                        if normal.dot(k.orientation[None] @ ti.Vector([0., 0., 1.])) < 0:
                            normal = -normal
                        relative = part.x - q
                        distance, front = relative.dot(normal), relative.dot(along)
                        # Small front tolerance accommodates sampling of the discrete SDF edge.
                        if ti.abs(distance) <= self.reach and -length-.25*self.voxel <= front <= .25*self.voxel:
                            self.side[p] = 1 if distance >= 0 else -1
                            self.reference_normal[p] = part.F.transpose() @ normal

    @ti.func
    def clear_node(self, I):
        self.mass[I] = 0.
        self.velocity[I] = ti.Vector.zero(ti.f32, 3)
        self.normal_sum[I] = ti.Vector.zero(ti.f32, 3)
        self.uncut_mass[I] = 0.
        self.plus_min[I] = 1.e8
        self.minus_max[I] = -1.e8

    @ti.func
    def scatter(self, p, I, mass, momentum, normal, node_offset):
        if self.side[p] > 0:
            self.mass[I] += mass
            self.velocity[I] += momentum
            ti.atomic_min(self.plus_min[I], node_offset.dot(normal))
        else:
            self.sim.grid_m[I] += mass
            self.sim.grid_v[I] += momentum
            if self.side[p] == 0:
                self.uncut_mass[I] += mass
            else:
                ti.atomic_max(self.minus_max[I], node_offset.dot(normal))
        self.normal_sum[I] += mass * normal

    @ti.func
    def couple(self, I):
        # Before external contact response, impose a unilateral impulse on approaching faces.
        a, b = self.sim.grid_m[I], self.mass[I]
        if a > 0 and b > 0:
            va, vb = self.sim.grid_v[I]/a, self.velocity[I]/b
            if self.uncut_mass[I] > 1.e-6*(a+b):
                self.sim.grid_v[I] += self.velocity[I]
                self.sim.grid_m[I] = a+b
                self.mass[I] = 0.
            else:
                ti.atomic_add(self.active_nodes[None], 1)
                normal = self.normal_sum[I].normalized(1.e-16)
                rel = vb-va
                vn = rel.dot(normal)
                gap = self.plus_min[I]-self.minus_max[I]-self.spacing
                if vn < 0 and gap + vn*self.sim.dt <= 0:
                    ti.atomic_add(self.contact_nodes[None], 1)
                    tangent = rel-vn*normal
                    delta = -vn*normal - tangent*ti.min(1., self.friction*(-vn)/(tangent.norm()+1.e-12))
                    impulse = (a*b/(a+b))*delta
                    self.sim.grid_v[I] -= impulse
                    self.velocity[I] += impulse

    def state(self):
        return dict(side=self.side.to_numpy(), reference_normals=self.reference_normal.to_numpy() @ TO_WORLD.T,
                    split_nodes=int(self.active_nodes[None]), compressive_nodes=int(self.contact_nodes[None]),
                    spacing_m=self.spacing)
