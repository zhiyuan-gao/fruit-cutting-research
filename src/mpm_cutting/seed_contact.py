"""Two-way normal/friction impulses between soft MPM particles and Bullet convex seeds."""

import numpy as np
import taichi as ti
from scipy.spatial import ConvexHull


@ti.data_oriented
class SeedContact:
    def __init__(self, sim, local_clouds, masses, spacing):
        self.sim = sim
        self.count = len(masses)
        self.skin = 0.45 * spacing
        self.origin = np.asarray(sim._bmin_py, dtype=np.float32)
        self.cell = max(0.012, max(np.linalg.norm(p, axis=1).max() for p in local_clouds) * 2)
        self.grid_size = tuple(np.ceil((sim._bmax_py - self.origin) / self.cell).astype(int) + 2)
        self.groups = (self.count + 31) // 32
        self.mask = ti.field(ti.u32, shape=(*self.grid_size, self.groups))
        self.pos = ti.Vector.field(3, ti.f32, shape=self.count)
        self.rot = ti.Matrix.field(3, 3, ti.f32, shape=self.count)
        self.vel = ti.Vector.field(3, ti.f32, shape=self.count)
        self.omega = ti.Vector.field(3, ti.f32, shape=self.count)
        self.mass = ti.field(ti.f32, shape=self.count)
        self.mass.from_numpy(np.asarray(masses, np.float32))
        self.radius = ti.field(ti.f32, shape=self.count)
        self.radius.from_numpy(
            np.asarray([np.linalg.norm(p, axis=1).max() for p in local_clouds], np.float32)
        )
        planes = [np.unique(ConvexHull(p).equations.round(7), axis=0) for p in local_clouds]
        self.offset = ti.field(ti.i32, shape=self.count + 1)
        self.offset.from_numpy(np.int32(np.cumsum([0] + list(map(len, planes)))))
        self.plane = ti.Vector.field(4, ti.f32, shape=sum(map(len, planes)))
        self.plane.from_numpy(np.concatenate(planes).astype(np.float32))
        self.impulse = ti.Vector.field(3, ti.f32, shape=self.count)
        self.spin = ti.Vector.field(3, ti.f32, shape=self.count)
        self.activity = ti.field(ti.f32, shape=self.count)

    @ti.func
    def cell_id(self, p):
        cell = ti.cast(
            ti.floor((p - ti.Vector([self.origin[0], self.origin[1], self.origin[2]])) / self.cell), ti.i32
        )
        return ti.max(
            0, ti.min(cell, ti.Vector([self.grid_size[0] - 1, self.grid_size[1] - 1, self.grid_size[2] - 1]))
        )

    @ti.kernel
    def bins(self):
        for b in self.pos:
            self.impulse[b] = ti.Vector.zero(ti.f32, 3)
            self.spin[b] = ti.Vector.zero(ti.f32, 3)
            self.activity[b] = 0.0
            r = self.radius[b] + self.skin
            lo = self.cell_id(self.pos[b] - r)
            hi = self.cell_id(self.pos[b] + r)
            for i, j, k in ti.ndrange((lo[0], hi[0] + 1), (lo[1], hi[1] + 1), (lo[2], hi[2] + 1)):
                ti.atomic_or(self.mask[i, j, k, b // 32], ti.u32(1) << (b % 32))

    @ti.kernel
    def transfer(self):
        for i in self.sim.particles:
            p = self.sim.particles[i].x
            cell = self.cell_id(p)
            for group in ti.static(range(self.groups)):
                bits = self.mask[cell[0], cell[1], cell[2], group]
                offset = -1
                while bits != 0:
                    offset += 1
                    active = bits & ti.u32(1)
                    bits = bits >> 1
                    if active == 0:
                        continue
                    b = group * 32 + offset
                    arm = p - self.pos[b]
                    if arm.norm_sqr() < (self.radius[b] + self.skin) ** 2:
                        local = self.rot[b].transpose() @ arm
                        best = -1.0e5
                        normal = ti.Vector.zero(ti.f32, 3)
                        for j in range(self.offset[b], self.offset[b + 1]):
                            eq = self.plane[j]
                            n = ti.Vector([eq[0], eq[1], eq[2]])
                            d = n.dot(local) + eq[3]
                            if d > best:
                                best = d
                                normal = n
                            if best >= self.skin:
                                break
                        if best < self.skin:
                            n = self.rot[b] @ normal
                            v = self.sim.particles[i].v - self.vel[b] - self.omega[b].cross(arm)
                            vn = v.dot(n)
                            tangent = v - vn * n
                            m = self.sim.particles[i].mass
                            reduced = m * self.mass[b] / (m + self.mass[b])
                            # Damped unilateral penalty. No tensile impulse at a separating interface.
                            freq = 2500.0
                            jn = ti.max(
                                0.0, reduced * self.sim.dt * (freq**2 * (self.skin - best) - 2 * freq * vn)
                            )
                            jv = jn * n - tangent * ti.min(reduced, 0.25 * jn / (1.0e-12 + tangent.norm()))
                            self.sim.particles[i].v += jv / m
                            self.impulse[b] -= jv
                            self.spin[b] -= arm.cross(jv)
                            self.activity[b] += jv.norm()

    def exchange(self, positions, rotations, velocities, omegas):
        for field, data in (
            (self.pos, positions),
            (self.rot, rotations),
            (self.vel, velocities),
            (self.omega, omegas),
        ):
            field.from_numpy(np.ascontiguousarray(data, np.float32))
        self.mask.fill(0)
        self.bins()
        self.transfer()
        return self.impulse.to_numpy(), self.spin.to_numpy(), self.activity.to_numpy()
