"""Externally prescribed rigid motion for the upstream blade in the MPM frame."""
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from mpmcore.colliders import KnifeCollider

from adapter import TO_WORLD


@ti.data_oriented
class RigidKnifeCollider(KnifeCollider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.translation = ti.Vector.field(3, ti.f32, shape=())
        self.orientation = ti.Matrix.field(3, 3, ti.f32, shape=())
        self.linear = ti.Vector.field(3, ti.f32, shape=())
        self.angular = ti.Vector.field(3, ti.f32, shape=())

    @ti.func
    def local_point(self, p):
        return self.orientation[None].transpose() @ (p - self.translation[None])

    @ti.func
    def sample(self, p):
        return self._sample_grid(self.sdf, self.local_point(p), self.origin[None][1], 0.0)

    @ti.func
    def sample_mask(self, p):
        mask = 1.0
        if ti.static(self.mask != None):
            mask = self._sample_grid(self.mask, self.local_point(p), self.origin[None][1], 0.0)
        return mask

    @ti.func
    def velocity_at(self, p):
        return self.linear[None] + self.angular[None].cross(p - self.translation[None])

    @ti.kernel
    def set_boundary(self, p: ti.math.vec3, r: ti.math.mat3, v: ti.math.vec3,
                     w: ti.math.vec3, down: ti.i32):
        self.translation[None] = p
        self.orientation[None] = r
        self.linear[None] = v
        self.angular[None] = w
        self._down[None] = down


@ti.data_oriented
class RegisteredBoardKnifeCollider(RigidKnifeCollider):
    @ti.func
    def _board_sample(self, p):
        # The board has its own XYZ origin and voxel size, independent of the knife pack.
        return self._board.sample(p)


@ti.data_oriented
class RigidKnifeControl:
    def __init__(self, sim, initial):
        self.sim = sim
        self.fracture = getattr(sim, "fracture", None)
        self.pit = getattr(sim, "pit", None)
        self.force_field = None
        self.loading = False
        indices = sim._knife_neg_idx[:, ::-1]
        voxel = sim._knife_voxel
        lo = sim._knife_origin + (indices.min(axis=0) - 1) * voxel
        hi = sim._knife_origin + (indices.max(axis=0) + 1) * voxel
        self.box_center = (lo + hi) / 2
        self.box_half_size = (hi - lo) / 2
        self.margin = 2 * float(sim.dx_s[None])
        self.command(initial["position"], initial["quaternion"], np.zeros(3), np.zeros(3))

    def command(self, position, quaternion, linear, angular):
        self.t0 = self.sim.sim_time
        self.start = np.array(position)
        self.start_rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        self.linear = np.array(linear)
        self.angular = np.array(angular)
        self.spin = np.any(self.angular)
        self.update_boundary(self.t0)
        if self.pit is not None:
            self.start = self.position.copy()
            self.start_rotation = self.rotation

    def update_boundary(self, time):
        elapsed = time - self.t0
        self.position = self.start + elapsed * self.linear
        self.rotation = self.start_rotation
        if self.spin:
            self.rotation = Rotation.from_rotvec(elapsed * self.angular) * self.start_rotation
        r = TO_WORLD.T @ self.rotation.as_matrix()
        p = TO_WORLD.T @ self.position
        sim = self.sim
        if self.pit is not None:
            p, r = self.pit.limit(p, r)
            self.position = TO_WORLD @ p
            self.rotation = Rotation.from_matrix(TO_WORLD @ r)
        stopped = self.pit is not None and self.pit.blocked
        v = np.zeros(3) if stopped else TO_WORLD.T @ self.linear
        w = np.zeros(3) if stopped else TO_WORLD.T @ self.angular
        sim.knife.set_boundary(p, r, v, w, int(not stopped and self.linear[2] < 0))
        # Cull against the rotated solid, before evaluating the same-step contacts.
        center = p + r @ self.box_center
        extent = np.abs(r) @ self.box_half_size + self.margin
        sim.kmin_f[None], sim.kmax_f[None] = center - extent, center + extent

    def step(self):
        sim = self.sim
        tracing = self.fracture is not None and self.linear[2] < 0
        if tracing:
            old_p = TO_WORLD.T @ self.position
            old_r = TO_WORLD.T @ self.rotation.as_matrix()
        if self.pit is None:
            self.update_boundary(sim.sim_time + sim.dt)
        else:
            self.start = self.position.copy()
            self.start_rotation = self.rotation
            self.t0 = sim.sim_time
            # Preserve the actual angular sweep even for commands exceeding half a turn per tick.
            steps = max(1, int(np.ceil(np.linalg.norm(self.angular)*sim.dt/np.pi)))
            for i in range(1, steps+1):
                self.update_boundary(sim.sim_time + sim.dt*i/steps)
                if self.pit.blocked:
                    break
        if tracing:
            self.fracture.find_engaged_edge()
            self.fracture.record_sweep(old_p, old_r)
        if self.loading:
            self.apply_forces()
        sim._reset_contact_accumulators()
        sim.knife.contact_accum_f[None] = 0.
        sim.p2g()
        sim.g2p()
        sim.knife_board_collision_detected[None] = 0
        sim.knife_mesh_collision_detected[None] = 0
        sim.sim_time += sim.dt

    def set_forces(self, forces):
        self.loading = forces is not None
        if self.loading:
            if self.force_field is None:
                self.force_field = ti.Vector.field(3, ti.f32, shape=int(self.sim.pcount[None]))
            self.force_field.from_numpy(np.ascontiguousarray(forces @ TO_WORLD))

    @ti.kernel
    def apply_forces(self):
        for p in self.force_field:
            self.sim.particles[p].v += self.sim.dt * self.force_field[p] / self.sim.particles[p].mass

    def state(self):
        stopped = self.pit is not None and self.pit.blocked
        return dict(time=self.sim.sim_time, position=self.position,
                    quaternion=self.rotation.as_quat()[[3, 0, 1, 2]],
                    linear_velocity=np.zeros(3) if stopped else self.linear,
                    angular_velocity=np.zeros(3) if stopped else self.angular,
                    cut_index=None, cutting_complete=None)
