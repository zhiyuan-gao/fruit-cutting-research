"""Pinned convex pit, soft-particle contact and continuous rigid knife contact in native MPM axes."""
import fcl
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import taichi as ti
import trimesh


def native_pit(volume, pack, lift=0.):
    vertices = trimesh.transform_points(volume["pit_vertices"], pack["T_asset_to_world"])
    vertices[:, 1] += lift
    return vertices


def carve_pit(pack, volume):
    """Subtract the convex solid before native particle seeding; keep the source outer SDF intact."""
    vertices = native_pit(volume, pack)
    pitch = float(pack["voxel_size"])
    origin = pack["origin"]
    sdf = pack["sdf"].copy()
    lo = np.maximum(0, np.floor((vertices.min(0)-origin)/pitch).astype(int)-3)
    hi = np.minimum(sdf.T.shape, np.ceil((vertices.max(0)-origin)/pitch).astype(int)+4)
    x, y, z = np.ix_(*[origin[i]+np.arange(lo[i],hi[i])*pitch for i in range(3)])
    # Max of outward face distances is exact inside this convex hull and has the
    # correct sign outside. Only the narrow band around the cavity is needed.
    field = np.full(hi-lo, -np.inf)
    for a, b, c, d in ConvexHull(vertices).equations:
        np.maximum(field, a*x+b*y+c*z+d, out=field)
    view = sdf.T[tuple(slice(i,j) for i,j in zip(lo,hi))]
    np.maximum(view, -field, out=view)
    return dict(pack, sdf=sdf)


def collision_mesh(vertices, faces):
    mesh = fcl.BVHModel()
    mesh.beginModel(num_tris_=len(faces), num_vertices_=len(vertices))
    mesh.addSubModel(np.ascontiguousarray(vertices, dtype=float), np.ascontiguousarray(faces, dtype=np.int32))
    mesh.endModel()
    return fcl.CollisionObject(mesh)


@ti.data_oriented
class RigidPit:
    clearance = 0.00003  # geometric contact skin in m; well below one particle pitch

    def __init__(self, sim, volume, lift, position):
        self.sim = sim
        vertices = native_pit(volume, sim.cfg["cutting_mesh_pack"], lift)
        self.vertices = vertices
        self.faces = volume["pit_faces"]
        self.solid = collision_mesh(vertices, self.faces)
        planes = ConvexHull(vertices).equations.astype(np.float32)
        self.planes = ti.Vector.field(4, ti.f32, shape=len(planes))
        self.planes.from_numpy(planes)
        self.lo = ti.Vector(vertices.min(axis=0).tolist())
        self.hi = ti.Vector(vertices.max(axis=0).tolist())
        shape = trimesh.load(sim.cfg["knife"]["mesh_path"], force="mesh")
        points = trimesh.transform_points(shape.vertices, sim.knife_pack["T_asset_to_world"])
        self.knife_mesh = collision_mesh(points, shape.faces)
        self.radius = np.linalg.norm(points,axis=1).max()
        self.p = position.copy()
        self.r = np.eye(3)
        self.gap = self.distance(self.p,self.r)
        if self.gap <= self.clearance or shape.contains(
                trimesh.transform_points(vertices[:1] - self.p, np.linalg.inv(sim.knife_pack["T_asset_to_world"]))).any():
            raise ValueError("Initial knife and rigid pit must be disjoint")
        self.blocked = False
        self.ever_blocked = False
        self.minimum_gap = self.gap
        self._last_trial = None

    def distance(self, position, rotation):
        self.knife_mesh.setTransform(fcl.Transform(rotation, position))
        return max(0., fcl.distance(self.knife_mesh, self.solid))

    def limit(self, target, rotation):
        """Bound swept motion by translation plus the longest possible vertex arc.

        For two samples a,b, the mesh distance anywhere between them is at least
        (d(a)+d(b)-motion_bound)/2. Subdivide unresolved segments in time order.
        """
        if self._last_trial is not None and self.blocked:
            p, r = self._last_trial
            if np.array_equal(target, p) and np.array_equal(rotation, r):
                return self.p, self.r
        if np.array_equal(target,self.p) and np.array_equal(rotation,self.r):
            return self.p, self.r
        initial_p = self.p.copy()
        initial_r = self.r
        translation = target - initial_p
        rotvec = (Rotation.from_matrix(rotation @ initial_r.T).as_rotvec()
                  if not np.array_equal(rotation,initial_r) else np.zeros(3))
        angle = np.linalg.norm(rotvec)
        length = np.linalg.norm(translation) + self.radius*angle
        def pose(t):
            return initial_p+t*translation, (Rotation.from_rotvec(t*rotvec).as_matrix() @ initial_r if angle else initial_r)
        def advance(a, da, b, db):
            span = (b-a)*length
            if db >= self.clearance and da+db > span+2.e-8:
                return b, db
            if span <= 0.5e-6:
                return a, da
            mid = (a+b)/2
            dm = self.distance(*pose(mid))
            reached, gap = advance(a, da, mid, dm)
            if reached < mid:
                return reached, gap
            return advance(mid, dm, b, db)
        end_gap = self.distance(target,rotation)
        t, self.gap = advance(0., self.gap, 1., end_gap)
        self.p, self.r = (target.copy(), rotation.copy()) if t==1 else pose(t)
        self.minimum_gap = min(self.minimum_gap,self.gap)
        self.blocked = t < 1
        self.ever_blocked |= self.blocked
        self._last_trial = (target.copy(),rotation.copy()) if self.blocked else None
        return self.p, self.r

    @ti.kernel
    def exclude_pulp(self):
        for i in self.sim.particles:
            p = self.sim.particles[i].x
            if (p >= self.lo).all() and (p <= self.hi).all():
                best = -1.e5
                n = ti.Vector([0.,0.,0.])
                for j in range(self.planes.shape[0]):
                    eq = self.planes[j]
                    normal = ti.Vector([eq[0],eq[1],eq[2]])
                    d = normal.dot(p)+eq[3]
                    if d > best:
                        best = d
                        n = normal
                    if d > 0.:
                        break
                if best <= 0.:
                    self.sim.particles[i].x = p + (1.e-6-best)*n
                    v = self.sim.particles[i].v
                    self.sim.particles[i].v = v - ti.min(v.dot(n),0.)*n

    def state(self):
        return dict(pit_contact=bool(self.blocked), pit_contact_seen=bool(self.ever_blocked),
                    pit_clearance_m=float(self.gap), minimum_pit_clearance_m=float(self.minimum_gap))
