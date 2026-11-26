"""Single straight cut on a fixed reference grid. All vectors are in world SI, Z up."""

from pathlib import Path

import numpy as np
import trimesh
import warp as wp

from blade_geometry import mesh_lower_edge


@wp.struct
class Particles:
    last_surface: wp.array(dtype=wp.vec3)
    last_reaction: wp.array(dtype=wp.vec3)
    rest: wp.array(dtype=wp.vec3)
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    F: wp.array(dtype=wp.mat33)
    side: wp.array(dtype=int)
    work: wp.array(dtype=float)
    index: wp.array2d(dtype=int)
    weight: wp.array2d(dtype=float)
    gradient: wp.array2d(dtype=wp.vec3)
    volume: float
    mass: float
    mu: float
    lam: float
    radius: float


@wp.struct
class Grid:
    mass: wp.array(dtype=float)
    momentum: wp.array(dtype=wp.vec3)
    impulse: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    delta: wp.array(dtype=wp.vec3)
    anchor: wp.array(dtype=int)


@wp.struct
class Tool:
    mesh: wp.uint64
    edge: wp.array(dtype=wp.vec2)
    xmin: float
    edge_dx: float
    half_band: float
    wrist: wp.vec3
    friction: float
    keff: float


@wp.struct
class Protocol:
    # Start-of-substep tip height, vertical velocity, anchor enable, unilateral Y acceleration.
    action: wp.array(dtype=wp.vec4)
    step: wp.array(dtype=int)
    cut: int
    dt: float
    # PIC fraction. The final sec^-1-controlled blend is supplied by the caller.
    pic: float
    normal_scale: float


@wp.struct
class Trace:
    force_impulse: wp.array(dtype=wp.vec3)
    moment_impulse: wp.array(dtype=wp.vec3)
    tool_work: wp.array(dtype=float)
    friction_loss: wp.array(dtype=float)
    clamp_impulse: wp.array(dtype=wp.vec3)
    load_impulse: wp.array(dtype=wp.vec3)
    crossings: wp.array(dtype=int)
    inside: wp.array(dtype=int)
    pair_impulse: wp.array(dtype=float)
    energy: wp.array(dtype=wp.vec2)
    min_j: wp.array(dtype=float)


@wp.func
def identity():
    return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.func
def rotation(f: wp.mat33):
    u, s, v = wp.svd3(f)
    return u * wp.transpose(v)


@wp.func
def edge_at(t: Tool, x: float):
    column = (x - t.xmin) / t.edge_dx
    i = wp.clamp(int(wp.floor(column)), 0, t.edge.shape[0] - 2)
    return wp.lerp(t.edge[i], t.edge[i + 1], wp.clamp(column - float(i), 0.0, 1.0))


@wp.kernel
def clear_grid(g: Grid):
    i = wp.tid()
    g.mass[i] = 0.0
    g.momentum[i] = wp.vec3()
    g.impulse[i] = wp.vec3()


@wp.func
def friction_impulse(
    relative: wp.vec3, normal: wp.vec3, normal_j: float, mass: float, mu: float
):
    tangent = relative - wp.dot(relative, normal) * normal
    return -wp.min(mu * normal_j, mass * wp.length(tangent)) * wp.normalize(tangent)


@wp.kernel
def scatter(
    p: Particles, g: Grid, tool: Tool, run: Protocol, log: Trace, neighbors: wp.uint64
):
    i = wp.tid()
    step = run.step[0]
    a = run.action[step]
    x = p.x[i]
    v = p.v[i]
    local = x - wp.vec3(0.0, 0.0, a[0])
    knife_velocity = wp.vec3(0.0, 0.0, a[1])
    relative = v - knife_velocity
    dt = run.dt
    jtool = wp.vec3()
    p.last_surface[i] = wp.vec3()
    q = wp.mesh_query_point_sign_normal(
        tool.mesh, local, 3.0 * p.radius + dt * wp.length(relative)
    )
    if q.result:
        surface = wp.mesh_eval_position(tool.mesh, q.face, q.u, q.v)
        distance = wp.length(local - surface)
        n = wp.normalize(local - surface) * q.sign
        if distance < 1.0e-9:
            n = wp.mesh_eval_face_normal(tool.mesh, q.face)
        gap = distance * q.sign - p.radius
        predicted = gap + dt * wp.min(0.0, wp.dot(relative, n))
        jn = run.normal_scale * p.mass * wp.max(-predicted, 0.0) / dt
        jt = friction_impulse(relative, n, jn, p.mass, tool.friction)
        jtool = jn * n + jt
        e = edge_at(tool, local[0])
        advance_speed = relative[2]
        front = e[1] - local[2]
        if (
            run.cut == 1
            and a[1] < 0.0
            and advance_speed > 0.0
            and front >= 0.0
            and front < 2.0 * p.radius
            and wp.abs(local[1] - e[0]) < tool.half_band
        ):
            # Clip the last directional impulse to the remaining per-particle work budget.
            limit = tool.keff * wp.pow(p.volume, 2.0 / 3.0)
            forward_j = wp.max(0.0, -jtool[2])
            allowed = wp.min(forward_j, wp.max(0.0, limit - p.work[i]) / advance_speed)
            jtool[2] += forward_j - allowed
            p.work[i] += allowed * advance_speed
        if jn > 0.0:
            p.last_surface[i] = surface + wp.vec3(0.0, 0.0, a[0])
            wp.atomic_add(log.force_impulse, step, -jtool)
            wp.atomic_add(
                log.moment_impulse, step, wp.cross(surface - tool.wrist, -jtool)
            )
            wp.atomic_add(log.tool_work, step, wp.dot(jtool, knife_velocity))
            # Sliding work estimate before the directional gate. Logged only as a diagnostic.
            wp.atomic_add(log.friction_loss, step, -wp.dot(jt, relative))
    p.last_reaction[i] = -jtool
    jp = wp.vec3()
    if p.side[i] != 0:
        for k in wp.hash_grid_query(neighbors, x, 2.1 * p.radius):
            if p.side[k] != 0 and p.side[i] != p.side[k]:
                d = x - p.x[k]
                n = wp.normalize(d)
                relative_pair = v - p.v[k]
                gap = (
                    wp.length(d)
                    - 2.0 * p.radius
                    + dt * wp.min(0.0, wp.dot(relative_pair, n))
                )
                jn = run.normal_scale * 0.5 * p.mass * wp.max(-gap, 0.0) / dt
                jp += jn * n + friction_impulse(
                    relative_pair, n, jn, 0.5 * p.mass, 0.20
                )
                if i < k:
                    wp.atomic_add(log.pair_impulse, step, jn)
    load = wp.vec3()
    if p.rest[i][1] > 0.0:
        load = wp.vec3(0.0, p.mass * a[3] * dt, 0.0)
        wp.atomic_add(log.load_impulse, step, load)
    f = p.F[i]
    # Cofactor columns give J F^-T, also defined without inverting a near-singular matrix.
    ft = wp.transpose(f)
    cof = wp.matrix_from_cols(
        wp.cross(ft[1], ft[2]), wp.cross(ft[2], ft[0]), wp.cross(ft[0], ft[1])
    )
    stress = 2.0 * p.mu * (f - rotation(f)) + p.lam * (wp.determinant(f) - 1.0) * cof
    for b in range(27):
        node = 3 * p.index[i, b] + p.side[i]
        w = p.weight[i, b]
        wp.atomic_add(g.mass, node, w * p.mass)
        wp.atomic_add(g.momentum, node, w * p.mass * v)
        wp.atomic_add(
            g.impulse,
            node,
            w * (jtool + jp + load) - dt * p.volume * (stress * p.gradient[i, b]),
        )


@wp.kernel
def update_grid(g: Grid, run: Protocol, log: Trace):
    node = wp.tid()
    start = 3 * node
    total = g.mass[start] + g.mass[start + 1] + g.mass[start + 2]
    step = run.step[0]
    for c in range(3):
        i = start + c
        m = g.mass[i]
        if m > 0.0:
            mom = g.momentum[i]
            impulse = g.impulse[i]
            for b in range(3):
                if b != c and (b == 0 or c == 0):
                    k = start + b
                    mom += (m * g.momentum[k] - g.mass[k] * g.momentum[i]) / total
                    impulse += (m * g.impulse[k] - g.mass[k] * g.impulse[i]) / total
            v = (mom + impulse) / m
            dv = impulse / m
            if run.action[step][2] > 0.0 and g.anchor[node] != 0:
                wp.atomic_add(log.clamp_impulse, step, -m * v)
                dv -= v
                v = wp.vec3()
            g.vel[i] = v
            g.delta[i] = dv


@wp.func
def crossed_into_mesh(mesh: wp.uint64, start: wp.vec3, end: wp.vec3):
    motion = end - start
    length = wp.length(motion)
    result = int(0)
    if length > 1.0e-10:
        hit = wp.mesh_query_ray(mesh, start, motion / length, length)
        if hit.result and hit.sign > 0.0 and hit.t > 1.0e-9:
            result = 1
    return result


@wp.func
def inside_mesh(mesh: wp.uint64, point: wp.vec3):
    q = wp.mesh_query_point_sign_normal(mesh, point, 0.02)
    return int(q.result and q.sign < 0.0)


@wp.kernel
def gather(p: Particles, g: Grid, tool: Tool, run: Protocol, log: Trace):
    i = wp.tid()
    step = run.step[0]
    a = run.action[step]
    vpic = wp.vec3()
    delta = wp.vec3()
    gradient = wp.mat33()
    for b in range(27):
        node = 3 * p.index[i, b] + p.side[i]
        w = p.weight[i, b]
        vpic += w * g.vel[node]
        delta += w * g.delta[node]
        gradient += wp.outer(g.vel[node], p.gradient[i, b])
    # Update x and total F from the same reference-grid velocity field.
    p.F[i] += run.dt * gradient
    p.v[i] = wp.lerp(p.v[i] + delta, vpic, run.pic)
    old = p.x[i]
    x = old + run.dt * vpic
    p.x[i] = x
    local0 = old - wp.vec3(0.0, 0.0, a[0])
    local1 = x - wp.vec3(0.0, 0.0, a[0] + a[1] * run.dt)
    # Check the accepted, linearly translated path independently of the contact predictor.
    if crossed_into_mesh(tool.mesh, local0, local1):
        wp.atomic_add(log.crossings, step, 1)
    if inside_mesh(tool.mesh, local1):
        wp.atomic_add(log.inside, step, 1)
    if run.cut == 1 and a[1] < 0.0 and p.side[i] == 0:
        e = edge_at(tool, local1[0])
        lateral = local1[1] - e[0]
        if local1[0] >= tool.xmin and local1[0] < tool.xmin + tool.edge_dx * float(
            tool.edge.shape[0] - 1
        ):
            if (
                wp.abs(lateral) < tool.half_band
                and local1[2] >= e[1]
                and local1[2] - e[1] < 3.0 * p.radius + wp.abs(a[1]) * run.dt
            ):
                p.side[i] = wp.where(lateral > 0.0, 2, 1)
    j = wp.determinant(p.F[i])
    wp.atomic_min(log.min_j, step, j)
    elastic = p.F[i] - rotation(p.F[i])
    energy = p.volume * (
        p.mu * wp.ddot(elastic, elastic) + 0.5 * p.lam * (j - 1.0) * (j - 1.0)
    )
    wp.atomic_add(
        log.energy, step, wp.vec2(0.5 * p.mass * wp.length_sq(p.v[i]), energy)
    )


@wp.kernel
def advance(run: Protocol):
    run.step[0] += 1


def stencil(points, dx):
    """Quadratic reference B-splines and their physical-space derivatives."""
    origin = (np.floor(points.min(0) / dx) - 2) * dx
    u = (points - origin) / dx
    base = np.floor(u - 0.5).astype(int)
    f = u - base
    w = np.stack(
        [0.5 * (1.5 - f) ** 2, 0.75 - (f - 1) ** 2, 0.5 * (f - 0.5) ** 2], axis=1
    )
    dw = np.stack([f - 1.5, 2 * (1 - f), f - 0.5], axis=1) / dx
    shape = base.max(0) + 3
    nodes, weights, gradients = [], [], []
    for o in np.ndindex(3, 3, 3):
        nodes.append(np.ravel_multi_index((base + o).T, shape))
        basis = [w[:, k, d] for d, k in enumerate(o)]
        weights.append(np.prod(basis, axis=0))
        gradients.append(
            np.column_stack(
                [
                    dw[:, o[d], d] * basis[(d + 1) % 3] * basis[(d + 2) % 3]
                    for d in range(3)
                ]
            )
        )
    coords = origin + np.indices(shape).reshape(3, -1).T * dx
    return np.stack(nodes, 1), np.stack(weights, 1), np.stack(gradients, 1), coords


class ReferenceCut:
    """Three-dimensional uniform, clamped coupon with a prescribed translating knife."""

    def __init__(
        self,
        mesh,
        xyz,
        spacing,
        dx,
        material,
        action,
        dt,
        keff,
        friction,
        cut,
        wrist,
        anchor_y,
        device="cuda:0",
        pic_rate=160.0,
        normal_scale=0.25,
    ):
        self.device = device

        def array(a, dtype):
            return wp.array(np.ascontiguousarray(a), dtype=dtype, device=device)

        def zeros(n, dtype=float):
            return wp.zeros(n, dtype=dtype, device=device)

        self.p = p = Particles()
        p.last_surface = zeros(len(xyz), wp.vec3)
        p.last_reaction = zeros(len(xyz), wp.vec3)
        p.rest = array(xyz, wp.vec3)
        p.x = array(xyz, wp.vec3)
        p.v = zeros(len(xyz), wp.vec3)
        p.F = array(np.tile(np.eye(3), (len(xyz), 1, 1)), wp.mat33)
        p.side = zeros(len(xyz), int)
        p.work = zeros(len(xyz))
        idx, w, grad, grid_x = stencil(xyz, dx)
        p.index, p.weight, p.gradient = (
            array(idx, int),
            array(w, float),
            array(grad, wp.vec3),
        )
        p.volume = spacing**3
        p.mass = material["density_kg_m3"] * p.volume
        young, nu = material["E_MPa"] * 1e6, material["nu"]
        p.mu, p.lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
        p.radius = 0.5 * spacing
        self.g = g = Grid()
        self.nodes = len(grid_x)
        g.mass = zeros(3 * self.nodes)
        for name in ("momentum", "impulse", "vel", "delta"):
            setattr(g, name, zeros(3 * self.nodes, wp.vec3))
        anchor = np.abs(grid_x[:, 1]) >= anchor_y - 1.0e-8
        g.anchor = array(anchor, int)
        if not anchor.any():
            raise ValueError("Fixture lies outside reference grid")
        self.tool = t = Tool()
        vertices, faces, edge = mesh
        self.mesh = wp.Mesh(array(vertices, wp.vec3), array(faces.ravel(), int))
        t.mesh = self.mesh.id
        t.edge = array(edge[:, 1:], wp.vec2)
        t.xmin, t.edge_dx = float(edge[0, 0]), float(edge[1, 0] - edge[0, 0])
        t.half_band, t.wrist, t.friction, t.keff = (
            6 * dx,
            wp.vec3(*wrist),
            friction,
            keff,
        )
        self.run = r = Protocol()
        r.normal_scale = normal_scale
        r.action, r.step, r.cut, r.dt, r.pic = (
            array(action, wp.vec4),
            zeros(1, int),
            int(cut),
            dt,
            1.0 - np.exp(-pic_rate * dt),
        )
        self.log = Trace()
        for name, dtype in self.log._cls.__annotations__.items():
            setattr(self.log, name, zeros(len(action), dtype.dtype))
        self.log.min_j.fill_(10.0)
        self.neighbors = wp.HashGrid(32, 32, 32, device=device)
        self.neighbors.build(p.x, 2.1 * p.radius)

    def substep(self):
        p, g, t, r, log = self.p, self.g, self.tool, self.run, self.log
        wp.launch(clear_grid, 3 * self.nodes, [g], device=self.device)
        self.neighbors.build(p.x, 2.1 * p.radius)
        wp.launch(
            scatter, len(p.x), [p, g, t, r, log, self.neighbors.id], device=self.device
        )
        wp.launch(update_grid, self.nodes, [g, r, log], device=self.device)
        wp.launch(gather, len(p.x), [p, g, t, r, log], device=self.device)
        wp.launch(advance, 1, [r], device=self.device)

    def execute(self, steps, snapshots, record_contacts=False):
        # Captured batches retain independent substep traces while avoiding Python dispatch.
        with wp.ScopedCapture(device=self.device) as capture:
            for _ in range(16):
                self.substep()
        done = 0
        states = {}
        for end in sorted(set([*snapshots, steps])):
            while done + 16 <= end:
                wp.capture_launch(capture.graph)
                done += 16
            while done < end:
                self.substep()
                done += 1
            states[end] = {
                k: getattr(self.p, k).numpy().copy()
                for k in ("x", "v", "F", "side", "work")
            }
            if record_contacts:
                for k in ("last_surface", "last_reaction"):
                    states[end][k] = getattr(self.p, k).numpy().copy()
        wp.synchronize_device(self.device)
        return states


def knife_geometry(root: Path):
    """Use the repository mesh, registered as in the existing vertical-cut experiment."""
    m = trimesh.load(
        root / "third_party/EEF-Cutting-Simulation/assets/knife.obj", force="mesh"
    )
    with np.load(root / ".cache/tools/knife.npz") as pack:
        m.apply_transform(pack["T_asset_to_world"])
    if not m.is_volume:
        raise ValueError(
            "Knife must have closed, consistently wound collision geometry"
        )
    # Work on the flat portion of the blade and retain the entire solid for collisions.
    xs = np.linspace(0.015, 0.065, 201)
    edge, valid = mesh_lower_edge(m, xs, 0.070)
    if not valid.all():
        raise ValueError("Missing blade lower envelope")

    def world(v):
        return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)

    origin = world(edge)[len(xs) // 2]
    return world(m.vertices) - origin, m.faces.copy(), world(edge) - origin
