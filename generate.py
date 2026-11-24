"""Generate a registered USD fruit, MPM quadrature particles and independent rigid bodies."""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent
FRUITS = sorted(p.name for p in (ROOT / "templates").iterdir() if (p / "template.json").exists())
FRUIT_PATH = "/World/CulinaryCut/Fruit"


def affine(p, matrix):
    return p @ matrix[:3, :3].T + matrix[:3, 3]


def choose(rng, interval):
    return float(rng.uniform(*interval))


class Morph:
    """A common smooth rest-coordinate map, including a localized interior offset."""

    def __init__(self, rng, bounds, settings):
        self.center = bounds.mean(axis=0)
        self.half = (bounds[1] - bounds[0]) * 0.5
        self.axis = int(self.half.argmax())
        self.scale = choose(rng, settings["size_scale"]) * rng.uniform(*settings["axis_scale"], 3)
        self.bend = rng.uniform(*settings["bend"], 3)
        self.bend[self.axis] = 0
        self.taper = choose(rng, settings["taper"])
        self.shift = rng.uniform(*settings["inner_shift"], 3)
        self.expansion = choose(rng, settings["inner_expansion"])
        self.z_offset = 0.0

    def __call__(self, points):
        u = (points - self.center) / self.half
        t = u[:, self.axis : self.axis + 1]
        radial = u.copy()
        radial[:, self.axis] = 0
        w = np.exp(-2.2 * np.sum(u * u, axis=1, keepdims=True))
        warped = u + self.taper * t * radial + self.bend * t * t + w * (self.shift + self.expansion * u)
        return self.center + self.half * self.scale * warped + [0, 0, self.z_offset]

    def derivative(self, points):
        delta = 1.0e-5
        return np.stack(
            [(self(points + d) - self(points - d)) / (2 * delta) for d in np.eye(3) * delta], axis=-1
        )

    def record(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vars(self).items()}


def sample_materials(rng, spec, stage):
    result = {}
    for name, baseline in spec["materials"].items():
        values = dict(baseline, **stage["materials"].get(name, {}))
        result[name] = {"mode": values.pop("mode")}
        for key, source in values.items():
            result[name][key] = choose(rng, source["range"])
    # Preserve the geometry stream of v1 samples when a template retires a material draw.
    rng.random(spec.get("reserved_material_draws", 0))
    if spec["fruit"] == "orange":
        result["Membrane"]["E_MPa"] = result["Pulp"]["E_MPa"]
    if spec["fruit"] == "strawberry":
        result["Peel"]["E_MPa"] = result["Pulp"]["E_MPa"]
    return result


def recolor(path, style):
    if style == "original":
        return
    with Image.open(path) as original:
        rgb = np.asarray(original.convert("RGB"), dtype=float) / 255.0
    if style == "banana_green":
        # Change the yellow portions and retain brown marks and shadows.
        mask = np.clip(3 * (np.minimum(rgb[..., 0], rgb[..., 1]) - rgb[..., 2]), 0.0, 1.0)[..., None]
        tint = rgb * [0.62, 1.04, 0.83]
    elif style in ("avocado_green", "avocado_dark"):
        mask = 1.0
        tint = (
            rgb * [0.75, 1.18, 0.66] + [0.012, 0.035, 0.006]
            if style == "avocado_green"
            else rgb * [0.55, 0.44, 0.60]
        )
    else:
        raise ValueError(style)
    Image.fromarray(np.uint8(255 * np.clip((1 - mask) * rgb + mask * tint, 0, 1))).save(path)


def set_points(mesh, points):
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points, np.float32)))
    mesh.CreateExtentAttr().Set([Gf.Vec3f(*p) for p in [points.min(0), points.max(0)]])


def deform_normals(morph, points, normals):
    jacobian = morph.derivative(points)
    result = np.linalg.solve(jacobian.transpose(0, 2, 1), np.asarray(normals)[..., None])[..., 0]
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)


def components(mesh):
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    _, inverse = np.unique(points.round(8), axis=0, return_inverse=True)
    tri = inverse[faces]
    graph = coo_matrix(
        (np.ones(tri.size), (tri.ravel(), np.roll(tri, 1, axis=1).ravel())), shape=(inverse.max() + 1,) * 2
    )
    _, labels = connected_components(graph, directed=False)
    return points, faces, labels[inverse]


def seed_prim(stage, parent, number, points, faces, colors, normals, material, density):
    # The center is in USD world metres. Mesh vertices are local to this body.
    center = points.mean(axis=0)
    path = parent + f"/Seed_{number:03d}"
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateDensityAttr(float(density))
    mesh = UsdGeom.Mesh.Define(stage, path + "/Geometry")
    set_points(mesh, points - center)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.ravel(), np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    if normals is not None:
        mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(normals, np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if colors is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("tissueColor", Sdf.ValueTypeNames.Color3fArray, "vertex").Set(
            Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors, np.float32))
        )
    if material:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    else:
        mesh.CreateDisplayColorAttr([(0.62, 0.43, 0.17)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    return path


def rigid_geometry(stage, morph, materials):
    records = []
    for name, row in materials.items():
        if row["mode"] == "soft_mpm":
            continue
        if row["mode"] == "fixed_rigid":
            path = "/World/CulinaryCut/RigidPit" if name == "Pit" else FRUIT_PATH + "/Stone"
            mesh = UsdGeom.Mesh.Get(stage, path)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            records.append((name, path, morph(pts)))
            continue
        path = FRUIT_PATH + "/" + name
        mesh = UsdGeom.Mesh.Get(stage, path)
        if not mesh:
            raise ValueError(f"Missing independently connected seed geometry at {path}")
        pts, faces, labels_v = components(mesh)
        material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
        colors = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("tissueColor").Get())
        normals = np.asarray(mesh.GetNormalsAttr().Get())
        pieces = []
        for k in np.unique(labels_v):
            subset = faces[labels_v[faces[:, 0]] == k]
            vertices, ids = np.unique(subset, return_inverse=True)
            pieces.append((pts[vertices], ids.reshape(-1, 3), colors[vertices], normals[vertices]))
        stage.RemovePrim(path)
        UsdGeom.Xform.Define(stage, path)
        pieces.sort(key=lambda piece: tuple(piece[0].mean(0)))
        for i, (pts, faces, colors, normals) in enumerate(pieces, 1):
            if normals is not None:
                normals = deform_normals(morph, pts, normals)
            pts = morph(pts)
            prim = seed_prim(stage, path, i, pts, faces, colors, normals, material, row["density_kg_m3"])
            records.append((name, prim, pts))
    return records


def make_particles(anatomy, morph, spec, materials, spacing, rind_depth, bodies):
    labels = anatomy["labels"]
    g = anatomy["grid_to_usd"]
    voxel = np.linalg.norm(g[:3, :3], axis=0)
    axes = [np.arange(0, n - 0.5, spacing / v) for n, v in zip(labels.shape, voxel)]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ix = tuple(np.rint(coords).astype(int).T)
    ids = labels[ix].copy()
    lobes = anatomy["lobes"][ix]
    names = spec["material_order"]
    soft = np.array([False] + [materials[k]["mode"] == "soft_mpm" for k in names])
    if rind_depth is not None:
        distance = ndimage.distance_transform_edt(labels > 0, sampling=voxel)
        ids[(ids == 1) & (distance[ix] <= rind_depth + 0.5 * voxel.min())] = names.index("GreenRind") + 1
    keep = soft[ids]
    base = affine(coords[keep], g)
    points = morph(base)
    ids, lobes = ids[keep], lobes[keep]
    vol = spacing**3 * np.linalg.det(morph.derivative(base))
    if vol.min() <= 0:
        raise ValueError("Morph folded the particle grid")
    keep = np.ones(len(points), bool)
    for _, _, vertices in bodies:
        hull = ConvexHull(vertices)
        near = np.flatnonzero(((points >= vertices.min(0)) & (points <= vertices.max(0))).all(1))
        for chunk in np.array_split(near, max(1, len(near) // 256)):
            inside = (points[chunk] @ hull.equations[:, :3].T + hull.equations[:, 3] <= 1.0e-9).all(1)
            keep[chunk[inside]] = False
    if set(ids[keep]) != set(np.flatnonzero(soft)):
        raise ValueError("Particle spacing misses a tissue; choose a smaller --spacing-mm.")
    return dict(
        x=points[keep].astype("f4"), volume_m3=vol[keep].astype("f4"), tissue=ids[keep], segment=lobes[keep]
    )


def generate(name, seed, index, output, stage_name=None, spacing=0.002):
    template = ROOT / "templates" / name
    spec = json.loads((template / "template.json").read_text())
    # Stable keyed stream, independent of selection order and batch size.
    key = hashlib.sha256(f"fruit-v1/{name}/{seed}/{index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:16], "little"))
    stage_name = stage_name or str(rng.choice(sorted(spec["stages"])))
    cfg = spec["stages"][stage_name]
    mats = sample_materials(rng, spec, cfg)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        work = Path(tmp)
        result = work / "result"
        result.mkdir()
        base = template / "base.usdz"
        base_digest = hashlib.sha256(base.read_bytes()).hexdigest()
        if base_digest != (template / "base.sha256").read_text().strip():
            raise ValueError(f"Template hash mismatch for {name}")
        with zipfile.ZipFile(base) as package:
            entries = package.namelist()
            package.extractall(work)
        scene = Usd.Stage.Open(str(work / entries[0]))
        meshes = [UsdGeom.Mesh(p) for p in scene.Traverse() if p.IsA(UsdGeom.Mesh)]
        cloud = np.concatenate([np.asarray(m.GetPointsAttr().Get()) for m in meshes])
        morph = Morph(rng, np.stack([cloud.min(0), cloud.max(0)]), spec["geometry"])
        morph.z_offset = float(cloud[:, 2].min() - morph(cloud)[:, 2].min())
        with np.load(template / "anatomy.npz") as anatomy:
            bodies = rigid_geometry(scene, morph, mats)
            for m in meshes:
                if not m:
                    continue
                p = np.asarray(m.GetPointsAttr().Get(), dtype=float)
                normals = m.GetNormalsAttr().Get()
                if normals is not None and len(normals) == len(p):
                    ns = deform_normals(morph, p, normals)
                    m.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(ns, np.float32)))
                set_points(m, morph(p))
            rind = choose(rng, spec["geometry"]["green_rind_depth_m"]) if name == "melon" else None
            data = make_particles(anatomy, morph, spec, mats, spacing, rind, bodies)
        for asset in entries[1:]:
            if asset.lower().endswith((".png", ".jpg", ".jpeg")):
                recolor(work / asset, cfg["texture"])
        scene.GetRootLayer().Save()
        writer = Usd.ZipFileWriter.CreateNew(str(result / "fruit.usdz"))
        for asset in entries:
            if not writer.AddFile(str(work / asset), asset):
                raise IOError(asset)
        if not writer.Save():
            raise IOError("USDZ packaging failed")
        rigid = []
        for i, (material, path, points) in enumerate(bodies):
            hull = ConvexHull(points)
            data[f"body_{i}"] = points[hull.vertices].astype("f4")
            rigid.append(
                dict(material=material, path=path, mode=mats[material]["mode"], volume_m3=hull.volume)
            )
        np.savez_compressed(result / "mpm.npz", **data)
        record = dict(
            version=1,
            fruit=name,
            seed=seed,
            index=index,
            stage=stage_name,
            texture=cfg["texture"],
            stage_note=cfg["note"],
            template_sha256=base_digest,
            profile_sha256=hashlib.sha256((template / "template.json").read_bytes()).hexdigest(),
            units="metre, kilogram, second; E_MPa in MPa",
            frame="USD Z-up",
            spacing_m=spacing,
            morph=morph.record(),
            green_rind_depth_m=rind,
            material_order=spec["material_order"],
            materials=mats,
            rigid_bodies=rigid,
            particle_count=len(data["x"]),
        )
        (result / "sample.json").write_text(json.dumps(record, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(result, output)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fruit", nargs="+", choices=FRUITS + ["all"], default=["all"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stage")
    parser.add_argument("--spacing-mm", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.start < 0 or not 0.75 <= args.spacing_mm <= 3.0:
        parser.error("count >= 1, start >= 0 and spacing in [0.75,3] mm required")
    fruits = FRUITS if "all" in args.fruit else list(dict.fromkeys(args.fruit))
    if args.stage and any(
        args.stage not in json.loads((ROOT / "templates" / f / "template.json").read_text())["stages"]
        for f in fruits
    ):
        parser.error("The requested stage must be supported by every selected fruit")
    for fruit in fruits:
        for i in range(args.start, args.start + args.count):
            dest = args.out / fruit / f"{args.seed}_{i:04d}"
            rec = generate(fruit, args.seed, i, dest, args.stage, args.spacing_mm / 1000.0)
            print(dest, rec["stage"], rec["particle_count"], len(rec["rigid_bodies"]), flush=True)
