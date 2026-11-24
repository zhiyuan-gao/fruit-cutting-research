"""Convert the EmbodiedSWE whole avocado and prepare its CulinaryCut signed-distance field."""

from collections import Counter
import hashlib
import io
import math
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from pxr import Usd, UsdGeom
import trimesh

from build_tissue_volumes import layered_labels, peel_interface, save_layers

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/EEF-Cutting-Simulation"
sys.path.insert(0, str(UPSTREAM))
from run import _save_pack_npz, _sdf_cache_key
from fruit_mesh_sdf import translated_mesh_sdf


def write_if_changed(path, text):
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).chmod(0o644)
    os.replace(tmp.name, path)


def lay_on_side(points):
    # Old vertical axis runs along CulinaryCut's X, parallel to the blade.
    return np.column_stack((points[:, 2], points[:, 1], -points[:, 0]))


def whole_mesh(obj):
    source = obj.with_name("avocado.usd")
    stage = Usd.Stage.Open(str(source))
    meshes = [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    (mesh,) = meshes
    if (
        UsdGeom.GetStageMetersPerUnit(stage) != 1
        or UsdGeom.GetStageUpAxis(stage) != "Z"
    ):
        raise ValueError("Expected the original meter-scale, Z-up USD")
    assert np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3)
    transform = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    )
    p = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    p = (p @ transform[:3, :3] + transform[3, :3]) * 1.4
    points = lay_on_side(p)
    faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    uv = np.array(
        UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").ComputeFlattened()
    ).reshape(-1, 3, 2)
    assert len(uv) == len(faces)
    edges = np.stack((faces, np.roll(faces, -1, axis=1)), axis=2).reshape(-1, 2)
    counts = Counter(map(tuple, np.sort(edges, axis=1)))
    boundary = [(a, b) for a, b in edges if counts[tuple(sorted((a, b)))] == 1]
    assert len(boundary) == 3 and len(set(np.ravel(boundary))) == 3
    a, b = boundary[0]
    (c,) = set(np.ravel(boundary)) - {a, b}
    # Match the old prepare_fruit() patch, including the UVs for the single new face.
    faces = np.vstack((faces, [b, a, c]))
    uv = np.concatenate((uv, np.zeros((1, 3, 2))))
    geom = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    if geom.volume < 0:
        faces = faces[:, [0, 2, 1]]
        uv = uv[:, [0, 2, 1]]
        geom = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    assert geom.is_watertight and geom.is_volume and geom.is_winding_consistent
    assert len(geom.split(only_watertight=False, repair=False)) == 1
    (albedo,) = [
        p.GetAttribute("inputs:diffuse_texture").Get().path
        for p in stage.Traverse()
        if p.HasAttribute("inputs:diffuse_texture")
    ]
    assert (obj.parent / albedo).is_file()
    write_if_changed(
        obj.with_suffix(".mtl"), f"newmtl Peel\nKd 1 1 1\nmap_Kd {albedo}\n"
    )
    with io.StringIO() as out:
        out.write("mtllib whole.mtl\no WholeAvocado\n")
        np.savetxt(out, points, fmt="v %.11g %.11g %.11g")
        np.savetxt(out, uv.reshape(-1, 2), fmt="vt %.9g %.9g")
        out.write("usemtl Peel\n")
        for i, (a, b, c) in enumerate(faces + 1):
            out.write(f"f {a}/{3 * i + 1} {b}/{3 * i + 2} {c}/{3 * i + 3}\n")
        write_if_changed(obj, out.getvalue())
    return geom, dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_faces=len(faces) - 1,
        patch_faces=1,
        whole_faces=len(faces),
        watertight=bool(geom.is_watertight),
        components=1,
        uniform_scale=1.4,
        native_extent_m=geom.extents.tolist(),
        albedo=albedo,
    )


def add_pit_and_peel(obj, pack):
    from scipy.spatial import ConvexHull

    folder = ROOT / "assets/tissues"
    spec = json.loads((folder / "avocado.json").read_text())
    geometry = spec["geometry"]
    # Same 32 by 16 latitude-longitude ellipsoid as the original RoboNinja scene.
    rx, ry, rz = geometry["pit_radii_m"]
    n, rings = 32, 16
    points = [(0.0, 0.0, rz), (0.0, 0.0, -rz)]
    for ring in range(1, rings):
        theta = math.pi * ring / rings
        for i in range(n):
            phi = 2 * math.pi * i / n
            points.append(
                (
                    rx * math.sin(theta) * math.cos(phi),
                    ry * math.sin(theta) * math.sin(phi),
                    rz * math.cos(theta),
                )
            )
    triangles = []
    for i in range(n):
        nxt = (i + 1) % n
        triangles.extend(
            [
                (0, 2 + i, 2 + nxt),
                (1, 2 + (rings - 2) * n + nxt, 2 + (rings - 2) * n + i),
            ]
        )
        for ring in range(rings - 2):
            a, b = 2 + ring * n + i, 2 + ring * n + nxt
            triangles.extend([(a, a + n, b + n), (a, b + n, b)])
    # USD Point3f precision of the reference primitive.
    local = np.asarray(points, dtype=np.float32).astype(float)
    vertices = lay_on_side(local + geometry["pit_center_in_upright_fruit_m"])
    hull = ConvexHull(vertices)
    tf = pack["T_asset_to_world"]
    if not np.allclose(tf[:3, :3], np.eye(3)):
        raise ValueError(
            "The prepared whole OBJ must use a translation-only scene pose"
        )
    origin = pack["origin"].astype(float) - tf[:3, 3]
    pitch = float(pack["voxel_size"])
    labels = layered_labels(pack["sdf"].T, geometry["peel_thickness_m"])
    lower = np.ceil((vertices.min(axis=0) - origin) / pitch).astype(int)
    upper = np.floor((vertices.max(axis=0) - origin) / pitch).astype(int) + 1
    if (lower < 0).any() or (upper > labels.shape).any():
        raise ValueError("Pit extends beyond the fruit grid")
    axes = [origin[i] + pitch * np.arange(lower[i], upper[i]) for i in range(3)]
    x, y, z = np.ix_(*axes)
    inside = np.ones(upper - lower, dtype=bool)
    for a, b, c, d in hull.equations:
        inside &= a * x + b * y + c * z + d <= 1e-10
    block = labels[tuple(slice(lo, hi) for lo, hi in zip(lower, upper))]
    if (
        not np.all(block[inside] == 2)
        or abs(inside.sum() * pitch**3 / hull.volume - 1) > 0.01
    ):
        raise ValueError(
            "Registered pit must be inside the pulp and match the source volume"
        )
    block[inside] = 3
    faces = np.asarray(triangles, dtype=np.int32)
    save_layers(
        spec,
        obj,
        labels,
        origin,
        pitch,
        pit_vertices=vertices,
        pit_faces=faces,
        peel_interface_vox=peel_interface(
            pack["sdf"].T, geometry["peel_thickness_m"], pitch
        ),
    )


def prepare_sdf(geom, obj, cfg, cache, tissues=False):
    fruit = cfg["cutting_mesh"]
    tf = fruit.get("initial_transform", {})
    voxels = fruit["sdf_voxel"]
    pack, probes, err = translated_mesh_sdf(geom, obj, tf, voxels)
    sdf = pack["sdf"]
    for axis in range(3):
        assert np.all(sdf.take([0, -1], axis=axis) > 0)
    pitch = float(pack["voxel_size"])
    stride = max(1, round(fruit["particle_spacing"] / pitch))
    volume = float(np.count_nonzero(sdf < 0) * pitch**3)
    assert abs(volume / geom.volume - 1) < 0.01
    key = _sdf_cache_key(str(obj), tf, voxels)
    with NamedTemporaryFile(dir=cache, suffix=".npz", delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        _save_pack_npz(str(staged), pack)
        staged.replace(cache / f"cuttingmesh.{key}.npz")
    finally:
        staged.unlink(missing_ok=True)
    if tissues:
        add_pit_and_peel(obj, pack)
    return dict(
        sdf_probes=probes,
        sdf_max_error_m=err,
        mesh_volume_m3=float(geom.volume),
        sdf_volume_m3=volume,
        particles=int((sdf[::stride, ::stride, ::stride] < 0).sum()),
        particle_pitch_m=stride * pitch,
    )
