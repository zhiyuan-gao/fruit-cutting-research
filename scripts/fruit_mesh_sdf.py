"""GPU sampling of a closed fruit mesh on the native CulinaryCut distance grid."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import trimesh
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"third_party/EEF-Cutting-Simulation"))
from sdf_utils.mesh_sdf import mesh_to_sdf
from sdf_utils.transforms import parse_transform


@wp.kernel
def signed_distance(mesh: wp.uint64, positions: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
    i = wp.tid()
    p = positions[i]
    closest = wp.mesh_query_point_sign_winding_number(mesh, p, 10.0)
    result[i] = closest.sign * wp.length(p - wp.mesh_eval_position(mesh, closest.face, closest.u, closest.v))


def mesh_distance_query(geom):
    scale = float(max(geom.extents))
    normalized = trimesh.Trimesh(vertices=geom.vertices/scale, faces=geom.faces, process=False)
    wp.init()
    gpu = wp.Mesh(points=wp.array(normalized.vertices, dtype=wp.vec3, device="cuda"),
                  indices=wp.array(geom.faces.astype(np.int32).ravel(), dtype=wp.int32, device="cuda"),
                  support_winding_number=True)

    def query(points):
        q = wp.array(points/scale, dtype=wp.vec3, device="cuda")
        values = wp.empty(len(points), dtype=float, device="cuda")
        wp.launch(signed_distance, dim=len(points), inputs=[gpu.id, q, values], device="cuda")
        return values.numpy()*scale
    return query


def translated_mesh_sdf(geom, obj, tf, resolution):
    """Sample a closed fruit mesh in the native grid under a pure translation."""
    transform = parse_transform(tf)
    if not np.allclose(transform[:3,:3], np.eye(3)):
        raise ValueError("Expected a translation-only fruit pose")
    shift = np.asarray(tf.get("translate",[0.,0.,0.]),dtype=np.float64)
    query = mesh_distance_query(geom)
    scale = float(max(geom.extents))
    normalized = trimesh.Trimesh(vertices=geom.vertices/scale, faces=geom.faces, process=False)

    rng = np.random.default_rng(11)
    surface = geom.vertices[::83]
    probes = np.vstack((geom.center_mass, geom.bounds[1]+0.003,
                        surface+rng.uniform(-0.0008,0.0008,surface.shape)))
    ref = -trimesh.proximity.signed_distance(normalized, probes/scale)*scale
    err = float(np.max(np.abs(ref-query(probes))))
    assert err < 0.00005, err
    def sample(_mesh, points):
        return np.concatenate([query(block-shift) for block in np.array_split(points, 32)])
    with patch("trimesh.proximity.signed_distance", sample):
        pack = mesh_to_sdf(str(obj), tf, resolution, flip_mode="off")
    return pack, len(ref), err
