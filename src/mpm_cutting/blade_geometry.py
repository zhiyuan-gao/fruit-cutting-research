"""Geometry shared by the native solver and the reference-grid trial."""

import numpy as np


def mesh_lower_edge(mesh, columns_x, grip_start):
    """Sample the lower envelope of blade triangles in native knife coordinates (Y up)."""
    segments = mesh.vertices[mesh.edges_unique]
    a, b = segments[:, 0], segments[:, 1]
    span = b[:, 0] - a[:, 0]
    edge = np.zeros((len(columns_x), 3), dtype=np.float32)
    valid = np.zeros(len(columns_x), dtype=np.int32)
    for i, x in enumerate(columns_x):
        if x >= grip_start:
            continue
        crossed = (
            (np.minimum(a[:, 0], b[:, 0]) <= x)
            & (np.maximum(a[:, 0], b[:, 0]) >= x)
            & (span != 0)
        )
        if np.any(crossed):
            t = (x - a[crossed, 0]) / span[crossed]
            section = a[crossed] + t[:, None] * (b[crossed] - a[crossed])
            edge[i] = section[np.argmin(section[:, 1])]
            valid[i] = 1
    return edge, valid
