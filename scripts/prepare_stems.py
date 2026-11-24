"""Restore the apple stalk and isolate the peach stalk on the pinned CulinaryCut meshes."""

from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/EEF-Cutting-Simulation/assets"
TARGET = ROOT / "assets/tissues"


def vertex_indices(face):
    return [int(c.split("/")[0]) - 1 for c in face.split()[1:]]


def apple():
    rows = (SOURCE / "apple.obj").read_text().splitlines()
    verts = np.array([r.split()[1:4] for r in rows if r.startswith("v ")], dtype=float)
    faces = [r for r in rows if r.startswith("f ")]
    socket = [vertex_indices(f) for f in faces if 55 in vertex_indices(f)]
    edges = [(i, j) for f in socket for i, j in zip(f, f[1:] + f[:1])]
    counts = Counter(tuple(sorted(pair)) for pair in edges)
    boundary = {i: j for i, j in edges if counts[tuple(sorted((i, j)))] == 1}
    rim = [min(boundary)]
    while boundary[rim[-1]] != rim[0]:
        rim.append(boundary[rim[-1]])
    assert len(rim) == 3
    points = verts[rim + rim[:1]] - verts[55]
    angles = np.unwrap(np.arctan2(points[:, 2], points[:, 0]))
    theta = np.concatenate(
        [np.linspace(a, b, 4, endpoint=False) for a, b in zip(angles, angles[1:])]
    )
    new_vertices, stem, previous = [], [], []
    for t, radius in [
        (0.17, 0.00245),
        (0.38, 0.0022),
        (0.62, 0.00205),
        (0.84, 0.00192),
        (1.0, 0.00205),
    ]:
        center = verts[55] + [-0.008 * t**1.45, 0.024 * t, -0.0018 * t + 0.001 * t * t]
        first = len(verts) + len(new_vertices)
        ring = list(range(first, first + len(theta)))
        for angle in theta:
            r = radius * (1 + 0.08 * np.sin(3 * angle) + 0.035 * np.cos(5 * angle))
            new_vertices.append(
                center
                + [r * np.cos(angle), 0.0005 * t**4 * np.cos(angle), r * np.sin(angle)]
            )
        if previous:
            for j in range(12):
                following = (j + 1) % 12
                stem.extend(
                    [
                        [previous[j], previous[following], ring[following]],
                        [previous[j], ring[following], ring[j]],
                    ]
                )
        else:
            for j, v in enumerate(rim):
                k = 4 * j
                stem.append([v, rim[(j + 1) % 3], ring[(k + 4) % 12]])
                stem.extend(
                    [v, ring[(k + l + 1) % 12], ring[k + l]] for l in (3, 2, 1, 0)
                )
        previous = ring
    tip = len(verts) + len(new_vertices)
    new_vertices.append(np.mean(new_vertices[-12:], axis=0))
    stem.extend([previous[i], previous[(i + 1) % 12], tip] for i in range(12))
    header = [r for r in rows if not r.startswith(("f ", "mtllib ", "g "))]
    header.insert(0, "mtllib ../../third_party/EEF-Cutting-Simulation/assets/apple.mtl")
    header.extend("v " + " ".join(f"{n:.9f}" for n in q) for q in new_vertices)
    body = [f for f in faces if 55 not in vertex_indices(f)]
    stalk = ["f " + " ".join(f"{i + 1}/1/1" for i in face) for face in stem]
    (TARGET / "apple-with-stem.obj").write_text(
        "\n".join(header + ["g Body"] + body + ["g Stem"] + stalk) + "\n"
    )


def peach():
    rows = (SOURCE / "peach.obj").read_text().splitlines()
    # The first connected fruit body uses vertices 1..157. Remaining components are leaves.
    faces = [r for r in rows if r.startswith("f ") and max(vertex_indices(r)) < 157]
    channels = [[r for r in rows if r.startswith(k)] for k in ("v ", "vt ", "vn ")]
    corners = [[list(map(int, c.split("/"))) for c in f.split()[1:]] for f in faces]
    used = [sorted({c[j] for f in corners for c in f}) for j in range(3)]
    lookup = [{old: new for new, old in enumerate(ids, 1)} for ids in used]
    header = [
        "# CulinaryCut peach body with original UVs.",
        "mtllib ../../third_party/EEF-Cutting-Simulation/assets/peach.mtl",
        "o PeachBody",
    ]
    for raw, ids in zip(channels, used):
        header.extend(raw[i - 1] for i in ids)
    header.append("usemtl blinn1SG.001")
    groups = {"Body": [], "Stem": []}
    for f in corners:
        group = "Stem" if all(118 <= c[0] <= 132 for c in f) else "Body"
        groups[group].append(
            "f " + " ".join("/".join(str(lookup[j][c[j]]) for j in range(3)) for c in f)
        )
    for name, group in groups.items():
        header.extend(["g " + name, *group])
    (TARGET / "peach-body.obj").write_text("\n".join(header) + "\n")
