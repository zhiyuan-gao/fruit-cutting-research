"""Material connectivity for a single cut through a connected, sampled soft fruit."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class SingleCutPieces:
    def __init__(self, rest, spacing):
        # Reference neighbors persist under deformation and never reconnect on contact.
        self.edges = cKDTree(rest).query_pairs(1.25 * spacing, output_type="ndarray")
        self.ids = np.zeros(len(rest), dtype=np.int32)
        count, _ = self.components(self.edges)
        if count != 1:
            raise ValueError(
                "Piece tracking requires one connected reference particle graph"
            )
        self.marked = 0

    def components(self, edges):
        a, b = edges.T
        graph = coo_matrix(
            (np.ones(len(a), bool), (a, b)), shape=(len(self.ids), len(self.ids))
        )
        return connected_components(graph, directed=False)

    def update(self, side):
        marked = np.count_nonzero(side)
        if marked != self.marked:
            a, b = self.edges.T
            count, labels = self.components(self.edges[side[a] * side[b] != -1])
            if count > 2:
                raise RuntimeError(
                    f"Single-cut experiment produced {count} material components"
                )
            if count == 2:
                self.ids = labels + 1
            self.marked = marked
        return self.ids
