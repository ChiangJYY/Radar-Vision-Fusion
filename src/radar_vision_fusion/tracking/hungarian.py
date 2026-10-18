from __future__ import annotations

import math
from typing import List, Tuple


EPSILON = 1e-10


class OrderedSet:
    def __init__(self, size: int) -> None:
        self._flags = [False] * size
        self._items: List[int] = []

    def clear(self) -> None:
        size = len(self._flags)
        self._flags = [False] * size
        self._items.clear()

    def has(self, value: int) -> bool:
        return self._flags[value]

    def insert(self, value: int) -> None:
        if not self._flags[value]:
            self._flags[value] = True
            self._items.append(value)

    def size(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> int:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

    def __eq__(self, other: "OrderedSet") -> bool:
        if self.size() != other.size():
            return False
        for item in self._items:
            if not other.has(item):
                return False
        return True


class Matching:
    def __init__(self, size: int) -> None:
        self._p = [-1] * size
        self._ip = [-1] * size

    def add(self, i: int, j: int) -> None:
        self._p[i] = j
        self._ip[j] = i

    def get_j(self, i: int) -> int:
        return self._p[i]

    def get_i(self, j: int) -> int:
        return self._ip[j]

    def find_unmatched_i(self) -> int:
        for idx, value in enumerate(self._p):
            if value == -1:
                return idx
        return len(self._p)

    def copy_to(self, out: List[int]) -> None:
        for idx, value in enumerate(self._p):
            out[idx] = value


def _add_neighbours(
    i: int,
    parent_index: int,
    size: int,
    edge: List[float],
    lx: List[float],
    ly: List[float],
    neighbours: OrderedSet,
    parent_neighbours: List[int],
) -> None:
    before = neighbours.size()
    for j in range(size):
        if abs(lx[i] + ly[j] - edge[i * size + j]) < EPSILON:
            neighbours.insert(j)
    parent_neighbours.extend([parent_index] * (neighbours.size() - before))


def _find_neighbours(
    s_nodes: List[int],
    size: int,
    edge: List[float],
    lx: List[float],
    ly: List[float],
    neighbours: OrderedSet,
    parent_neighbours: List[int],
) -> None:
    neighbours.clear()
    parent_neighbours.clear()
    for idx, value in enumerate(s_nodes):
        _add_neighbours(value, idx, size, edge, lx, ly, neighbours, parent_neighbours)


def _improve_feasible_labeling(
    size: int,
    edge: List[float],
    s_nodes: List[int],
    t_nodes: OrderedSet,
    lx: List[float],
    ly: List[float],
) -> None:
    a = float("inf")
    for s_value in s_nodes:
        for j in range(size):
            if not t_nodes.has(j):
                a = min(a, lx[s_value] + ly[j] - edge[s_value * size + j])

    if not (a < float("inf")):
        return

    for s_value in s_nodes:
        lx[s_value] -= a
    for j in t_nodes:
        ly[j] += a


def _augment_matching(
    s_nodes: List[int],
    t_nodes: OrderedSet,
    parent_t: List[int],
    matching: Matching,
) -> None:
    j = t_nodes[t_nodes.size() - 1]
    for idx in range(t_nodes.size() - 1, -1, -1):
        if t_nodes[idx] == j:
            ii = parent_t[idx]
            i = s_nodes[ii]
            tmp_j = matching.get_j(i)
            matching.add(i, j)
            j = tmp_j
            if ii == 0:
                return


def _pick_y(neighbours: OrderedSet, t_nodes: OrderedSet) -> int:
    for idx in range(neighbours.size()):
        if not t_nodes.has(neighbours[idx]):
            return idx
    raise RuntimeError("No available Y node found while expected one.")


def hungarian(size: int, edge: List[float]) -> Tuple[List[int], List[float], List[float]]:
    """Maximum weight assignment on a flattened finite square matrix."""
    if size < 0 or len(edge) != size * size:
        raise ValueError("Expected size * size edge weights and nonnegative size")
    if not all(math.isfinite(value) for value in edge):
        raise ValueError("Hungarian edge weights must be finite")
    lx = [max(edge[i * size + j] for j in range(size)) for i in range(size)]
    ly = [0.0] * size

    matching = Matching(size)
    s_nodes: List[int] = []
    t_nodes = OrderedSet(size)
    parent_t: List[int] = []
    neighbours = OrderedSet(size)
    parent_neighbours: List[int] = []

    while True:
        free_i = matching.find_unmatched_i()
        if free_i == size:
            perm = [0] * size
            matching.copy_to(perm)
            return perm, lx, ly

        s_nodes.clear()
        t_nodes.clear()
        parent_t.clear()

        s_nodes.append(free_i)
        _find_neighbours(s_nodes, size, edge, lx, ly, neighbours, parent_neighbours)

        while True:
            if neighbours == t_nodes:
                _improve_feasible_labeling(size, edge, s_nodes, t_nodes, lx, ly)
                _find_neighbours(
                    s_nodes, size, edge, lx, ly, neighbours, parent_neighbours
                )

            jj = _pick_y(neighbours, t_nodes)
            j = neighbours[jj]
            t_nodes.insert(j)
            parent_t.append(parent_neighbours[jj])

            matched_i = matching.get_i(j)
            if matched_i == -1:
                _augment_matching(s_nodes, t_nodes, parent_t, matching)
                break

            s_nodes.append(matched_i)
            _add_neighbours(
                matched_i,
                len(s_nodes) - 1,
                size,
                edge,
                lx,
                ly,
                neighbours,
                parent_neighbours,
            )


def _update_associations(
    rows_size: int,
    cols_size: int,
    size: int,
    edge: List[float],
    match: List[int],
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    associated_pairs: List[Tuple[int, int]] = []
    unassociated_rows: List[int] = []
    unassociated_cols: List[int] = []

    for i in range(size):
        j = match[i]
        if edge[i * size + j] > 0:
            associated_pairs.append((i, j))
        else:
            if i < rows_size:
                unassociated_rows.append(i)
            if j < cols_size:
                unassociated_cols.append(j)
    return associated_pairs, unassociated_rows, unassociated_cols


def hungarian_maximize(
    similarities: List[List[float]],
    min_similarity: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not similarities:
        return [], [], []

    rows_size = len(similarities)
    cols_size = len(similarities[0])
    if any(len(row) != cols_size for row in similarities):
        raise ValueError("Similarity matrix must be rectangular")
    size = max(rows_size, cols_size)
    edge = [0.0] * (size * size)

    for i in range(rows_size):
        for j in range(cols_size):
            if similarities[i][j] > min_similarity:
                edge[i * size + j] = similarities[i][j]

    match, _, _ = hungarian(size, edge)
    return _update_associations(rows_size, cols_size, size, edge, match)


def hungarian_minimize(
    distances: List[List[float]],
    max_distance: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Maximize gated match count, then minimize the sum of matched distances.

    Nonfinite or out-of-gate candidates remain unavailable. Each accepted edge
    receives a cardinality bonus larger than all possible distance savings,
    so an excellent single match cannot displace two valid matches.
    """
    if not distances:
        return [], [], []

    rows_size = len(distances)
    cols_size = len(distances[0])
    if any(len(row) != cols_size for row in distances):
        raise ValueError("Distance matrix must be rectangular")
    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    size = max(rows_size, cols_size)
    edge = [0.0] * (size * size)

    for i in range(rows_size):
        for j in range(cols_size):
            distance = distances[i][j]
            if math.isfinite(distance) and distance < 0:
                raise ValueError("Distances must be nonnegative")
            if math.isfinite(distance) and distance < max_distance:
                # Normalization bounds each cost in [0,1), independently of
                # units and without overflowing for large physical gates.
                edge[i * size + j] = size + 1.0 - distance / max_distance

    match, _, _ = hungarian(size, edge)
    return _update_associations(rows_size, cols_size, size, edge, match)

