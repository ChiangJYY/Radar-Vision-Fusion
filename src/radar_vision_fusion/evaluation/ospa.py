"""Original OSPA computation and result-file reader; no experiment entry point."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

def hungarian(perf: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find minimum edge weight matching given a MxN edge weight matrix using Hungarian Algorithm.
    Returns matching matrix and cost.
    """
    matching = np.zeros_like(perf)

    # Condense the performance matrix by removing unconnected vertices
    num_y = np.sum(~np.isinf(perf), axis=0)
    num_x = np.sum(~np.isinf(perf), axis=1)

    # Find connected vertices
    x_con = np.where(num_x != 0)[0]
    y_con = np.where(num_y != 0)[0]

    # Assemble condensed performance matrix
    p_size = max(len(x_con), len(y_con))
    if p_size == 0:
        return matching, 0.0

    p_cond = np.zeros((p_size, p_size))
    if len(x_con) > 0 and len(y_con) > 0:
        p_cond[:len(x_con), :len(y_con)] = perf[np.ix_(x_con, y_con)]

    # Ensure perfect matching exists
    edge = p_cond.copy()
    edge[p_cond != np.inf] = 0
    cnum = min_line_cover(edge)

    # Project additional vertices and edges
    p_max = np.max(p_cond[p_cond != np.inf]) if np.any(p_cond != np.inf) else 0
    p_size = len(p_cond) + cnum
    p_cond = np.ones((p_size, p_size)) * p_max
    if len(x_con) > 0 and len(y_con) > 0:
        p_cond[:len(x_con), :len(y_con)] = perf[np.ix_(x_con, y_con)]

    # Main program: controls which step is executed
    exit_flag = True
    stepnum = 1
    r_cov = None
    c_cov = None
    M = None
    Z_r = None
    Z_c = None

    while exit_flag:
        if stepnum == 1:
            p_cond, stepnum = step1(p_cond)
        elif stepnum == 2:
            r_cov, c_cov, M, stepnum = step2(p_cond)
        elif stepnum == 3:
            c_cov, stepnum = step3(M, p_size)
        elif stepnum == 4:
            M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(p_cond, r_cov, c_cov, M)
        elif stepnum == 5:
            M, r_cov, c_cov, stepnum = step5(M, Z_r, Z_c, r_cov, c_cov)
        elif stepnum == 6:
            p_cond, stepnum = step6(p_cond, r_cov, c_cov)
        elif stepnum == 7:
            exit_flag = False

    # Remove virtual satellites and targets, uncondense matching
    if len(x_con) > 0 and len(y_con) > 0:
        matching[np.ix_(x_con, y_con)] = M[:len(x_con), :len(y_con)]

    cost = np.sum(perf[matching == 1])
    return matching, cost


def step1(p_cond: np.ndarray) -> Tuple[np.ndarray, int]:
    """Find smallest number in each row and subtract from its row."""
    p_size = len(p_cond)
    for i in range(p_size):
        rmin = np.min(p_cond[i, :])
        p_cond[i, :] -= rmin
    return p_cond, 2


def step2(p_cond: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Find zeros and star them. Repeat for each zero."""
    p_size = len(p_cond)
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    M = np.zeros((p_size, p_size), dtype=int)

    for i in range(p_size):
        for j in range(p_size):
            if p_cond[i, j] == 0 and r_cov[i] == 0 and c_cov[j] == 0:
                M[i, j] = 1
                r_cov[i] = 1
                c_cov[j] = 1

    # Re-initialize cover vectors
    r_cov = np.zeros(p_size, dtype=int)
    c_cov = np.zeros(p_size, dtype=int)
    return r_cov, c_cov, M, 3


def step3(M: np.ndarray, p_size: int) -> Tuple[np.ndarray, int]:
    """Cover each column with a starred zero."""
    c_cov = np.sum(M, axis=0)
    if np.sum(c_cov) == p_size:
        return c_cov, 7
    else:
        return c_cov, 4


def step4(p_cond: np.ndarray, r_cov: np.ndarray, c_cov: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Find noncovered zero and prime it."""
    p_size = len(p_cond)
    zflag = True
    Z_r = 0
    Z_c = 0

    while zflag:
        # Find first uncovered zero
        row = -1
        col = -1
        exit_flag = True
        i = 0
        j = 0

        while exit_flag and i < p_size:
            if p_cond[i, j] == 0 and r_cov[i] == 0 and c_cov[j] == 0:
                row = i
                col = j
                exit_flag = False
            j += 1
            if j >= p_size:
                j = 0
                i += 1
            if i >= p_size:
                exit_flag = False

        if row == -1:
            stepnum = 6
            zflag = False
            Z_r = 0
            Z_c = 0
        else:
            # Prime the uncovered zero
            M[row, col] = 2
            # If there is a starred zero in that row
            starred_cols = np.where(M[row, :] == 1)[0]
            if len(starred_cols) > 0:
                r_cov[row] = 1
                zcol = starred_cols[0]
                c_cov[zcol] = 0
            else:
                stepnum = 5
                zflag = False
                Z_r = row
                Z_c = col

    return M, r_cov, c_cov, Z_r, Z_c, stepnum


def step5(M: np.ndarray, Z_r: int, Z_c: int, r_cov: np.ndarray, c_cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Construct series of alternating primed and starred zeros."""
    zflag = True
    Z_r_list = [Z_r]
    Z_c_list = [Z_c]
    ii = 0

    while zflag:
        # Find starred zero in column
        rindex = np.where(M[:, Z_c_list[ii]] == 1)[0]
        if len(rindex) > 0:
            ii += 1
            Z_r_list.append(rindex[0])
            Z_c_list.append(Z_c_list[ii - 1])
        else:
            zflag = False

        if zflag:
            # Find primed zero in row
            cindex = np.where(M[Z_r_list[ii], :] == 2)[0]
            if len(cindex) > 0:
                ii += 1
                Z_r_list.append(Z_r_list[ii - 1])
                Z_c_list.append(cindex[0])

    # Unstar all starred zeros in path and star all primed zeros
    for idx in range(len(Z_r_list)):
        if M[Z_r_list[idx], Z_c_list[idx]] == 1:
            M[Z_r_list[idx], Z_c_list[idx]] = 0
        else:
            M[Z_r_list[idx], Z_c_list[idx]] = 1

    # Clear covers
    r_cov = np.zeros_like(r_cov)
    c_cov = np.zeros_like(c_cov)

    # Remove all primes
    M[M == 2] = 0

    return M, r_cov, c_cov, 3


def step6(p_cond: np.ndarray, r_cov: np.ndarray, c_cov: np.ndarray) -> Tuple[np.ndarray, int]:
    """Add minimum uncovered value to covered rows, subtract from uncovered columns."""
    a = np.where(r_cov == 0)[0]
    b = np.where(c_cov == 0)[0]

    if len(a) > 0 and len(b) > 0:
        minval = np.min(p_cond[np.ix_(a, b)])
        p_cond[np.where(r_cov == 1)[0], :] += minval
        p_cond[:, np.where(c_cov == 0)[0]] -= minval

    return p_cond, 4


def min_line_cover(edge: np.ndarray) -> int:
    """Calculate minimum line cover deficiency."""
    # Step 2
    r_cov, c_cov, M, stepnum = step2(edge)
    # Step 3
    c_cov, stepnum = step3(M, len(edge))
    # Step 4
    M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(edge, r_cov, c_cov, M)
    # Calculate deficiency
    cnum = len(edge) - np.sum(r_cov) - np.sum(c_cov)
    return cnum


def trk_ospa_dist(
    X: np.ndarray,
    Xlab: np.ndarray,
    Y: np.ndarray,
    Ylab: np.ndarray,
    ospa_p: float = 2.0,
    ospa_c: float = 3.0,
    ospa_l: float = 25.0,
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Compute Schumacher distance (OSPA) between two finite sets X and Y.
    """
    n = X.shape[1] if X.size > 0 else 0
    m = Y.shape[1] if Y.size > 0 else 0

    if n == 0 and m == 0:
        return 0.0, 0.0, 0.0

    if n == 0 or m == 0:
        dist = (1.0 / max(m, n) * ospa_c ** ospa_p * abs(m - n)) ** (1.0 / ospa_p)
        return dist, 0.0, dist

    # Calculate cost/weight matrix for pairings
    D = np.zeros((n, m))
    for j in range(m):
        for i in range(n):
            # Position error (only consider [X, Y] coordinates, indices 0 and 1)
            bdist = np.sum(np.abs(Y[[0, 1], j] - X[[0, 1], i]) ** ospa_p)
            # Label error
            ldist = ospa_l ** ospa_p * (1 if Xlab[i] != Ylab[j] else 0)
            D[i, j] = (bdist + ldist) ** (1.0 / ospa_p)

    D = np.minimum(ospa_c, D) ** ospa_p

    # Compute optimal assignment and cost using Hungarian algorithm
    matching, cost = hungarian(D)

    # Calculate final distance
    dist = (1.0 / max(m, n) * (ospa_c ** ospa_p * abs(m - n) + cost)) ** (1.0 / ospa_p)

    # Output components
    loce = (1.0 / max(m, n) * cost) ** (1.0 / ospa_p)
    carde = (1.0 / max(m, n) * ospa_c ** ospa_p * abs(m - n)) ** (1.0 / ospa_p)

    return dist, loce, carde


def perf_asses(
    trk: np.ndarray,
    est_trk: np.ndarray,
    apar: float = 25.0,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assess performance using OSPA metric.
    """
    ospa_l = apar
    ospa_p = 2.0
    ospa_c = 3.0

    a1, K1, num_trk = trk.shape
    a2, K2, num_etrk = est_trk.shape

    if a1 != a2 or K1 != K2:
        raise ValueError("Wrong input: track dimensions must match")
    if K1 == 0:
        raise ValueError("At least one evaluation frame is required")

    # Find global best assignment of true tracks to estimated tracks
    DELTA = 80.0
    D = np.zeros((num_trk, num_etrk))

    for i in range(num_trk):
        t = trk[:, :, i]
        for j in range(num_etrk):
            et = est_trk[:, :, j]
            for k in range(K1):
                if not (np.isnan(t[0, k]) and np.isnan(et[0, k])):
                    d = np.sqrt(np.sum((t[:, k] - et[:, k]) ** 2))
                    if not np.isnan(d):
                        D[i, j] += min(DELTA, d)
                    else:
                        D[i, j] += DELTA
            D[i, j] /= K1

    # Use Hungarian algorithm to find optimal assignment
    matching, _ = hungarian(D)

    # Find track correspondences: for each estimated track j, find which true track i it matches
    trk_corr = np.full(num_etrk, -1, dtype=int)  # -1 means unmatched
    for i in range(num_trk):
        matched_j = np.where(matching[i, :] == 1)[0]
        if len(matched_j) > 0:
            j = matched_j[0]
            if j < num_etrk:
                trk_corr[j] = i

    # Plot trajectories if requested
    if plot:
        import matplotlib.pyplot as plt

        # Generate colors
        v = [0.1, 0.5, 0.9]
        col = []
        for vi in v:
            for vj in v:
                for vk in v:
                    col.append([vi, vj, vk])

        plt.figure(10)
        for i in range(num_trk):
            plt.plot(trk[0, :, i], trk[1, :, i], '-', color=col[2 * i % len(col)])
            matched_estimates = np.flatnonzero(trk_corr == i)
            if matched_estimates.size:
                j = matched_estimates[0]
                plt.plot(
                    est_trk[0, :, j],
                    est_trk[1, :, j],
                    ':',
                    color=col[2 * i % len(col)],
                )
        plt.show()

    # For every time step k, assign labels and run OSPA
    dist = np.zeros(K1)
    loce = np.zeros(K1)
    carde = np.zeros(K1)
    card_X = np.zeros(K1, dtype=int)
    card_Y = np.zeros(K1, dtype=int)

    for k in range(K1):
        # Collect true tracks at time k
        X = []
        Xl = []
        for i in range(num_trk):
            if not np.isnan(trk[0, k, i]):
                X.append(trk[:, k, i])
                Xl.append(i)

        if len(X) == 0:
            X = np.empty((2, 0))
        else:
            X = np.column_stack(X)
        Xl = np.array(Xl)
        card_X[k] = X.shape[1] if X.size > 0 else 0

        # Collect estimated tracks at time k
        Y = []
        Yl = []
        for i in range(num_etrk):
            if not np.isnan(est_trk[0, k, i]):
                Y.append(est_trk[:, k, i])
                # Find corresponding true track label
                if trk_corr[i] >= 0:
                    Yl.append(trk_corr[i])
                else:
                    Yl.append(12345)  # Unmatched label

        if len(Y) == 0:
            Y = np.empty((2, 0))
        else:
            Y = np.column_stack(Y)
        Yl = np.array(Yl)
        card_Y[k] = Y.shape[1] if Y.size > 0 else 0

        # Compute OSPA distance
        if X.size == 0 and Y.size == 0:
            dist[k] = 0.0
            loce[k] = 0.0
            carde[k] = 0.0
        else:
            dist[k], loce[k], carde[k] = trk_ospa_dist(
                X, Xl, Y, Yl, ospa_p, ospa_c, ospa_l
            )

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(2)
        plt.plot(range(1, K1 + 1), card_X, 'r')
        plt.plot(range(1, K1 + 1), card_Y, 'b')
        plt.show()

        plt.figure(3)
        plt.plot(range(1, K1 + 1), dist, 'r', linewidth=2)
        plt.plot(range(1, K1 + 1), loce, 'b--')
        plt.plot(range(1, K1 + 1), carde, 'k:')
        plt.show()

    return dist, loce, carde, card_X, card_Y


def read_ospa_file(filename: str) -> Tuple[Dict[int, List], Dict[int, List], Dict[int, List], Dict[int, List]]:
    """
    Read OSPA evaluation data file.
    
    Args:
        filename: Path to the OSPA data file
    
    Returns:
        ground_truth_data: Dictionary mapping frame_index to list of [id, x, y]
        fusion_track_data: Dictionary mapping frame_index to list of [id, x, y]
        camera_track_data: Dictionary mapping frame_index to list of [id, x, y]
        radar_track_data: Dictionary mapping frame_index to list of [id, x, y]
    """
    ground_truth_data: Dict[int, List] = {}
    fusion_track_data: Dict[int, List] = {}
    camera_track_data: Dict[int, List] = {}
    radar_track_data: Dict[int, List] = {}
    
    frame_index = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Index explicit frame headers in file order. Empty frames are
            # observations too, including empty frames at the end of a file.
            if line.startswith('Frame'):
                header = line.split()
                if len(header) < 2 or header[0] != 'Frame':
                    raise ValueError(f"Malformed frame header: {line!r}")
                try:
                    int(header[1].rstrip(':'))
                except ValueError as exc:
                    raise ValueError(f"Malformed frame header: {line!r}") from exc
                frame_index += 1
                for data in (ground_truth_data, fusion_track_data,
                             camera_track_data, radar_track_data):
                    data[frame_index] = []
                continue
            
            # Parse trajectory data
            parts = line.split()
            if len(parts) >= 4:
                trajectory_type = parts[0]
                if frame_index == 0:
                    raise ValueError("Trajectory record precedes the first frame header")
                raw_id = float(parts[1])
                if not np.isfinite(raw_id) or not raw_id.is_integer():
                    raise ValueError(f"Invalid trajectory ID: {parts[1]!r}")
                trajectory_id = int(raw_id)
                x = float(parts[2])
                y = float(parts[3])
                
                # Store data based on trajectory type
                data_list = [trajectory_id, x, y]
                
                if trajectory_type == 'Ground_Truth':
                    if frame_index not in ground_truth_data:
                        ground_truth_data[frame_index] = []
                    ground_truth_data[frame_index].append(data_list)
                elif trajectory_type == 'Fusion_track':
                    if frame_index not in fusion_track_data:
                        fusion_track_data[frame_index] = []
                    fusion_track_data[frame_index].append(data_list)
                elif trajectory_type == 'Camera_track':
                    if frame_index not in camera_track_data:
                        camera_track_data[frame_index] = []
                    camera_track_data[frame_index].append(data_list)
                elif trajectory_type == 'Radar_track':
                    if frame_index not in radar_track_data:
                        radar_track_data[frame_index] = []
                    radar_track_data[frame_index].append(data_list)
    
    if frame_index == 0:
        raise ValueError(f"No frame headers found in {filename}")
    return ground_truth_data, fusion_track_data, camera_track_data, radar_track_data


def convert_to_track_array(
    data_dict: Dict[int, List],
    num_frames: int,
    max_tracks: Optional[int] = None,
) -> np.ndarray:
    """
    Convert records to (2, num_frames, number_of_unique_track_ids).

    A column always represents the same trajectory ID across the whole
    sequence. Missing observations remain NaN; row order within a frame has
    no effect. Ground-truth and estimated IDs are mapped independently and
    associated by perf_asses.
    
    Args:
        data_dict: Dictionary mapping frame_index to list of [id, x, y]
        num_frames: Total number of frames
        max_tracks: Deprecated compatibility argument; ignored. Peak concurrent
            track count cannot represent sequential births or fragmented IDs.
    
    Returns:
        Array with columns ordered by sorted unique ID and NaN for missing data
    """
    if not isinstance(num_frames, (int, np.integer)) or num_frames < 0:
        raise ValueError("num_frames must be a nonnegative integer")
    track_ids = set()
    for frame_index, records in data_dict.items():
        if (not isinstance(frame_index, (int, np.integer))
                or not 1 <= frame_index <= num_frames):
            raise ValueError(f"Frame index {frame_index!r} is outside 1..{num_frames}")
        seen = set()
        for record in records:
            if len(record) != 3:
                raise ValueError("Each trajectory record must contain [id, x, y]")
            track_id, x, y = record
            if not np.isfinite(track_id) or float(track_id) != int(track_id):
                raise ValueError(f"Invalid trajectory ID: {track_id!r}")
            if track_id in seen:
                raise ValueError(f"Duplicate trajectory ID {track_id} in frame {frame_index}")
            if not np.isfinite([x, y]).all():
                raise ValueError(f"Non-finite position in frame {frame_index}")
            seen.add(track_id)
        track_ids.update(seen)

    id_to_column = {track_id: column for column, track_id in enumerate(sorted(track_ids))}
    converted_data = np.full((2, num_frames, len(id_to_column)), np.nan)
    for frame_index, records in data_dict.items():
        for track_id, x, y in records:
            converted_data[:, frame_index - 1, id_to_column[track_id]] = (x, y)
    
    return converted_data
