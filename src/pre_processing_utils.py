
import numpy as np

from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_trajectory_data(lon, lat, alt, time):
    dists = []
    speeds = []
    vecs = []

    for i in range(len(lon) - 1):
        d = haversine_distance(lon[i], lat[i], lon[i + 1], lat[i + 1])
        dt = time[i + 1] - time[i]

        if dt <= 0:
            continue

        speed = d / dt

        dists.append(d)
        speeds.append(speed)

        dx = lon[i + 1] - lon[i]
        dy = lat[i + 1] - lat[i]

        vecs.append([dx, dy])

    return dists, speeds, vecs

def apply_trajectory_data(row):
    return get_trajectory_data(row['longitude'], row['latitude'], row['altitude'], row['times'])

def get_curvature_data(vecs):
    turns = []
    curvatures = []

    for i in range(len(vecs) - 1):
        v1 = vecs[i]
        v2 = vecs[i + 1]

        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6

        angle = np.arccos(np.clip(dot / norm, -1, 1))
        turns.append(angle)

        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        curvatures.append(cross / norm)

    return turns, curvatures

from shapely import wkb

def parse_ewkb(ewkb_string):
    # Load the single geometry
    geom = wkb.loads(ewkb_string, hex=True)

    # Use the fast zip trick to get a tuple of 4 lists: ([x...], [y...], [z...], [rsc...])
    return tuple(map(list, zip(*geom.coords)))


def get_duration(times):
    x = np.fromstring(times.strip("[]"), sep=',')
    return float(x[-1] - x[0]) if len(x) > 0 else 0.0

def get_local_2d_circularity(x_list, y_list, window_size=20):
    trajectory = np.column_stack((x_list, y_list))
    n_points = len(trajectory)
    local_scores = np.zeros(n_points)

    for i in range(n_points):
        start = max(0, i - window_size // 2)
        end = min(n_points, i + window_size // 2)
        window = trajectory[start:end]

        if len(window) < 2:
            continue

        # Covariance for 2D (results in 2x2 matrix)
        cov_matrix = np.cov(window, rowvar=False, bias=True)

        # Get the 2 eigenvalues (v1, v2)
        evals = np.linalg.eigvalsh(cov_matrix)
        evals = np.sort(evals)[::-1]

        v1 = evals[0]
        v2 = evals[1] if len(evals) > 1 else 0

        # Circularity score for 2D
        # If v1 == v2, the shape is a perfect circle.
        # Score approaches 1.0 for circles and 0 for straight lines.
        score = v2 / v1 if v1 > 1e-9 else 0.0
        local_scores[i] = score

    return local_scores

def get_local_3d_circularity(x_list, y_list, z_list, window_size=20):
    trajectory = np.column_stack((x_list, y_list, z_list))
    n_points = len(trajectory)

    if n_points < window_size:
        return [0.0] * n_points

    local_scores = []
    half_win = window_size // 2

    for i in range(n_points):
        start = max(0, i - half_win)
        end = min(n_points, i + half_win)
        window = trajectory[start:end]

        if len(window) < 3:
            local_scores.append(0.0)
            continue

        # PCA Step 1: Center the data
        centered_matrix = window - np.mean(window, axis=0)

        # PCA Step 2: Singular Value Decomposition
        # s contains the singular values.
        # The eigenvalues of the covariance matrix are proportional to s**2.
        _, s, _ = np.linalg.svd(centered_matrix, full_matrices=False)

        # Calculate explained variance (eigenvalues)
        eigenvalues = s ** 2

        # Calculate explained variance ratio (v1, v2, v3)
        v = eigenvalues / np.sum(eigenvalues)

        # Handle cases where v might have fewer than 3 components due to alignment
        v1 = v[0] if len(v) > 0 else 1e-9
        v2 = v[1] if len(v) > 1 else 0.0
        v3 = v[2] if len(v) > 2 else 0.0

        # Calculate local circularity score
        score = (v2 - v3) / v1
        local_scores.append(score)

    return local_scores

# Wrapper for df batch apply
def apply_2dpca_local(row):
    return get_local_2d_circularity(row['latitude'], row['longitude'])

def apply_3dpca_local(row):
    return get_local_3d_circularity(row['latitude'], row['longitude'], row['altitude'])


