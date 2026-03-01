from shapely import wkb

def parse_ewkb(ewkb_string):
    # Load the single geometry
    geom = wkb.loads(ewkb_string, hex=True)

    # Use the fast zip trick to get a tuple of 4 lists: ([x...], [y...], [z...], [rsc...])
    return tuple(map(list, zip(*geom.coords)))

import numpy as np

def get_duration(times):
    x = np.fromstring(times.strip("[]"), sep=',')
    return float(x[-1] - x[0]) if len(x) > 0 else 0.0

"""
def get_pca_score(x_list, y_list, z_list):
    trajectory = np.column_stack((x_list, y_list, z_list))

    # If the trajectory has fewer than 3 points, PCA won't work well
    if len(trajectory) < 3:
        return 0.0

    # Initialize PCA to find the 3 main components
    pca = PCA(n_components=3)
    pca.fit(trajectory)

    # Get the explained variance ratios (V1, V2, V3)
    # They are automatically sorted from largest to smallest
    v1, v2, v3 = pca.explained_variance_ratio_

    # Calculate the "Circularity Score"
    score = (v2 - v3) / v1

    return score, v1, v2, v3
"""

def get_local_2d_circularity(x_list, y_list, window_size=20):
    # Notice we only stack X and Y
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

"""
def get_local_circularity(x_list, y_list, z_list, window_size=20):
    x_list *= 100000
    y_list *= 100000
    trajectory = np.column_stack((x_list, y_list, z_list))
    n_points = len(trajectory)

    if n_points < window_size:
        return [0.0] * n_points

    local_scores = []

    # Slide the window across the trajectory
    for i in range(n_points):
        # Define window boundaries (centered on the current point)
        start = max(0, i - window_size // 2)
        end = min(n_points, i + window_size // 2)

        window = trajectory[start:end]

        # We need at least 3 points for 3D PCA
        if len(window) < 3:
            local_scores.append(0.0)
            continue

        pca = PCA(n_components=3)
        pca.fit(window)
        v1, v2, v3 = pca.explained_variance_ratio_

        # Calculate the same circularity score, but just for this window
        score = (v2 - v3) / v1
        local_scores.append(score)

    return local_scores
"""
# Wrapper for df batch apply
def apply_pca_local(row):
    return get_local_2d_circularity(row['latitude'], row['longitude'])

import pandas as pd

def df_transform(df):
    df = df.copy()
    df_tr = pd.DataFrame(index=df.index)
    df_tr['track_id'] = df['track_id']

    # Original data after filtering
    parsed_data = df['trajectory'].apply(parse_ewkb)
    df_tr[['longitude', 'latitude', 'altitude', 'rcs']] = pd.DataFrame(parsed_data.tolist(), index=df.index)
    df_tr['airspeed'] = df['airspeed']
    df_tr['min_z'] = df['min_z']
    df_tr['max_z'] = df['max_z']
    dummies = pd.get_dummies(df['radar_bird_size'], prefix='size', drop_first=True)
    df_tr = pd.concat([df_tr, dummies], axis=1)

    # Derived features
    df_tr['duration'] = df['trajectory_time'].apply(get_duration)
    df_tr['avg_rcs'] = df_tr['rcs'].apply(np.mean)
    df_tr['std_rcs'] = df_tr['rcs'].apply(np.std)
    df_tr['min_rcs'] = df_tr['rcs'].apply(np.min)
    df_tr['max_rcs'] = df_tr['rcs'].apply(np.max)
    df_tr['height_fluctuation'] = df_tr['altitude'].apply(np.max) - df_tr['altitude'].apply(np.min)
    df_tr['latitude_fluctuation'] = df_tr['latitude'].apply(np.max) - df_tr['latitude'].apply(np.min)
    df_tr['longitude_fluctuation'] = df_tr['longitude'].apply(np.max) - df_tr['longitude'].apply(np.min)
    df_tr['local_circularity_scores'] = df_tr.apply(apply_pca_local, axis=1)
    df_tr['local_circularity_max'] = df_tr['local_circularity_scores'].apply(np.max)

    df_tr = df_tr.drop(columns=['latitude', 'longitude', 'altitude', 'rcs', 'local_circularity_scores'])
    return df_tr