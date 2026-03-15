import pandas as pd
from pre_processing_utils import *
import numpy as np
import ast

def df_transform_experimental(df, features_config, labels=False):
    df = df.copy()
    df_tr = pd.DataFrame(index=df.index)
    df_tr['track_id'] = df['track_id']

    parsed_data = df['trajectory'].apply(parse_ewkb)
    df[['longitude', 'latitude', 'altitude', 'rcs']] = pd.DataFrame(parsed_data.tolist(), index=df.index)

    df['times'] = df['trajectory_time'].apply(ast.literal_eval)

    trajectory_data = df.apply(apply_trajectory_data, axis=1)
    df[['distances','speeds','vectors']] = pd.DataFrame(trajectory_data.tolist(), index=df.index)

    curvature_data = df['vectors'].apply(get_curvature_data)
    df[['turns','curvatures']] = pd.DataFrame(curvature_data.tolist(), index=df.index)

    df_tr['airspeed'] = df['airspeed']
    df_tr['min_z'] = df['min_z']
    df_tr['max_z'] = df['max_z']

    if features_config['size_encoding']=='one_hot':
        all_sizes = ['Small bird', 'Large bird', 'Flock', 'Medium bird']
        for size in all_sizes:
            df_tr[f'size_{size}'] = (df['radar_bird_size'] == size).astype(int)
    elif features_config['size_encoding']=='ordinal':
        df_tr['is_flock'] = (df['radar_bird_size']=='Flock').astype(int)
        size_map = {'Large bird': 2, 'Medium bird': 1, 'Small bird': 0}
        df_tr['bird_size'] = df['radar_bird_size'].map(size_map)

    df_tr['duration'] = df['trajectory_time'].apply(get_duration)

    df['timestamp_start'] = pd.to_datetime(df['timestamp_start_radar_utc'])
    df_tr['hour'] = df['timestamp_start'].dt.hour
    df_tr['month'] = df['timestamp_start'].dt.month

    df_tr['avg_rcs'] = df['rcs'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['std_rcs'] = df['rcs'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)
    df_tr['min_rcs'] = df['rcs'].apply(lambda x: np.min(x) if len(x) > 0 else 0.0)
    df_tr['max_rcs'] = df['rcs'].apply(lambda x: np.max(x) if len(x) > 0 else 0.0)

    df_tr['avg_lat'] = df['latitude'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['avg_lon'] = df['longitude'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    
    df_tr['height_fluctuation'] = df['altitude'].apply(lambda x: np.max(x) - np.min(x) if len(x) > 0 else 0.0)
    df_tr['latitude_fluctuation'] = df['latitude'].apply(lambda x: np.max(x) - np.min(x) if len(x) > 0 else 0.0)
    df_tr['longitude_fluctuation'] = df['longitude'].apply(lambda x: np.max(x) - np.min(x) if len(x) > 0 else 0.0)
    df_tr['height_fluctuation_scaled'] = df_tr['height_fluctuation'] / df_tr['duration']
    df_tr['latitude_fluctuation_scaled'] = df_tr['latitude_fluctuation'] / df_tr['duration']
    df_tr['longitude_fluctuation_scaled'] = df_tr['longitude_fluctuation'] / df_tr['duration']

    df_tr['altitude_mean'] = df['altitude'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['altitude_std'] = df['altitude'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)

    df['altitude_diff'] = df['altitude'].apply(lambda x: np.diff(x))
    df_tr['altitude_climb_mean'] = df['altitude_diff'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['altitude_climb_std'] = df['altitude_diff'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)

    df['local_2d_scores'] = df.apply(apply_2dpca_local, axis=1)
    df['local_3d_scores'] = df.apply(apply_3dpca_local, axis=1)

    df_tr['local_2d_circularity_max'] = df['local_2d_scores'].apply(lambda x: np.max(x) if len(x) > 0 else 0.0)
    df_tr['local_3d_circularity_mean'] = df['local_3d_scores'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)

    df_tr["path_length"] = df['distances'].apply(lambda x: np.sum(x) if len(x) > 0 else 0.0)
    df_tr["step_mean"] = df['distances'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr["step_std"] = df['distances'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)
    df_tr["speed_mean"] = df['speeds'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr["speed_std"] = df['speeds'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)

    df_tr['turn_mean'] = df['turns'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['turn_std'] = df['turns'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)
    df_tr['curvature_mean'] = df['curvatures'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['curvature_std'] = df['curvatures'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)

    df['speed_diff'] = df['speeds'].apply(lambda x: np.diff(x))
    df_tr['acc_mean'] = df['speed_diff'].apply(lambda x: np.mean(x) if len(x) > 0 else 0.0)
    df_tr['acc_std'] = df['speed_diff'].apply(lambda x: np.std(x) if len(x) > 0 else 0.0)

    if labels and 'bird_group' in df.columns:
        df_tr['bird_group'] = df['bird_group']

    return df_tr

def prepare_for_training(df_transformed, features_config, class_names, target_col='bird_group'):
    features = [c for c in df_transformed.columns if c not in ['track_id', target_col]]
    X = df_transformed[features].copy()
    # X = X.select_dtypes(include=[np.number])

    mapping = {name: i for i, name in enumerate(class_names)}
    
    if target_col in df_transformed.columns:
        # Use .map() to ensure consistency across all folds and test data
        y = df_transformed[target_col].map(mapping).astype(int)
    else:
        y = None
    for c in features_config:
        if features_config[c] == False:
            X.drop(columns=[c], inplace=True)
            features.remove(c)
    
    return X, y, features
