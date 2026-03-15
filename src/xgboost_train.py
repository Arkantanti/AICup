import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from custom_scoring import macro_ap_xgboost
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def run_cv_training_xgboost(X, y, class_names, params=None):
    """
    Executes Stratified K-Fold training and returns models + OOF predictions.
    Incorporates class weighting based on the 'use_weights' parameter in config.
    """
    if params is None:
        params = {}

    # 1. Handle Parameters
    # Extract training-specific params, providing defaults for essentials
    xgb_params = {
        'n_estimators': params.get('n_estimators', 1000),
        'learning_rate': params.get('learning_rate', 0.05),
        'max_depth': params.get('max_depth', 6),
        'objective': 'multi:softprob',
        'num_class': len(class_names),
        'tree_method': 'hist',
        'early_stopping_rounds': params.get('early_stopping_rounds', 50),
        'random_state': 42,
        'eval_metric': macro_ap_xgboost
    }


    oof_preds = np.zeros((len(X), len(class_names)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    name_to_id = {name: i for i, name in enumerate(class_names)}

    over_names = {'Cormorants': 400, 'Ducks': 100, 'Geese': 120}
    under_names = {'Gulls': 800}

    def safe_over_strat(y):
        counts = Counter(y)
        return {
            name_to_id[name]: max(target, counts[name_to_id[name]])
            for name, target in over_names.items()
            if name in name_to_id and name_to_id[name] in counts
        }

    def safe_under_strat(y):
        counts = Counter(y)
        return {
            name_to_id[name]: min(target, counts[name_to_id[name]])
            for name, target in under_names.items()
            if name in name_to_id and name_to_id[name] in counts
        }

    # 2. Pass the functions directly to the samplers
    over = SMOTE(sampling_strategy=safe_over_strat, random_state=42)
    under = RandomUnderSampler(sampling_strategy=safe_under_strat, random_state=42)

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if params.get("resampling", True):
            X_res, y_res = over.fit_resample(X_train, y_train)
            X_train, y_train = under.fit_resample(X_res, y_res)

        model = XGBClassifier(**xgb_params)

        # Apply weights only to the training subset of this fold
        fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}

        if params.get('use_weights', False):
            current_weights = compute_sample_weight('balanced', y_train)
            fit_params['sample_weight'] = current_weights

        model.fit(X_train, y_train, **fit_params)
        
        oof_preds[val_idx] = model.predict_proba(X_val)
        models.append(model)
        
        print(f"Fold {fold + 1} | Best Iter: {model.best_iteration} | Score: {1 - model.best_score:.4f}")

    return models, oof_preds

def run_training(X, y, class_names, params=None):
    if params is None:
        params = {}

    # 1. Handle Parameters
    # Important: Early stopping is removed here because we train on the 100% of data
    xgb_params = {
        'n_estimators': params.get('n_estimators', 1000),
        'learning_rate': params.get('learning_rate', 0.05),
        'max_depth': params.get('max_depth', 6),
        'objective': 'multi:softprob',
        'num_class': len(class_names),
        'tree_method': 'hist',
        'random_state': 42
    }

    # 2. Compute Weights
    full_weights = None
    if params.get('use_weights', False):
        full_weights = compute_sample_weight('balanced', y)

    # 3. Final Model Training
    print(f"Training final model on entire dataset ({len(X)} samples)...")
    model = XGBClassifier(**xgb_params)
    
    fit_params = {'verbose': True}
    if full_weights is not None:
        fit_params['sample_weight'] = full_weights

    model.fit(X, y, **fit_params)

    return model