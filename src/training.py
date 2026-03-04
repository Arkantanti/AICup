import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

def run_cv_training(X, y, class_names, params=None):
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
        'random_state': 42
    }

    oof_preds = np.zeros((len(X), len(class_names)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    # 2. Compute Full Weights if requested in config
    full_weights = None
    if params.get('use_weights', False):
        full_weights = compute_sample_weight('balanced', y)

    # 3. Training Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**xgb_params)

        # Apply weights only to the training subset of this fold
        fit_params = {'eval_set': [(X_val, y_val)], 'verbose': False}
        if full_weights is not None:
            fit_params['sample_weight'] = full_weights[train_idx]

        model.fit(X_train, y_train, **fit_params)
        
        oof_preds[val_idx] = model.predict_proba(X_val)
        models.append(model)
        
        print(f"Fold {fold + 1} | Best Iter: {model.best_iteration} | Score: {model.best_score:.4f}")

    return models, oof_preds