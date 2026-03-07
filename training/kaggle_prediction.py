import pandas as pd
import numpy as np

def generate_ensemble_submission(models, features_config, test_path='../data/test.csv', output_file='../submissions/submission.csv'):
    """
    Inputs a list of trained XGBoost models, averages their predictions,
    and saves the species probabilities to a CSV file.
    """
    # 1. Load and Preprocess Test Data
    test_df = pd.read_csv(test_path)
    track_ids = test_df['track_id'].values
    
    from pre_processing import df_transform_experimental, prepare_for_training
    transformed_test = df_transform_experimental(test_df, features_config, labels=False)
    X_test, _, _ = prepare_for_training(transformed_test, features_config)
    
    # 2. Collect probabilities from all models
    # We create a list of probability matrices (each is N_samples x 9_classes)
    all_probs = []
    for i, model in enumerate(models):
        print(f"Generating predictions for model {i+1}/{len(models)}...")
        probs = model.predict_proba(X_test)
        all_probs.append(probs)
    
    # 3. Average the probabilities (Soft Voting)
    # np.mean across the new 'model' axis
    avg_probs = np.mean(all_probs, axis=0)
    
    # 4. Create the submission DataFrame
    CLASS_NAMES = ["Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", 
                   "Gulls", "Birds of Prey", "Waders", "Songbirds"]
    
    submission = pd.DataFrame(avg_probs, columns=CLASS_NAMES)
    submission.insert(0, 'track_id', track_ids)
    
    # 5. Save to file
    submission.to_csv(output_file, index=False)
    print(f"Ensemble submission saved to: {output_file}")

def generate_submission(model, features_config, test_path='../data/test.csv', output_file='../submissions/submission.csv'):
    test_df = pd.read_csv(test_path)
    track_ids = test_df['track_id'].values
    
    from pre_processing import df_transform_experimental, prepare_for_training
    transformed_test = df_transform_experimental(test_df, features_config,labels=False)
    X_test, _, _ = prepare_for_training(transformed_test, features_config)
    
    probs = model.predict_proba(X_test)
    
    CLASS_NAMES = ["Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", 
                   "Gulls", "Birds of Prey", "Waders", "Songbirds"]
    
    submission = pd.DataFrame(probs, columns=CLASS_NAMES)
    submission.insert(0, 'track_id', track_ids)
    
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")