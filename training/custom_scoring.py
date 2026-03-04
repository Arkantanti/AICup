import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def calculate_macro_ap(y_true_codes, y_prob_matrix, class_names):
    """
    Computes the Macro-Averaged Average Precision score.
    """
    # 1. One-hot encode the true labels to match probability matrix shape (N, 9)
    y_true_one_hot = pd.get_dummies(y_true_codes).reindex(
        columns=range(len(class_names)), 
        fill_value=0
    ).values

    # 2. Calculate AP for each class individually
    # This follows the 'Sweep' logic we discussed earlier
    individual_aps = average_precision_score(
        y_true_one_hot, 
        y_prob_matrix, 
        average=None  # Returns an array of 9 scores
    )

    # 3. Create a readable summary
    score_registry = dict(zip(class_names, individual_aps))
    
    # 4. Compute the Macro Average (The final competition metric)
    macro_ap = np.mean(individual_aps)
    
    return macro_ap, score_registry