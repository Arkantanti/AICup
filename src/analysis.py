import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score


def find_hard_negatives(y_true, y_probs, class_names, top_n=5):
    # Flatten the errors: find where (Predicted Prob - Actual) is highest
    # (Actual is 1 for the correct class, 0 for others)
    y_true_oh = pd.get_dummies(y_true).reindex(columns=range(len(class_names)), fill_value=0).values
    errors = np.abs(y_probs - y_true_oh)
    
    # Get indices of the biggest mistakes
    flat_idx = np.argsort(errors.ravel())[-top_n:]
    row_indices, col_indices = np.unravel_index(flat_idx, errors.shape)
    
    print("\n--- Top Confidence Errors (Hard Negatives) ---")
    for r, c in zip(row_indices, col_indices):
        print(f"Track ID Index: {r} | True Class: {class_names[y_true[r]]}")
        print(f"Model confidently predicted {class_names[c]} with {y_probs[r, c]:.4f} prob.\n")

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_multiclass_calibration(y_true, y_probs, n_bins=10):
    """
    Creates a grid of calibration curves (one per species).
    Hard-coded for the 9 species in the AICup competition.
    """
    # Hard-coded bird species list
    class_names = [
        "Clutter", "Cormorants", "Pigeons", "Ducks", 
        "Geese", "Gulls", "Birds of Prey", "Waders", "Songbirds"
    ]
    
    n_classes = len(class_names)
    cols = 3
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, name in enumerate(class_names):
        ax = axes[i]
        
        # 1. Binary target for the specific species
        y_true_binary = (y_true == i).astype(int)
        
        # 2. Predicted probabilities for this species
        prob_pos = y_probs[:, i]
        
        # 3. Calculate curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, prob_pos, n_bins=n_bins
        )

        # Plotting logic
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect") 
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", color='tab:blue')
        
        ax.set_title(f"{name}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        if i % cols == 0:
            ax.set_ylabel("Actual Fraction")
        if i >= n_classes - cols:
            ax.set_xlabel("Predicted Prob")

    # Remove any extra empty subplots in the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Individual Species Calibration Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def debug_per_class_scores(y_true, y_probs):
    class_names =  ["Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", "Gulls", "Birds of Prey", "Waders", "Songbirds"]

    # One-hot encode truth
    y_true_oh = pd.get_dummies(y_true).reindex(columns=range(len(class_names)), fill_value=0)
    
    # Calculate scores
    aps = average_precision_score(y_true_oh, y_probs, average=None)
    
    # Create a summary table
    summary = pd.DataFrame({
        'Class': class_names,
        'AP_Score': aps,
        'Support': np.bincount(y_true, minlength=len(class_names))
    }).sort_values('AP_Score', ascending=False)
    
    print("\n--- Per-Class Performance ---")
    print(summary)
    return summary


def get_final_metrics(y_true_indices, y_probs, class_names):
    # One-hot encode truth for the AP calculation
    y_true_oh = pd.get_dummies(y_true_indices).reindex(columns=range(len(class_names)), fill_value=0)
    
    # Calculate Macro AP (The primary goal)
    # Using individual class scores to find weaknesses [cite: 2026-01-18]
    individual_aps = average_precision_score(y_true_oh, y_probs, average=None)
    macro_ap = np.mean(individual_aps)
    
    # Print results
    print(f"\n{'='*30}\nFINAL CROSS-VALIDATION MACRO AP: {macro_ap:.4f}\n{'='*30}")
    for name, ap in zip(class_names, individual_aps):
        print(f"{name:15}: {ap:.4f}")
    
    return macro_ap

def plot_feature_importance(models, features, top_n=15):
    """Averages importance across folds and plots top features."""
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    feat_imp.tail(top_n).plot(kind='barh', color='skyblue')
    plt.title(f'Top {top_n} Most Influential Features')
    plt.xlabel('Average Gain/Importance')
    plt.show()

def print_detailed_scores(final_score, detailed_scores):
    """Formatted printer for the macro AP results."""
    print(f"\nOverall Macro AP: {final_score:.4f}")
    print("-" * 30)
    for bird, score in detailed_scores.items():
        print(f"{bird:15} | AP: {score:.4f}")