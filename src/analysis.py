import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    #print(summary)
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
    if top_n == -1:
        feat_imp.plot(kind='barh', color='skyblue')
    else:
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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

CLASS_NAMES = ["Clutter", "Cormorants", "Pigeons", "Ducks", "Geese", "Gulls", "Birds of Prey", "Waders", "Songbirds"]

import seaborn as sns

def plot_bird_feature(df, column_name, title=None, savefig=False, save_path=None):
    # Set the overall style to dark
    plt.style.use('dark_background')

    # Custom color palette matching the specific colors in your reference image
    # Blue, Orange, Green, Red, Purple, Brown, Pink, Grey, Yellow
    custom_colors = [
        "#0044ff", "#ff7700", "#11cc33", "#ee0000",
        "#8822ff", "#aa5500", "#ff44aa", "#999999", "#ffcc00"
    ]

    # Create figure with the specific dark navy background color
    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0f111a')
    ax.set_facecolor('#0f111a')

    sns.boxplot(
        data=df,
        x='bird_group',
        y=column_name,
        hue='bird_group',  # Maps colors to groups correctly
        legend=False,  # Removes the redundant legend
        palette=custom_colors,
        showfliers=False,  # Removes the outlier dots
        linewidth=1.5,
        width=0.5,
        ax=ax
    )

    # Formatting Labels and Title
    title_text = f'Boxplot bird groups X {column_name.replace("_", " ")}'
    if title is not None: title_text = f'{title}'
    ax.set_title(title_text, color='white', fontsize=14, pad=20)
    ax.set_xlabel('', fontsize=12)
    ax.set_ylabel('', fontsize=12)

    # Customizing Grid Lines (Horizontal only, faint grey)
    ax.yaxis.grid(True, linestyle='-', which='major', color='#333333', alpha=0.6)
    ax.xaxis.grid(False)

    # Remove outer spines for a clean floating look
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    # Rotate x-labels exactly like the reference image
    plt.xticks(rotation=-45, ha='left', color='white', fontsize=11)
    plt.yticks(color='white')

    plt.tight_layout()

    if savefig:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_bird_confusion_pretty(y_true, oof_preds):
    y_pred = np.argmax(oof_preds, axis=1)

    # 1. Create the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize='true')

    # 2. Convert to DataFrame for easier Seaborn handling
    df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)

    # 3. Set the aesthetic style
    sns.set_theme(style="white")
    plt.figure(figsize=(14, 11))

    # 4. Create Heatmap
    # annot=True adds the numbers; fmt='.2f' handles decimals
    # mask handles cases where you might want to hide zero-values
    ax = sns.heatmap(df_cm,
                     annot=True,
                     fmt='.2g',
                     cmap='YlGnBu',  # Modern color palette (Yellow-Green-Blue)
                     linewidths=0.5,
                     linecolor='lightgrey',
                     cbar_kws={'label': 'Recall (Percentage of Class Correct)'},
                     square=True)

    # 5. Beautify Labels
    plt.title("Bird Species Classification: Normalized Confusion Matrix", fontsize=18, pad=20, fontweight='bold')
    plt.xlabel("Predicted Species", fontsize=14, labelpad=15)
    plt.ylabel("Actual Species", fontsize=14, labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    plt.tight_layout()
    plt.show()

def plot_cumulative_importance(models, features, threshold=0.90):
    """
    Plots individual feature importance and the cumulative importance ratio.
    """
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    # 1. Prepare and Sort Data
    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    # 2. Calculate Cumulative Sum
    df['Cumulative'] = df['Importance'].cumsum() / df['Importance'].sum()

    # 3. Create the Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar Chart (Individual)
    ax1.bar(df['Feature'], df['Importance'], color='steelblue', alpha=0.7, label='Individual')
    ax1.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Line Chart (Cumulative) - Secondary Axis
    ax2 = ax1.twinx()
    ax2.plot(df['Feature'], df['Cumulative'], color='crimson', marker='o', markersize=4, label='Cumulative')
    ax2.set_ylabel('Cumulative Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.05)

    # 4. Add Threshold Line
    ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=2)

    # Find the index where threshold is met for the text label
    cutoff_idx = np.where(df['Cumulative'] >= threshold)[0][0]
    ax2.text(cutoff_idx, threshold - 0.05, f' {int(threshold * 100)}% Cutoff',
             color='green', fontweight='bold', ha='right')

    plt.title('Feature Importance with Cumulative Ratio', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.show()
