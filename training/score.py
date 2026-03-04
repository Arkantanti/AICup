import pandas as pd
import sklearn.metrics


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
) -> float:
    # Takes in two pandas dataframe and computes the Macro-averaged Average Precision Score

    # Ensure all required columns are present
    needed_columns = [
        "Clutter",
        "Cormorants",
        "Pigeons",
        "Ducks",
        "Geese",
        "Gulls",
        "Birds of Prey",
        "Waders",
        "Songbirds",
    ]

    # Reorder solution and submission columns/rows to match exactly
    solution = solution.loc[solution.index, needed_columns]
    submission = submission.loc[solution.index, needed_columns]

    # Compute the Average Precision score for all required columns
    bird_score = sklearn.metrics.average_precision_score(
        solution[needed_columns],
        submission[needed_columns],
        average='macro'
    )

    return bird_score