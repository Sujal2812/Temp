import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.model_selection import (
    cross_val_score
)


def evaluate_model(
    model,
    X,
    y,
    predictions
):

    rmse = np.sqrt(

        mean_squared_error(
            y,
            predictions
        )

    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    r2 = r2_score(
        y,
        predictions
    )

    cv = (

        cross_val_score(

            model,

            X,

            y,

            cv=5,

            scoring="neg_root_mean_squared_error"

        )

        .mean()

        * -1

    )

    result = {

        "RMSE": rmse,

        "MAE": mae,

        "R2 Score": r2,

        "Cross Validation": cv

    }

    return result