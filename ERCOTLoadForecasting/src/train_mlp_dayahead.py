"""Train a neural network (MLP) under the day-ahead protocol, for comparison.

WHY THIS EXISTS
---------------
Notebook 02 found that six very different model classes - naive, linear,
polynomial, random forest, gradient boosting, XGBoost - all landed within ~8%
of each other. That is the signature of a problem limited by its FEATURES, not
by the function class. Everything that followed (weather, then the honest
issuance protocol) moved the error far more than any algorithm swap did.

This script tests that reading directly: same features, same split, same
protocol as src/train_dayahead.py, but a multi-layer perceptron instead of
gradient boosting. If the clustering explanation is right, the MLP should land
near the GB model rather than beating it meaningfully.

WHAT IS DIFFERENT FROM THE TREE MODELS
--------------------------------------
Trees are invariant to feature scale; neural networks are not. Two scalers are
therefore needed, and leaving either out is the usual reason a first MLP on
load data trains badly:

- inputs: hour (0-23) and load (~30,000-80,000 MW) differ by three orders of
  magnitude, so gradients are dominated by the large columns
- target: ERCOT load is in tens of thousands of MW, which a default-initialised
  output layer cannot reach without a long warm-up

Both scalers are fitted INSIDE the cross-validation pipeline, so each fold's
scaler sees only that fold's training data and never its validation hours.

This model is a diagnostic. It is not served by the API and is not saved to
models/ - src/train_dayahead.py remains the production day-ahead model.
"""

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES_DAYAHEAD
from src.train_dayahead import build_dayahead_dataset

#REFERENCE NUMBERS FROM NOTEBOOK 06, ON THE SAME TEST YEAR
GB_TEST_MAE = 3073.5
NAIVE_48H_TEST_MAE = 3629.7


def train_mlp_dayahead():
    df = build_dayahead_dataset()

    #SAME CHRONOLOGICAL SPLIT AS NOTEBOOK 06 - NO RANDOM SPLITTING
    dev_df = df.loc[:"2024-12-31"]
    test_df = df.loc["2025-01-01":]

    X_dev = dev_df[FEATURES_DAYAHEAD]
    y_dev = dev_df["ERCOT"]

    X_test = test_df[FEATURES_DAYAHEAD]
    y_test = test_df["ERCOT"]

    print(f"development: {X_dev.shape}, test: {X_test.shape}")

    #SCALE INPUTS, THEN THE NETWORK; TransformedTargetRegressor SCALES y TOO
    #AND UNDOES IT ON predict(), SO THE METRICS BELOW ARE IN MW
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=600,
            early_stopping=False,
            random_state=42,
        )),
    ])

    model = TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )

    #TUNED ON DEVELOPMENT DATA ONLY - THE 2025 TEST YEAR IS NEVER USED TO CHOOSE
    #ANYTHING, EXACTLY AS IN NOTEBOOKS 02-06
    grid = {
        "regressor__mlp__hidden_layer_sizes": [(64,), (128, 64)],
        "regressor__mlp__alpha": [1e-4, 1e-2],
        "regressor__mlp__learning_rate_init": [1e-3],
    }

    search = GridSearchCV(
        model,
        grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    search.fit(X_dev, y_dev)

    print("best parameters:", search.best_params_)
    print("best dev CV MAE:", -search.best_score_)

    predictions = search.best_estimator_.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nMLP (day-ahead protocol) - Test")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)

    #THE POINT OF THE EXERCISE: WHERE DOES IT LAND AGAINST THE TREE MODEL?
    print(f"\nvs gradient boosting ({GB_TEST_MAE:.0f} MW): "
          f"{(mae - GB_TEST_MAE) / GB_TEST_MAE * 100:+.1f}% MAE")
    print(f"vs naive 48h baseline ({NAIVE_48H_TEST_MAE:.0f} MW): "
          f"{(mae - NAIVE_48H_TEST_MAE) / NAIVE_48H_TEST_MAE * 100:+.1f}% MAE")

    return search.best_estimator_


if __name__ == "__main__":
    train_mlp_dayahead()
