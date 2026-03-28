import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import ASSETS, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, RANDOM_STATE


def build_model_specs() -> dict[str, tuple[object, bool]]:
    return {
        "LinearRegression": (LinearRegression(), True),
        "SVR": (SVR(kernel="rbf", C=1.0, epsilon=0.005, gamma="scale"), True),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }


def evaluate_supervised_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    assets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = assets or ASSETS
    model_specs = build_model_specs()

    X_train = train_data[feature_cols].values
    X_test = test_data[feature_cols].values

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    metrics_rows = []
    prediction_table = pd.DataFrame(index=test_data.index)

    for asset in assets:
        y_train = train_data[asset].values
        y_test = test_data[asset].values
        prediction_table[f"{asset}_actual"] = y_test

        for model_name in MODEL_ORDER:
            model_template, needs_scaling = model_specs[model_name]
            model = clone(model_template)
            Xtr = Xs_train if needs_scaling else X_train
            Xte = Xs_test if needs_scaling else X_test

            model.fit(Xtr, y_train)
            train_pred = model.predict(Xtr)
            test_pred = model.predict(Xte)

            metrics_rows.append(
                {
                    "Asset": asset,
                    "Model": model_name,
                    "Train_RMSE": np.sqrt(mean_squared_error(y_train, train_pred)),
                    "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_pred)),
                    "Train_R2": r2_score(y_train, train_pred),
                    "Test_R2": r2_score(y_test, test_pred),
                    "Test_MAE": mean_absolute_error(y_test, test_pred),
                    "Dir_Accuracy": float(np.mean(np.sign(test_pred) == np.sign(y_test))),
                }
            )
            prediction_table[f"{asset}_{model_name}"] = test_pred

    metrics_df = pd.DataFrame(metrics_rows)
    prediction_table.index.name = "Date"
    return metrics_df, prediction_table.reset_index()


def plot_supervised_model_comparison(metrics_df: pd.DataFrame, output_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(ASSETS))
    width = 0.18

    for idx, model_name in enumerate(MODEL_ORDER):
        subset = (
            metrics_df[metrics_df["Model"] == model_name]
            .set_index("Asset")
            .reindex(ASSETS)
        )
        axes[0].bar(
            x + idx * width,
            subset["Test_RMSE"].values,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            alpha=0.85,
            edgecolor="white",
        )
        axes[1].bar(
            x + idx * width,
            subset["Dir_Accuracy"].values * 100,
            width,
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            alpha=0.85,
            edgecolor="white",
        )

    axes[0].set_xticks(x + 1.5 * width)
    axes[0].set_xticklabels(ASSETS, rotation=15)
    axes[0].set_ylabel("Test RMSE")
    axes[0].set_title("Test RMSE by Asset and Model", fontweight="bold")

    axes[1].set_xticks(x + 1.5 * width)
    axes[1].set_xticklabels(ASSETS, rotation=15)
    axes[1].set_ylabel("Directional Accuracy (%)")
    axes[1].axhline(50, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].set_title("Directional Accuracy by Asset and Model", fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)

    plt.suptitle("Supervised Return Prediction Summary (Test: 2025-2026)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
