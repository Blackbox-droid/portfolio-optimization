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


def select_best_models(metrics_df: pd.DataFrame) -> dict[str, str]:
    selection = {}
    for asset, group in metrics_df.groupby("Asset"):
        best_row = group.sort_values(["Test_RMSE", "Dir_Accuracy"], ascending=[True, False]).iloc[0]
        selection[asset] = best_row["Model"]
    return selection


def _validation_window(n_obs: int) -> int:
    return max(12, min(24, n_obs // 5))


def walk_forward_supervised_predictions(
    full_data: pd.DataFrame,
    feature_cols: list[str],
    test_dates: list[pd.Timestamp],
    assets: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets = assets or ASSETS
    model_specs = build_model_specs()

    prediction_rows = []
    selection_rows = []

    for date in test_dates:
        history = full_data.loc[full_data.index < date].copy()
        current = full_data.loc[[date]].copy()

        if history.empty or current.empty:
            continue

        val_window = _validation_window(len(history))
        train_history = history.iloc[:-val_window]
        val_history = history.iloc[-val_window:]

        if train_history.empty:
            train_history = history.iloc[:-1]
            val_history = history.iloc[-1:]

        for asset in assets:
            best_model_name = None
            best_rmse = np.inf
            best_dir_acc = -np.inf

            X_train = train_history[feature_cols].values
            X_val = val_history[feature_cols].values
            y_train = train_history[asset].values
            y_val = val_history[asset].values

            for model_name in MODEL_ORDER:
                model_template, needs_scaling = model_specs[model_name]
                model = clone(model_template)

                if needs_scaling:
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(X_train)
                    Xva = scaler.transform(X_val)
                else:
                    Xtr = X_train
                    Xva = X_val

                model.fit(Xtr, y_train)
                val_pred = model.predict(Xva)
                val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
                dir_acc = float(np.mean(np.sign(val_pred) == np.sign(y_val)))

                if val_rmse < best_rmse or (np.isclose(val_rmse, best_rmse) and dir_acc > best_dir_acc):
                    best_model_name = model_name
                    best_rmse = val_rmse
                    best_dir_acc = dir_acc

            best_template, needs_scaling = model_specs[best_model_name]
            best_model = clone(best_template)
            X_full = history[feature_cols].values
            y_full = history[asset].values
            X_pred = current[feature_cols].values

            if needs_scaling:
                final_scaler = StandardScaler()
                X_full_fit = final_scaler.fit_transform(X_full)
                X_pred_fit = final_scaler.transform(X_pred)
            else:
                X_full_fit = X_full
                X_pred_fit = X_pred

            best_model.fit(X_full_fit, y_full)
            pred_value = float(best_model.predict(X_pred_fit)[0])
            actual_value = float(current[asset].iloc[0])

            prediction_rows.append(
                {
                    "Date": date,
                    "Asset": asset,
                    "Predicted_Log_Return": pred_value,
                    "Actual_Log_Return": actual_value,
                }
            )
            selection_rows.append(
                {
                    "Date": date,
                    "Asset": asset,
                    "Selected_Model": best_model_name,
                    "Validation_RMSE": best_rmse,
                    "Validation_Dir_Accuracy": best_dir_acc,
                    "Training_Window_Months": int(len(history)),
                    "Validation_Window_Months": int(len(val_history)),
                }
            )

    prediction_df = pd.DataFrame(prediction_rows)
    selection_df = pd.DataFrame(selection_rows)
    return prediction_df, selection_df


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
