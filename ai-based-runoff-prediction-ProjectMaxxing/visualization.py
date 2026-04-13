import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve


sns.set_theme(style="whitegrid", context="talk")


def _safe_model_name(model):
    return model.__class__.__name__.lower()


def _to_numpy(values):
    if hasattr(values, "values"):
        return np.asarray(values.values).ravel()
    return np.asarray(values).ravel()


def _to_2d_features(X):
    if hasattr(X, "select_dtypes"):
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] == 0:
            return np.asarray(X), [f"feature_{i}" for i in range(np.asarray(X).shape[1])]
        return numeric_X.values, list(numeric_X.columns)

    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr, [f"feature_{i}" for i in range(arr.shape[1])]


def _prepare_paths(model_name):
    plot_dir = Path("reports") / "plots" / model_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def _update_model_registry(model_name, plot_paths):
    registry_path = Path("model_registry.json")
    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    registry[model_name] = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "plots": plot_paths,
    }

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def generate_all_plots(y_true, y_pred, model, X):
    """
    Generate evaluation plots and return list of saved PNG paths.

    Required signature:
        generate_all_plots(y_true, y_pred, model, X)
    """
    model_name = _safe_model_name(model)
    plot_dir = _prepare_paths(model_name)

    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)
    residuals = y_true_arr - y_pred_arr

    X_arr, feature_names = _to_2d_features(X)

    saved_paths = []

    # 1) Actual vs Predicted line plot
    plt.figure(figsize=(14, 6))
    sample_size = min(300, len(y_true_arr))
    plt.plot(y_true_arr[:sample_size], label="Actual", linewidth=2.2, color="#1f77b4")
    plt.plot(y_pred_arr[:sample_size], label="Predicted", linewidth=2.0, linestyle="--", color="#ff7f0e")
    plt.title(f"Actual vs Predicted ({model.__class__.__name__})", fontweight="bold")
    plt.xlabel("Sample Index")
    plt.ylabel("Runoff")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    p1 = plot_dir / "actual_vs_predicted.png"
    plt.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close()
    saved_paths.append(str(p1.as_posix()))

    # 2) Residual distribution plot
    plt.figure(figsize=(11, 6))
    sns.histplot(residuals, kde=True, bins=40, color="#6a5acd", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero error")
    plt.title(f"Residual Distribution ({model.__class__.__name__})", fontweight="bold")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.30)
    plt.tight_layout()
    p2 = plot_dir / "residual_distribution.png"
    plt.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close()
    saved_paths.append(str(p2.as_posix()))

    # 3) Feature importance (RF / XGB or any model exposing feature_importances_)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        top_k = min(15, len(importances))
        order = np.argsort(importances)[::-1][:top_k]
        names = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in order]
        vals = importances[order]

        plt.figure(figsize=(12, 7))
        sns.barplot(x=vals, y=names, hue=names, palette="viridis", legend=False)
        plt.title(f"Top Feature Importance ({model.__class__.__name__})", fontweight="bold")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        p3 = plot_dir / "feature_importance.png"
        plt.savefig(p3, dpi=220, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(p3.as_posix()))

    # 4) Training vs Validation loss (LSTM / keras models with history)
    history_obj = getattr(model, "history", None)
    history_dict = getattr(history_obj, "history", None)
    if isinstance(history_dict, dict) and "loss" in history_dict:
        plt.figure(figsize=(11, 6))
        plt.plot(history_dict.get("loss", []), label="Training Loss", linewidth=2.0)
        if "val_loss" in history_dict:
            plt.plot(history_dict.get("val_loss", []), label="Validation Loss", linewidth=2.0)
        plt.title(f"Training vs Validation Loss ({model.__class__.__name__})", fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.30)
        plt.tight_layout()
        p4 = plot_dir / "train_val_loss_curve.png"
        plt.savefig(p4, dpi=220, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(p4.as_posix()))

    # 5) Correlation heatmap
    if X_arr.ndim == 2 and X_arr.shape[1] >= 2:
        corr_cols = feature_names[: min(len(feature_names), 20)]
        corr_data = X_arr[:, : len(corr_cols)]
        corr = np.corrcoef(corr_data, rowvar=False)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            xticklabels=corr_cols,
            yticklabels=corr_cols,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(f"Feature Correlation Heatmap ({model.__class__.__name__})", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        p5 = plot_dir / "correlation_heatmap.png"
        plt.savefig(p5, dpi=220, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(p5.as_posix()))

    # 6) Learning curve (sklearn estimators)
    try:
        if X_arr.ndim == 2 and X_arr.shape[0] > 50 and hasattr(model, "fit"):
            lc_out = learning_curve(
                model,
                X_arr,
                y_true_arr,
                cv=3,
                scoring="neg_root_mean_squared_error",
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=1,
            )
            train_sizes, train_scores, val_scores = lc_out[:3]

            train_rmse = -train_scores.mean(axis=1)
            val_rmse = -val_scores.mean(axis=1)

            plt.figure(figsize=(11, 6))
            plt.plot(train_sizes, train_rmse, marker="o", linewidth=2, label="Train RMSE")
            plt.plot(train_sizes, val_rmse, marker="s", linewidth=2, label="Validation RMSE")
            plt.title(f"Learning Curve ({model.__class__.__name__})", fontweight="bold")
            plt.xlabel("Training Samples")
            plt.ylabel("RMSE")
            plt.legend()
            plt.grid(True, alpha=0.30)
            plt.tight_layout()
            p6 = plot_dir / "learning_curve.png"
            plt.savefig(p6, dpi=220, bbox_inches="tight")
            plt.close()
            saved_paths.append(str(p6.as_posix()))
    except Exception:
        pass

    _update_model_registry(model_name, saved_paths)
    return saved_paths
