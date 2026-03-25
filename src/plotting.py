from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend("Agg")


def _finish_plot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_fold_auc(fold_aucs: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    folds = np.arange(1, len(fold_aucs) + 1)
    bars = ax.bar(folds, fold_aucs, color="#2F6690", edgecolor="#173753")
    ax.set_title("Fold AUC")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_xticks(folds)
    ax.set_ylim(min(fold_aucs) - 0.01, max(fold_aucs) + 0.01)

    for bar, score in zip(bars, fold_aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _finish_plot(output_path)


def plot_summary(fold_aucs: list[float], overall_auc: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [f"Fold {index}" for index in range(1, len(fold_aucs) + 1)] + ["Overall"]
    values = fold_aucs + [overall_auc]
    colors = ["#3A7CA5"] * len(fold_aucs) + ["#F4A259"]
    positions = np.arange(len(labels))

    ax.bar(positions, values, color=colors, edgecolor="#1D3557")
    ax.set_title("Cross-Validation Summary")
    ax.set_ylabel("AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(min(values) - 0.01, max(values) + 0.01)

    for position, score in zip(positions, values):
        ax.text(position, score + 0.0005, f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    _finish_plot(output_path)


def plot_feature_importance(
    importance_mapping: dict[str, float],
    output_path: Path,
    top_n: int = 20,
) -> None:
    if not importance_mapping:
        importance_mapping = {"no_feature_importance": 0.0}

    sorted_items = sorted(
        importance_mapping.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]
    features = [item[0] for item in sorted_items][::-1]
    scores = [item[1] for item in sorted_items][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features, scores, color="#6C9A8B", edgecolor="#2F3E46")
    ax.set_title(f"Top {len(features)} Feature Importances")
    ax.set_xlabel("Importance (gain)")
    _finish_plot(output_path)


def plot_prediction_distribution(
    predictions: np.ndarray,
    output_path: Path,
    title: str = "OOF Prediction Distribution",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(predictions, bins=40, color="#BC4749", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Frequency")
    _finish_plot(output_path)
