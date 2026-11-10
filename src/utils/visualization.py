import os
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional


def set_academic_style():
    """Sets a clean, publication-ready seaborn style."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 300,
    })


def plot_cka_heatmap(
    matrix: np.ndarray,
    layers_x: List[str],
    layers_y: List[str],
    save_path: str,
    title: str = "Layer-wise Centered Kernel Alignment (CKA)",
):
    """
    Renders a 2D CKA similarity heatmap between two models or layers.
    """
    set_academic_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 6.5))
    ax = sns.heatmap(
        matrix,
        xticklabels=layers_x,
        yticklabels=layers_y,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "CKA Similarity Score"},
        linewidths=0.5,
    )
    plt.title(title, pad=15, weight="bold")
    plt.xlabel("Target Domain Representations")
    plt.ylabel("Source Domain Pre-trained Representations")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_transfer_gain_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    title: str = "Transfer Gain (TG) Comparison Across Benchmarks",
):
    """
    Bar chart highlighting positive vs negative transfer zones.
    results = {
        "SVHN": {"Scratch": 88.5, "Standard Fine-Tune": 84.1, "TransferGuard (Ours)": 90.4},
        "CIFAR-100": {"Scratch": 67.2, "Standard Fine-Tune": 73.8, "TransferGuard (Ours)": 75.1},
        "Flowers-102": {"Scratch": 52.0, "Standard Fine-Tune": 86.4, "TransferGuard (Ours)": 87.2},
    }
    """
    set_academic_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    datasets = list(results.keys())
    categories = []
    gains = []
    dataset_labels = []

    for d in datasets:
        scratch = results[d].get("Scratch", 0.0)
        for method, acc in results[d].items():
            if method == "Scratch":
                continue
            tg = acc - scratch
            categories.append(f"{d}\n({method})")
            gains.append(tg)
            dataset_labels.append(d)

    colors = ["#d9534f" if g < 0 else "#5cb85c" for g in gains]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(categories, gains, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_ylabel("Transfer Gain TG (%) = Acc(Transfer) - Acc(Scratch)")
    ax.set_title(title, pad=15, weight="bold")

    # Annotate bar values
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.3 if h >= 0 else -0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + offset,
            f"{h:+.2f}%",
            ha="center",
            va=va,
            weight="bold",
            fontsize=10,
        )

    # Shaded negative transfer region
    min_y = min(min(gains) - 2.0, -3.0)
    max_y = max(max(gains) + 2.0, 5.0)
    ax.set_ylim(min_y, max_y)
    ax.axhspan(min_y, 0, color="#f8d7da", alpha=0.35, label="Negative Transfer Zone (TG < 0)")
    ax.axhspan(0, max_y, color="#d4edda", alpha=0.35, label="Positive Transfer Zone (TG > 0)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(
    histories: Dict[str, List[Dict[str, float]]],
    save_path: str,
    title: str = "Convergence Trajectory Comparison",
):
    """Plots training and validation accuracy over epochs for multiple methods."""
    set_academic_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for method, hist in histories.items():
        epochs = [r["epoch"] for r in hist]
        val_accs = [r.get("val_acc1", 0.0) for r in hist]
        train_losses = [r.get("train_loss", 0.0) for r in hist]

        ax1.plot(epochs, val_accs, marker="o", markersize=3, label=method, linewidth=2)
        ax2.plot(epochs, train_losses, linestyle="--", label=method, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Top-1 Accuracy (%)")
    ax1.set_title("Validation Accuracy vs. Epoch", weight="bold")
    ax1.legend()

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Training Loss vs. Epoch", weight="bold")
    ax2.legend()

    fig.suptitle(title, weight="bold", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_layer_freezing_profile(
    layer_scores: Dict[str, float],
    freeze_threshold: float,
    bypass_threshold: float,
    save_path: str,
    title: str = "TransferGuard Stage Transferability & Routing Decisions",
):
    """Visualizes layer-wise transferability scores and assigned routing actions."""
    set_academic_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    stages = list(layer_scores.keys())
    scores = [layer_scores[s] for s in stages]

    colors = []
    for s in scores:
        if s >= freeze_threshold:
            colors.append("#2b8a3e")  # Green = Freeze pre-trained
        elif s <= bypass_threshold:
            colors.append("#c92a2a")  # Red = Bypass harmful
        else:
            colors.append("#e67700")  # Orange = Plastic tune

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(stages, scores, color=colors, width=0.5, edgecolor="black")

    ax.axhline(freeze_threshold, color="#2b8a3e", linestyle="--", linewidth=1.5, label=f"Freeze Threshold (τ={freeze_threshold})")
    ax.axhline(bypass_threshold, color="#c92a2a", linestyle=":", linewidth=1.5, label=f"Bypass Threshold (τ={bypass_threshold})")

    ax.set_ylabel("Transferability Score S_l in [0, 1]")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, pad=15, weight="bold")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            weight="bold",
            fontsize=10,
        )

    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

