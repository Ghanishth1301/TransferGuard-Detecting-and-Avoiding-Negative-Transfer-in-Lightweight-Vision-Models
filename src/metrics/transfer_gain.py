from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_transfer_gain(
    accuracy_transfer: float,
    accuracy_scratch: float,
) -> float:
    """
    Computes Transfer Gain (TG):
    TG = Accuracy(Transfer) - Accuracy(Scratch)
    
    Returns:
        Transfer Gain in percentage points.
    """
    return round(accuracy_transfer - accuracy_scratch, 3)


def evaluate_negative_transfer(
    accuracy_transfer: float,
    accuracy_scratch: float,
    threshold: float = 0.0,
) -> Dict[str, Union[float, bool, str]]:
    """
    Evaluates whether negative transfer demonstrably occurred.
    
    Returns:
        dict with keys:
            - transfer_gain: TG value
            - is_negative_transfer: boolean
            - severity: "None", "Mild", "Moderate", "Severe"
            - diagnosis: human-readable explanation
    """
    tg = compute_transfer_gain(accuracy_transfer, accuracy_scratch)
    is_negative = tg < -abs(threshold)

    if not is_negative:
        severity = "None (Positive Transfer)"
        diagnosis = f"Transfer learning provided a positive gain of +{tg:.2f}% over random initialization."
    elif tg >= -2.0:
        severity = "Mild"
        diagnosis = f"Mild negative transfer detected (TG = {tg:.2f}%). Target task is slightly degraded by source weights."
    elif tg >= -5.0:
        severity = "Moderate"
        diagnosis = f"Moderate negative transfer detected (TG = {tg:.2f}%). Significant representation conflict."
    else:
        severity = "Severe"
        diagnosis = f"Severe negative transfer detected (TG = {tg:.2f}%). Catastrophic domain mismatch."

    return {
        "accuracy_transfer": accuracy_transfer,
        "accuracy_scratch": accuracy_scratch,
        "transfer_gain": tg,
        "is_negative_transfer": is_negative,
        "severity": severity,
        "diagnosis": diagnosis,
    }


def build_benchmark_summary_table(
    results: Dict[str, Dict[str, float]]
) -> List[Dict[str, Union[str, float]]]:
    """
    Builds a structured benchmark summary comparing scratch vs methods.
    
    Expected format:
        results = {
            "SVHN": {"scratch": 88.5, "finetune": 84.2, "transferguard": 90.1},
            "CIFAR-100": {"scratch": 68.0, "finetune": 74.5, "transferguard": 75.8},
        }
    """
    table = []
    for dataset, scores in results.items():
        scratch_acc = scores.get("scratch", 0.0)
        for method, acc in scores.items():
            if method == "scratch":
                continue
            tg = compute_transfer_gain(acc, scratch_acc)
            table.append({
                "Dataset": dataset,
                "Method": method,
                "Accuracy (%)": acc,
                "Scratch Acc (%)": scratch_acc,
                "Transfer Gain (%)": tg,
                "Negative Transfer": "YES (Harmful)" if tg < 0 else "NO (Beneficial)",
            })
    return table

