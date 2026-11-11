#!/usr/bin/env python3
"""
scripts/run_cka_analysis.py
Computes layer-wise Centered Kernel Alignment (CKA) between ImageNet pre-trained representations
and target domain representations, saving diagnostic heatmaps.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from tqdm import tqdm
from src.models.mobilenet_v3 import build_mobilenet_v3
from src.models.shufflenet_v2 import build_shufflenet_v2
from src.metrics.cka import compute_layer_cka_matrix, CenteredKernelAlignment
from src.data.dataset_loader import get_dataset_loaders
from src.utils.visualization import plot_cka_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description="Run Layer-wise CKA Analysis")
    parser.add_argument("--dataset", type=str, default="svhn", choices=["svhn", "cifar100", "flowers102", "synthetic"])
    parser.add_argument("--model", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small", "mobilenet_v3_large", "shufflenet_v2_x1_0"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=256, help="Number of samples to evaluate for CKA")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/cka")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔬 Starting CKA Feature Similarity Analysis")
    print(f"   Backbone: {args.model}")
    print(f"   Target Dataset: {args.dataset}")
    print(f"   Device: {device}")
    print(f"   Evaluation subset: {args.subset} samples")

    # 1. Load Data
    _, val_loader, num_classes = get_dataset_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        debug_subset=args.subset,
    )

    # 2. Build Models: (A) Pretrained Source Model vs (B) Scratch/Random Initialized Model
    if "mobilenet" in args.model:
        model_source = build_mobilenet_v3(variant=args.model, pretrained=True, num_classes=num_classes).to(device)
        model_target = build_mobilenet_v3(variant=args.model, pretrained=False, num_classes=num_classes).to(device)
    else:
        model_source = build_shufflenet_v2(variant=args.model, pretrained=True, num_classes=num_classes).to(device)
        model_target = build_shufflenet_v2(variant=args.model, pretrained=False, num_classes=num_classes).to(device)

    model_source.eval()
    model_target.eval()

    # 3. Collect activations
    feats_source_accum = {}
    feats_target_accum = {}

    print("📊 Extracting layer representations...")
    sample_count = 0
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Batches"):
            images = images.to(device)
            _, feats_s = model_source.forward_with_features(images)
            _, feats_t = model_target.forward_with_features(images)

            for k in feats_s.keys():
                feats_source_accum.setdefault(k, []).append(feats_s[k].cpu())
                feats_target_accum.setdefault(k, []).append(feats_t[k].cpu())

            sample_count += images.size(0)
            if sample_count >= args.subset:
                break

    feats_source = {k: torch.cat(v, dim=0)[:args.subset] for k, v in feats_source_accum.items()}
    feats_target = {k: torch.cat(v, dim=0)[:args.subset] for k, v in feats_target_accum.items()}

    # 4. Compute CKA Matrix
    print("🧮 Calculating Centered Kernel Alignment matrix...")
    matrix, layers_s, layers_t = compute_layer_cka_matrix(feats_source, feats_target, unbiased=True)

    print("\n📈 Diagonal Layer-to-Layer Alignment Scores:")
    for i, name in enumerate(layers_s):
        print(f"   Stage [{name:12s}]: CKA = {matrix[i, i]:.4f}")

    # 5. Save Heatmap Plot
    save_fig = os.path.join(args.output_dir, f"cka_{args.model}_{args.dataset}.png")
    plot_cka_heatmap(
        matrix=matrix,
        layers_x=layers_t,
        layers_y=layers_s,
        save_path=save_fig,
        title=f"CKA Representation Alignment: ImageNet vs {args.dataset.upper()} ({args.model})",
    )
    print(f"\n✅ Diagnostic CKA Heatmap saved to: {save_fig}")

    # 6. Save Structured CKA JSON for downstream publication plots
    cka_data = {
        "dataset": args.dataset,
        "model": args.model,
        "stages": layers_s,
        "diagonal_cka": [round(float(matrix[i, i]), 4) for i in range(len(layers_s))],
        "full_matrix": matrix.tolist(),
    }
    json_path = os.path.join(args.output_dir, f"cka_{args.model}_{args.dataset}.json")
    with open(json_path, "w") as f:
        json.dump(cka_data, f, indent=2)
    print(f"📄 Saved structured CKA alignment metrics to: {json_path}")


if __name__ == "__main__":
    main()
