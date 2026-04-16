import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from datasets.radar_dataset import MeshImagePairDataset, get_transforms
from models.dgcnn2 import ResNet2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_split',    required=True)
    p.add_argument('--feature_store', required=True)
    p.add_argument('--checkpoint_2d', required=True)
    p.add_argument('--num_classes',   type=int,   default=19)
    p.add_argument('--emb_dims',      type=int,   default=1024)
    p.add_argument('--k',             type=int,   default=20)
    p.add_argument('--dropout',       type=float, default=0.5)
    p.add_argument('--use_cls_head',  action='store_true')
    return p.parse_args()


def load_mapping(splits_dir):
    """
    Load class_mapping.txt → {label(int): orig_folder_name(str)}
    """
    mapping      = {}
    mapping_file = os.path.join(splits_dir, 'class_mapping.txt')
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            next(f)  # skip header
            next(f)  # skip separator
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mapping[int(parts[0])] = parts[1]
    else:
        print(f"WARNING: class_mapping.txt not found at {mapping_file}")
    return mapping


def main(args):
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits_dir = os.path.dirname(args.test_split)
    mapping    = load_mapping(splits_dir)
    print(f"Class mapping loaded: {mapping}")

    # ── Load feature store ────────────────────────────────────────────────
    with open(args.feature_store, 'rb') as f:
        feature_store = pickle.load(f)

    entries         = list(feature_store.values())
    gallery_vectors = torch.stack(
        [e['vector'] for e in entries]).to(device)    # [N, 256]
    gallery_labels  = [e['label']       for e in entries]
    gallery_paths   = [e['glb_path']    for e in entries]
    gallery_folders = [e['orig_folder'] for e in entries]

    print(f"Gallery size: {len(entries)} entries")

    # ── Load 2D encoder ───────────────────────────────────────────────────
    model_2d = ResNet2(args).to(device)
    model_2d.load_state_dict(
        torch.load(args.checkpoint_2d, map_location=device))
    model_2d.eval()
    print(f"Loaded 2D encoder from {args.checkpoint_2d}")

    # ── Test dataset ──────────────────────────────────────────────────────
    test_ds = MeshImagePairDataset(
        args.test_split,
        transform=get_transforms('test'),
        augment_pc=False)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"Test set size: {len(test_ds)} samples\n")

    top1_correct = 0
    top5_correct = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for idx, (_, img, label, orig_folder) in enumerate(test_loader):
            img         = img.to(device)
            label       = label.item()
            orig_folder = orig_folder[0]   # unwrap list from dataloader

            z_query, _  = model_2d(img)                          # [1, 256]
            scores      = (gallery_vectors @ z_query.T).squeeze() # [N]
            top5_idx    = scores.argsort(
                descending=True)[:5].cpu().tolist()

            pred_label  = gallery_labels[top5_idx[0]]
            top5_labels = [gallery_labels[i] for i in top5_idx]

            top1_correct += int(pred_label == label)
            top5_correct += int(label in top5_labels)
            total        += 1
            all_preds.append(pred_label)
            all_labels.append(label)

            # Preview first 5 queries with original folder names
            if idx < 5:
                true_name = mapping.get(label, orig_folder)
                print(f"Query {idx+1} | True class: folder {true_name} "
                      f"(label {label})")
                for rank, i in enumerate(top5_idx):
                    pred_name = mapping.get(gallery_labels[i],
                                            gallery_folders[i])
                    print(f"  Rank {rank+1}: folder {pred_name:>4} | "
                          f"score {scores[i]:.4f} | {gallery_paths[i]}")
                print()

    # ── Metrics ───────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Top-1 Accuracy: {top1_correct/total*100:.2f}%  "
          f"({top1_correct}/{total})")
    print(f"Top-5 Accuracy: {top5_correct/total*100:.2f}%  "
          f"({top5_correct}/{total})")

    # Per-class accuracy with original folder names
    cls_correct = defaultdict(int)
    cls_total   = defaultdict(int)
    for p, l in zip(all_preds, all_labels):
        cls_total[l]   += 1
        cls_correct[l] += int(p == l)

    print("\nPer-class Top-1 accuracy:")
    print(f"  {'Folder':>8}  {'Label':>6}  {'Acc':>8}  {'Correct/Total':>15}")
    print("  " + "-" * 45)
    for l in sorted(cls_total):
        acc       = cls_correct[l] / cls_total[l] * 100
        orig_name = mapping.get(l, str(l))
        print(f"  {orig_name:>8}  {l:>6}  {acc:>7.2f}%  "
              f"{cls_correct[l]:>6}/{cls_total[l]:<6}")

    # ── Confusion matrix ──────────────────────────────────────────────────
    display_labels = [mapping.get(i, str(i))
                      for i in range(args.num_classes)]

    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(args.num_classes)))
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=display_labels)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title('Confusion Matrix — Retrieval Mode\n'
                 '(labels show original class folder numbers)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_retrieval.png', dpi=150)
    print("\nSaved confusion_matrix_retrieval.png")


if __name__ == '__main__':
    main(parse_args())