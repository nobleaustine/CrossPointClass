import argparse
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              f1_score, classification_report)
from torch.utils.data import DataLoader

from datasets.radar_dataset import MeshImagePairDataset, get_transforms
from models.dgcnn import ResNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_split',    required=True)
    p.add_argument('--checkpoint_2d', required=True)
    p.add_argument('--num_classes',   type=int,   default=23)
    p.add_argument('--emb_dims',      type=int,   default=1024)
    p.add_argument('--k',             type=int,   default=20)
    p.add_argument('--dropout',       type=float, default=0.5)
    p.add_argument('--use_cls_head',  action='store_true', default=True)
    return p.parse_args()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_2d = ResNet(args).to(device)
    model_2d.load_state_dict(
        torch.load(args.checkpoint_2d, map_location=device))
    model_2d.eval()

    test_ds = MeshImagePairDataset(
        args.test_split,
        transform=get_transforms('test'),
        augment_pc=False)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2)

    top1_correct = 0
    top5_correct = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():
        for idx, (_, img, label) in enumerate(test_loader):
            img   = img.to(device)
            label = label.item()

            _, logits = model_2d(img)
            probs     = torch.softmax(logits, dim=1).squeeze()
            top5_idx  = probs.argsort(descending=True)[:5].cpu().tolist()

            pred_label  = top5_idx[0]
            top5_labels = top5_idx

            top1_correct += int(pred_label == label)
            top5_correct += int(label in top5_labels)
            total        += 1
            all_preds.append(pred_label)
            all_labels.append(label)

            if idx < 5:
                print(f"\nQuery {idx+1} | True class: {label+1}")
                for rank, i in enumerate(top5_idx):
                    print(f"  Rank {rank+1}: class {i+1} | "
                          f"confidence {probs[i]*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"Top-1 Accuracy: {top1_correct/total*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_correct/total*100:.2f}%")

    f1_macro    = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_per_cls  = f1_score(all_labels, all_preds, average=None,
                           labels=list(range(args.num_classes)))

    print(f"\nF1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print("\nPer-class F1:")
    for i, f1 in enumerate(f1_per_cls):
        print(f"  Class {i+1:>3}: {f1:.4f}")

    print("\nFull classification report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[str(i+1) for i in range(args.num_classes)]))

    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(args.num_classes)))
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(i+1) for i in range(args.num_classes)])
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title('Confusion Matrix — Classification Mode')
    plt.tight_layout()
    plt.savefig('confusion_matrix_classification.png', dpi=150)
    print("\nSaved confusion_matrix_classification.png")


if __name__ == '__main__':
    main(parse_args())