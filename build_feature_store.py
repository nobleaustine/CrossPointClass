import os
import argparse
import pickle
import torch
import numpy as np
from models.dgcnn2 import DGCNN2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_split', required=True)
    p.add_argument('--val_split',   required=True)
    p.add_argument('--test_split',  required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--out',         default='feature_store.pkl')
    p.add_argument('--emb_dims',    type=int,   default=1024)
    p.add_argument('--k',           type=int,   default=20)
    p.add_argument('--dropout',     type=float, default=0.5)
    p.add_argument('--num_classes', type=int,   default=19)
    return p.parse_args()


def collect_unique_npy(split_files):
    """
    Read all split files and collect unique npy paths.
    Returns: {npy_path: (label, orig_folder)}
    """
    seen = {}
    for sf in split_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts       = line.split(',')
                npy_path    = parts[0]
                label       = int(parts[2])
                orig_folder = parts[3]
                if npy_path not in seen:
                    seen[npy_path] = (label, orig_folder)
    return seen


def load_mapping(out_dir):
    """
    Load class_mapping.txt → {label: orig_folder_name}
    """
    mapping      = {}
    mapping_file = os.path.join(out_dir, 'class_mapping.txt')
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            next(f)  # skip header
            next(f)  # skip separator
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    mapping[int(parts[0])] = parts[1]
    return mapping


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load 3D encoder
    model = DGCNN2(args).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded 3D encoder from {args.checkpoint}")

    # Collect all unique npy files across all splits
    npy_map = collect_unique_npy(
        [args.train_split, args.val_split, args.test_split])
    print(f"Unique GLBs to encode: {len(npy_map)}")

    # Load label → original folder mapping
    splits_dir = os.path.dirname(args.train_split)
    mapping    = load_mapping(splits_dir)
    print(f"Loaded class mapping: {mapping}")

    feature_store = {}
    class_counts  = {}

    with torch.no_grad():
        for i, (npy_path, (label, orig_folder)) in enumerate(npy_map.items()):
            try:
                pts   = np.load(npy_path).astype(np.float32)  # [2048, 3]
                pts_t = torch.from_numpy(pts).T.unsqueeze(0).to(device)
                                                               # [1, 3, 2048]
                z = model(pts_t).squeeze(0).cpu()             # [256]

                glb_path = npy_path.replace('.npy', '.glb')

                feature_store[npy_path] = {
                    'vector':      z,
                    'label':       label,
                    'orig_folder': orig_folder,
                    'npy_path':    npy_path,
                    'glb_path':    glb_path,
                }
                class_counts[orig_folder] = \
                    class_counts.get(orig_folder, 0) + 1

                if (i + 1) % 10 == 0:
                    print(f"  Encoded {i+1}/{len(npy_map)}")

            except Exception as e:
                print(f"  ERROR encoding {npy_path}: {e}")
                continue

    # Save feature store
    with open(args.out, 'wb') as f:
        pickle.dump(feature_store, f)

    print(f"\nFeature store saved to {args.out}")
    print(f"Total entries: {len(feature_store)}")
    print("\nEntries per class:")
    print(f"  {'Folder':>8}  {'Label':>8}  {'Count':>8}")
    print("  " + "-" * 30)
    for folder in sorted(class_counts, key=lambda x: int(x)):
        # find label for this folder
        lbl = [l for l, f in mapping.items() if f == folder]
        lbl_str = str(lbl[0]) if lbl else '?'
        print(f"  {folder:>8}  {lbl_str:>8}  {class_counts[folder]:>8}")


if __name__ == '__main__':
    main(parse_args())