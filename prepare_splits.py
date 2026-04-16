import os
import argparse
import numpy as np
import trimesh


def load_glb_and_sample(glb_path, n_points=2048, seed=42):
    scene = trimesh.load(glb_path, force='scene')
    if isinstance(scene, trimesh.Scene):
        geometries = list(scene.geometry.values())
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in geometries
        ])
    else:
        mesh = scene
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = pts.astype(np.float32)
    pts -= pts.mean(axis=0)
    pts /= np.linalg.norm(pts, axis=1).max()
    return pts


def main(args):
    os.makedirs(args.npy_root, exist_ok=True)
    os.makedirs(args.out_dir,  exist_ok=True)

    # Discover folders actually present, sorted numerically
    class_folders = sorted(
        [f for f in os.listdir(args.mesh_root)
         if os.path.isdir(os.path.join(args.mesh_root, f))],
        key=lambda x: int(x)
    )

    print(f"Found {len(class_folders)} class folders: {class_folders}")

    # First pass — sample all GLBs and collect triplets per class
    all_triplets_per_class = {}

    for cls_folder in class_folders:
        mesh_cls_dir  = os.path.join(args.mesh_root,  cls_folder)
        image_cls_dir = os.path.join(args.image_root, cls_folder)
        npy_cls_dir   = os.path.join(args.npy_root,   cls_folder)
        os.makedirs(npy_cls_dir, exist_ok=True)

        glb_files = [f for f in os.listdir(mesh_cls_dir)
                     if f.lower().endswith('.glb')]

        triplets = []

        for glb_file in glb_files:
            glb_stem = os.path.splitext(glb_file)[0]
            glb_path = os.path.join(mesh_cls_dir, glb_file)
            npy_path = os.path.join(npy_cls_dir, glb_stem + '.npy')

            if not os.path.exists(npy_path):
                print(f"  Sampling {glb_path} ...")
                try:
                    pts = load_glb_and_sample(glb_path)
                    np.save(npy_path, pts)
                    print(f"  Saved {npy_path}")
                except Exception as e:
                    print(f"  ERROR on {glb_path}: {e}")
                    continue
            else:
                print(f"  Skipping {npy_path} (exists)")

            if not os.path.isdir(image_cls_dir):
                continue

            image_files = os.listdir(image_cls_dir)
            matched = [
                f for f in image_files
                if f.startswith(glb_stem + '_')
                and f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            for img_file in matched:
                img_path = os.path.join(image_cls_dir, img_file)
                triplets.append((
                    os.path.abspath(npy_path),
                    os.path.abspath(img_path),
                    cls_folder   # store folder name, label assigned after filtering
                ))

        all_triplets_per_class[cls_folder] = triplets
        print(f"  Class folder {cls_folder}: {len(triplets)} triplets")

    # Second pass — filter out classes with 0 triplets and remap labels
    valid_folders = [
        f for f in class_folders
        if len(all_triplets_per_class[f]) > 0
    ]
    skipped_folders = [
        f for f in class_folders
        if len(all_triplets_per_class[f]) == 0
    ]

    print(f"\nSkipping folders with no image pairs: {skipped_folders}")
    print(f"Valid folders ({len(valid_folders)}): {valid_folders}")

    # Assign contiguous labels only to valid folders
    folder_to_label = {folder: idx for idx, folder in enumerate(valid_folders)}
    num_classes     = len(valid_folders)

    # Save class mapping file
    mapping_path = os.path.join(args.out_dir, 'class_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write(f"{'Label':<8} {'Folder':<10} {'OriginalClass':<15}\n")
        f.write("-" * 35 + "\n")
        for idx, folder in enumerate(valid_folders):
            f.write(f"{idx:<8} {folder:<10} {folder:<15}\n")
    print(f"\nClass mapping saved to {mapping_path}")
    print(f"\n{'Label':<8} {'Folder':<10}")
    for idx, folder in enumerate(valid_folders):
        print(f"  {idx:<8} {folder:<10}")

    # Rebuild triplets with correct contiguous labels
    train_all, val_all, test_all = [], [], []
    summary_lines = [
        f"{'Folder':<10} {'Label':<8} {'Train':>8} "
        f"{'Val':>8} {'Test':>8} {'Total':>8}"
    ]

    for cls_folder in valid_folders:
        label    = folder_to_label[cls_folder]
        triplets = [
            (npy_p, img_p, label, orig_folder)
            for npy_p, img_p, orig_folder in all_triplets_per_class[cls_folder]
        ]

        np.random.seed(42)
        indices = np.random.permutation(len(triplets))
        n       = len(triplets)
        n_train = int(0.70 * n)
        n_val   = int(0.90 * n)

        train = [triplets[i] for i in indices[:n_train]]
        val   = [triplets[i] for i in indices[n_train:n_val]]
        test  = [triplets[i] for i in indices[n_val:]]

        train_all.extend(train)
        val_all.extend(val)
        test_all.extend(test)

        summary_lines.append(
            f"{cls_folder:<10} {label:<8} {len(train):>8} "
            f"{len(val):>8} {len(test):>8} {n:>8}"
        )

    summary_lines.append("-" * 55)
    summary_lines.append(
        f"{'TOTAL':<10} {'':<8} {len(train_all):>8} "
        f"{len(val_all):>8} {len(test_all):>8} "
        f"{len(train_all)+len(val_all)+len(test_all):>8}"
    )

    # Write split files
    def write_split(path, triplets):
        with open(path, 'w') as f:
            for npy_p, img_p, lbl, orig_folder in triplets:
                f.write(f"{npy_p},{img_p},{lbl},{orig_folder}\n")

    write_split(os.path.join(args.out_dir, 'train.txt'), train_all)
    write_split(os.path.join(args.out_dir, 'val.txt'),   val_all)
    write_split(os.path.join(args.out_dir, 'test.txt'),  test_all)

    summary_path = os.path.join(args.out_dir, 'split_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print('\n' + '\n'.join(summary_lines))
    print(f"\nSplits written to {args.out_dir}")
    print(f"Total valid classes: {num_classes}")
    print(f"Skipped classes (no images): {skipped_folders}")
    print(f"Use --num_classes {num_classes} in all training and test commands")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_root',  required=True)
    parser.add_argument('--image_root', required=True)
    parser.add_argument('--npy_root',   required=True)
    parser.add_argument('--out_dir',    required=True)
    args = parser.parse_args()
    main(args)