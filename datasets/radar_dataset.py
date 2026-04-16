import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MeshImagePairDataset(Dataset):

    def __init__(self, split_file, transform=None, augment_pc=False):
        self.samples = []
        self.transform = transform
        self.augment_pc = augment_pc

        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                npy_path     = parts[0]
                image_path   = parts[1]
                label        = int(parts[2])
                orig_folder  = parts[3]          # original class folder name
                self.samples.append(
                    (npy_path, image_path, label, orig_folder))

    def __len__(self):
        return len(self.samples)

    def _augment_pointcloud(self, pts):
        noise = np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        noise = np.clip(noise, -0.05, 0.05)
        pts   = pts + noise
        scale = np.random.uniform(0.8, 1.2)
        pts   = pts * scale
        return pts

    def __getitem__(self, idx):
        npy_path, image_path, label, orig_folder = self.samples[idx]

        pts = np.load(npy_path).astype(np.float32)
        if self.augment_pc:
            pts = self._augment_pointcloud(pts)
        point_cloud = torch.from_numpy(pts)                # [2048, 3]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)                  # [3, 256, 256]

        return (point_cloud,
                image,
                torch.tensor(label, dtype=torch.long),
                orig_folder)


def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])