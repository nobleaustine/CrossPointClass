import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    device = x.device  # ← use device of input, not hardcoded cuda:1
    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(
        0, 3, 1, 2).contiguous()
    return feature


class DGCNN2(nn.Module):
    """
    3D encoder for supervised cross-modal contrastive learning.
    Takes point cloud [B, 3, N] → outputs L2-normalized embedding [B, 256].
    """

    def __init__(self, args):
        super(DGCNN2, self).__init__()
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2))

        # Projection head: emb_dims*2 → 512 → 256 (L2 normalized)
        self.proj_head = nn.Sequential(
            nn.Linear(args.emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)          # [B, emb_dims*2]

        z = self.proj_head(x)                    # [B, 256]
        z = F.normalize(z, dim=1)                # unit vector
        return z


class ResNet2(nn.Module):
    """
    2D encoder for supervised cross-modal contrastive learning.
    Shared ResNet50 backbone with two heads:
      - proj_head : contrastive projection → L2-normalized [B, 256]
      - cls_head  : classification logits  → [B, num_classes] (optional)
    """

    def __init__(self, args):
        super(ResNet2, self).__init__()
        self.use_cls_head = getattr(args, 'use_cls_head', False)
        num_classes       = getattr(args, 'num_classes', 23)

        # Pretrained ResNet50, fc removed
        backbone = tv_models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # backbone output: [B, 2048, 1, 1]

        # Head A — contrastive projection
        self.proj_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
        )

        # Head B — classification (optional)
        if self.use_cls_head:
            self.cls_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        feat = self.backbone(x)                  # [B, 2048, 1, 1]
        feat = feat.view(feat.size(0), -1)       # [B, 2048]

        z = self.proj_head(feat)
        z = F.normalize(z, dim=1)                # [B, 256] unit vector

        if self.use_cls_head:
            logits = self.cls_head(feat)         # [B, num_classes]
            return z, logits

        return z, None