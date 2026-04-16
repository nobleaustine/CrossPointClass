import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from datasets.radar_dataset import MeshImagePairDataset, get_transforms
from models.dgcnn import DGCNN, ResNet
from pytorch_metric_learning.losses import SupConLoss
from util import IOStream, AverageMeter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_split',  required=True)
    p.add_argument('--val_split',    required=True)
    p.add_argument('--exp_name',     default='sup_v1')
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--temperature',  type=float, default=0.1)
    p.add_argument('--emb_dims',     type=int,   default=1024)
    p.add_argument('--proj_dim',     type=int,   default=256)
    p.add_argument('--k',            type=int,   default=20)
    p.add_argument('--n_points',     type=int,   default=2048)
    p.add_argument('--lambda_cls',   type=float, default=0.5)
    p.add_argument('--num_classes',  type=int,   default=23)
    p.add_argument('--use_cls_head', action='store_true')
    p.add_argument('--dropout',      type=float, default=0.5)
    return p.parse_args()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = os.path.join('checkpoints', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    io = IOStream(os.path.join(ckpt_dir, 'run.log'))
    io.cprint(str(args))
    io.cprint(f"Device: {device}")

    wandb.init(project='SupervisedCrossPoint', name=args.exp_name)

    train_ds = MeshImagePairDataset(
        args.train_split,
        transform=get_transforms('train'),
        augment_pc=True)
    val_ds = MeshImagePairDataset(
        args.val_split,
        transform=get_transforms('val'),
        augment_pc=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=4,
        drop_last=True, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True)

    io.cprint(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model_3d = DGCNN(args).to(device)
    model_2d = ResNet(args).to(device)

    all_params = list(model_3d.parameters()) + list(model_2d.parameters())
    optimizer  = torch.optim.Adam(
        all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion_supcon = SupConLoss(temperature=args.temperature).to(device)
    criterion_ce     = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):

        # ── TRAIN ──────────────────────────────────────────────
        model_3d.train()
        model_2d.train()
        t_loss   = AverageMeter()
        t_supcon = AverageMeter()
        t_ce     = AverageMeter()

        for pc, img, labels, _ in train_loader:
            pc     = pc.permute(0, 2, 1).to(device)   # [B, 3, 2048]
            img    = img.to(device)
            labels = labels.to(device)

            z_3d         = model_3d(pc)
            z_2d, logits = model_2d(img)

            embeddings  = torch.cat([z_3d, z_2d], dim=0)
            labels_cat  = torch.cat([labels, labels], dim=0)

            loss_supcon = criterion_supcon(embeddings, labels_cat)
            loss_total  = loss_supcon
            loss_ce     = torch.tensor(0.0, device=device)

            if args.use_cls_head and logits is not None:
                loss_ce    = criterion_ce(logits, labels)
                loss_total = loss_supcon + args.lambda_cls * loss_ce

            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            b = pc.size(0)
            t_loss.update(loss_total.item(), b)
            t_supcon.update(loss_supcon.item(), b)
            t_ce.update(loss_ce.item(), b)

        scheduler.step()

        # ── VALIDATE ───────────────────────────────────────────
        model_3d.eval()
        model_2d.eval()
        v_loss   = AverageMeter()
        v_supcon = AverageMeter()
        v_ce     = AverageMeter()
        correct  = 0
        total    = 0

        with torch.no_grad():
            for pc, img, labels, _ in val_loader:
                pc     = pc.permute(0, 2, 1).to(device)
                img    = img.to(device)
                labels = labels.to(device)

                z_3d         = model_3d(pc)
                z_2d, logits = model_2d(img)

                embeddings = torch.cat([z_3d, z_2d], dim=0)
                labels_cat = torch.cat([labels, labels], dim=0)

                ls = criterion_supcon(embeddings, labels_cat)
                lt = ls
                lc = torch.tensor(0.0, device=device)

                if args.use_cls_head and logits is not None:
                    lc = criterion_ce(logits, labels)
                    lt = ls + args.lambda_cls * lc
                    preds    = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total   += labels.size(0)

                b = pc.size(0)
                v_loss.update(lt.item(), b)
                v_supcon.update(ls.item(), b)
                v_ce.update(lc.item(), b)

        val_acc = correct / total if total > 0 else 0.0

        log_msg = (
            f"Epoch {epoch:03d} | "
            f"T_loss {t_loss.avg:.4f} | T_supcon {t_supcon.avg:.4f} | "
            f"T_ce {t_ce.avg:.4f} | "
            f"V_loss {v_loss.avg:.4f} | V_supcon {v_supcon.avg:.4f} | "
            f"V_ce {v_ce.avg:.4f} | V_acc {val_acc:.4f}"
        )
        io.cprint(log_msg)

        wandb.log({
            'train_loss_total':  t_loss.avg,
            'train_loss_supcon': t_supcon.avg,
            'train_loss_ce':     t_ce.avg,
            'val_loss_total':    v_loss.avg,
            'val_loss_supcon':   v_supcon.avg,
            'val_loss_ce':       v_ce.avg,
            'val_top1_acc':      val_acc,
            'learning_rate':     scheduler.get_last_lr()[0],
            'epoch':             epoch,
        })

        if v_loss.avg < best_val_loss:
            best_val_loss = v_loss.avg
            torch.save(model_3d.state_dict(),
                       os.path.join(ckpt_dir, 'best_3d_encoder.pth'))
            torch.save(model_2d.state_dict(),
                       os.path.join(ckpt_dir, 'best_2d_encoder.pth'))
            io.cprint(f"  ✓ Saved best (val_loss={best_val_loss:.4f})")

    torch.save(model_3d.state_dict(),
               os.path.join(ckpt_dir, 'last_3d_encoder.pth'))
    torch.save(model_2d.state_dict(),
               os.path.join(ckpt_dir, 'last_2d_encoder.pth'))
    io.cprint("Training complete.")


if __name__ == '__main__':
    args = parse_args()
    train(args)