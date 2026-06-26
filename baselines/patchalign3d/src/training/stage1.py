"""Stage 1: align 3D patch features to offline DINO patch features."""

import argparse
import datetime
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

from patchalign3d.datasets.trainset import TrainingSetDataset, collate_trainset
from patchalign3d.models import point_transformer

import wandb


class PatchToDinoProj(nn.Module):
    """Linear projection from encoder tokens to DINO feature space."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        # patch_emb: (B, C, G) -> (B, G, D)
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class DinoTargetCache:
    """LRU cache for canonical DINO patch targets.

    Expects files at:
      <data_root>/labeled/rendered/<item_id>/oriented/<subdir>/patch_features.pt
    """

    def __init__(self, root, subdir="patch_dino", capacity=512):
        self.root = Path(root)
        self.subdir = subdir
        self.capacity = max(16, int(capacity))
        self._store = OrderedDict()

    def _path_for(self, item_id):
        return self.root / "labeled" / "rendered" / item_id / "oriented" / self.subdir / "patch_features.pt"

    def get(self, item_id):
        if item_id in self._store:
            self._store.move_to_end(item_id)
            return self._store[item_id]
        path = self._path_for(item_id)
        obj = torch.load(path, map_location="cpu")
        centers_xyz = obj["centers_xyz"].float()  # (G,3)
        patch_feats = obj["patch_feats"].float()  # (G,D)
        d_dim = int(obj.get("feature_dim", patch_feats.shape[-1]))
        out = (centers_xyz, F.normalize(patch_feats, dim=-1), d_dim)
        self._store[item_id] = out
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_points(sample, device):
    pts = sample["points"]
    if not isinstance(pts, torch.Tensor):
        pts = torch.as_tensor(pts)
    if pts.ndim != 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[1] >= 3:
        y = pts[:, 1].clone()
        pts[:, 1] = pts[:, 2]
        pts[:, 2] = y
    return pts.t().unsqueeze(0).to(device, non_blocking=True).float()


def build_model(args, device):
    cfg = EasyDict(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        cls_dim=50,
        num_heads=6,
        group_size=args.group_size,
        num_group=args.num_group,
        encoder_dims=256,
        color=False,
        num_classes=16,
    )
    model = point_transformer.get_model(cfg).to(device)
    return model



def train_epoch_dino(model, proj, loader, optimizer, device, cache, augment=None, amp=True, scaler=None):
    model.train()
    proj.train()
    total_loss = 0.0
    total_batches = 0
    sum_cos = 0.0
    sum_ctr_dist = 0.0
    n_stats = 0
    for batch in tqdm(loader, total=len(loader), desc="Train (DINO stage1)", smoothing=0.9):
        optimizer.zero_grad(set_to_none=True)
        item_ids = [s.get("item_id", "") for s in batch]
        targets = [cache.get(iid) for iid in item_ids]
        pts = torch.cat([prepare_points(s, device) for s in batch], dim=0)
        if augment is not None:
            pts = augment(pts)
        with torch.cuda.amp.autocast(enabled=amp):
            patch_emb, patch_centers, _ = model.forward_patches(pts)
            pred = proj(patch_emb)
            pred = F.normalize(pred, dim=-1)
            B, G, _ = pred.shape
            batch_loss = 0.0
            for b in range(B):
                centers_canon, feats_canon, _ = targets[b]
                ctr_b = patch_centers[b].transpose(0, 1).contiguous()
                centers_c = centers_canon.to(device)
                dmat = torch.cdist(ctr_b.unsqueeze(0), centers_c.unsqueeze(0)).squeeze(0)
                map_idx = dmat.argmin(dim=-1)
                tgt = feats_canon.to(device)[map_idx]
                tgt = F.normalize(tgt, dim=-1)
                cos = (pred[b] * tgt).sum(dim=-1)
                batch_loss = batch_loss + (1.0 - cos).mean()
                with torch.no_grad():
                    dmin = dmat.min(dim=-1).values
                    sum_cos += cos.mean().item()
                    sum_ctr_dist += dmin.mean().item()
                    n_stats += 1
        if scaler is not None and amp:
            scaler.scale(batch_loss / max(1, len(batch))).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            (batch_loss / max(1, len(batch))).backward()
            optimizer.step()
        total_loss += (batch_loss.item() / max(1, len(batch)))
        total_batches += 1
    avg_loss = total_loss / max(1, total_batches)
    stats = {"cos_mean": (sum_cos / max(1, n_stats)), "center_dist_mean": (sum_ctr_dist / max(1, n_stats))}
    return avg_loss, stats


@torch.no_grad()
def eval_epoch_dino(model, proj, loader, device, *, cache: DinoTargetCache, amp: bool = True):
    model.eval()
    proj.eval()
    total_loss = 0.0
    batches = 0
    sum_cos = 0.0
    sum_ctr_dist = 0.0
    n_stats = 0
    for batch in tqdm(loader, total=len(loader), desc="Val (DINO stage1)", smoothing=0.9):
        item_ids = [s.get("item_id", "") for s in batch]
        targets = [cache.get(iid) for iid in item_ids]
        pts = torch.cat([prepare_points(s, device) for s in batch], dim=0)
        with torch.cuda.amp.autocast(enabled=amp):
            patch_emb, patch_centers, _ = model.forward_patches(pts)
            pred = proj(patch_emb)
            pred = F.normalize(pred, dim=-1)
            B, G, _ = pred.shape
            batch_loss = 0.0
            for b in range(B):
                centers_canon, feats_canon, _ = targets[b]
                ctr_b = patch_centers[b].transpose(0, 1).contiguous()
                centers_c = centers_canon.to(device)
                dmat = torch.cdist(ctr_b.unsqueeze(0), centers_c.unsqueeze(0)).squeeze(0)
                map_idx = dmat.argmin(dim=-1)
                tgt = feats_canon.to(device)[map_idx]
                tgt = F.normalize(tgt, dim=-1)
                cos = (pred[b] * tgt).sum(dim=-1)
                batch_loss = batch_loss + (1.0 - cos).mean()
                dmin = dmat.min(dim=-1).values
                sum_cos += cos.mean().item()
                sum_ctr_dist += dmin.mean().item()
                n_stats += 1
        total_loss += (batch_loss.item() / max(1, len(batch)))
        batches += 1
    avg_loss = total_loss / max(1, batches)
    stats = {"cos_mean": (sum_cos / max(1, n_stats)), "center_dist_mean": (sum_ctr_dist / max(1, n_stats))}
    return avg_loss, stats


def parse_args():
    p = argparse.ArgumentParser("Stage 1: DINO patch alignment on training data")
    # Model
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_group", type=int, default=128)
    # No color/normal channels in this setup
    # Data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--npoint", type=int, default=2048)
    p.add_argument("--min_per_label", type=int, default=32)
    p.add_argument("--random_sample_train", action="store_true", default=False)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--enc_learning_rate", type=float, default=1e-4)
    p.add_argument("--decay_rate", type=float, default=5e-2)
    p.add_argument("--train_encoder", action="store_true", default=False)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    # DINO cache
    p.add_argument("--dino_feature_subdir", type=str, default="patch_dino")
    p.add_argument("--dino_dim", type=int, default=768, help="Fallback DINO dim if not found in file")
    # Augmentation
    # Augmentation disabled for stage 1
    # Logging
    p.add_argument("--log_dir", type=str, default="logs/stage1")
    p.add_argument("--wandb_project", type=str, default="")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    run_name = (args.wandb_run_name or "").strip()
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path(args.log_dir) / (run_name if run_name else f"stage1_{timestr}")
    ckpt_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage1_dino")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(exp_dir / "log.txt"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    def log_string(s: str):
        print(s)
        logger.info(s)

    log_string(f"Save dir: {exp_dir}")

    train_set = TrainingSetDataset(
        root=args.data_root,
        split="train",
        id_list_file=args.train_list,
        npoints=args.npoint,
        min_per_label=args.min_per_label,
        seed=args.seed,
        random_subsample=args.random_sample_train,
    )
    val_set = TrainingSetDataset(
        root=args.data_root,
        split="val",
        id_list_file=args.val_list,
        npoints=args.npoint,
        min_per_label=args.min_per_label,
        seed=args.seed + 1000,
    )
    dl_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False, collate_fn=collate_trainset, pin_memory=True)
    dl_va = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_trainset, pin_memory=True)
    log_string(f"Train set: {len(train_set)} | val: {len(val_set)}")

    dino_dim = args.dino_dim
    cache_probe = DinoTargetCache(args.data_root, subdir=args.dino_feature_subdir)
    probe = cache_probe.get(train_set.samples[0])
    dino_dim = int(probe[2])
    log_string(f"DINO dim: {dino_dim}")

    model = build_model(args, device)
    proj = PatchToDinoProj(in_dim=384, out_dim=dino_dim).to(device)

    params = []
    if args.train_encoder:
        params.append({"params": model.parameters(), "lr": args.enc_learning_rate})
    params.append({"params": proj.parameters(), "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.decay_rate)

    augment = None
    cache = DinoTargetCache(args.data_root, subdir=args.dino_feature_subdir)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(args, "amp", True)))

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            mode=args.wandb_mode,
            name=args.wandb_run_name or None,
            config=vars(args),
        )

    def save_checkpoint(epoch_idx, name):
        path = ckpt_dir / f"{name}.pt"
        torch.save({"epoch": epoch_idx, "model": model.state_dict(), "proj": proj.state_dict(), "args": vars(args)}, path)
        log_string(f"[ckpt] saved to {path}")

    for epoch in range(1, args.epoch + 1):
        loss_tr, stats_tr = train_epoch_dino(model, proj, dl_tr, optimizer, device, cache=cache, augment=augment, amp=args.amp, scaler=scaler)
        log_string(f"Epoch {epoch:03d} | train loss {loss_tr:.4f} | cos {stats_tr['cos_mean']:.4f} | center_dist {stats_tr['center_dist_mean']:.4f}")
        if wandb_run:
            wandb_run.log({"train/loss": loss_tr, "train/cos": stats_tr["cos_mean"], "train/center_dist": stats_tr["center_dist_mean"], "epoch": epoch})
        if args.eval_every and (epoch % args.eval_every == 0):
            loss_va, stats_va = eval_epoch_dino(model, proj, dl_va, device, cache=cache, amp=args.amp)
            log_string(f"[val] loss {loss_va:.4f} | cos {stats_va['cos_mean']:.4f} | center_dist {stats_va['center_dist_mean']:.4f}")
            if wandb_run:
                wandb_run.log({"val/loss": loss_va, "val/cos": stats_va["cos_mean"], "val/center_dist": stats_va["center_dist_mean"], "epoch": epoch})
        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(epoch, f"epoch_{epoch:03d}")
    save_checkpoint(args.epoch, "last")
    if wandb_run:
        wandb_run.finish()
    log_string("Training finished.")


if __name__ == "__main__":
    main()
