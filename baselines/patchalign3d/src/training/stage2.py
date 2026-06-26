"""Stage 2: CLIP-guided patch-text training on the training set (chairs) with multi-label BCE."""

import argparse
import datetime
import logging
import os
import random
import re
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
from patchalign3d.datasets.shapenet import PartNormalDataset
from patchalign3d.models import point_transformer
from patchalign3d.inference import eval as eval_tools

import open_clip
import wandb

# Prompt templates (part-only focus)
PART_ONLY_TEMPLATES = [
    "{}",
    "a {}",
    "{} part",
]
PART_PLUS_CAT_TEMPLATES = [
    "a {} of a {}", "the {} of a {}",   "{} of {}",
    "a {} part of a {}",
]

def _clean_text(s):
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class LearnableTemp(nn.Module):
    def __init__(self, init_tau=0.07, mode="exp", init_linear=np.log(10.0), max_scale=100.0):
        super().__init__()
        self.mode = mode
        self.max_scale = float(max_scale)
        if mode == "linear":
            self.scale = nn.Parameter(torch.tensor(float(init_linear), dtype=torch.float32))
            self.log_scale = None
        else:
            init_scale = 1.0 / max(init_tau, 1e-6)
            self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
            self.scale = None

    def forward(self):
        if self.mode == "linear":
            return torch.clamp(self.scale, max=self.max_scale)
        return self.log_scale.exp().clamp(max=self.max_scale)


class LRUTextCache:
    """On-the-fly CLIP text encoder with LRU caching."""

    def __init__(self, device, capacity=20000, clip_model=None, tokenizer=None, text_dim=512):
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.capacity = max(1024, int(capacity))
        self.text_dim = int(text_dim)
        self._store = OrderedDict()

    def _touch(self, key):
        self._store.move_to_end(key)

    @torch.no_grad()
    def encode_label(self, name):
        key = _clean_text(name)
        if key in self._store:
            self._touch(key)
            return self._store[key]
        texts = [tpl.format(key) for tpl in PART_ONLY_TEMPLATES]
        toks = self.tokenizer(texts).to(self.device)
        feat = self.clip_model.encode_text(toks)
        feat = F.normalize(feat, dim=-1)
        out = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1).squeeze(0).float().cpu()
        self._store[key] = out
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return out

    @torch.no_grad()
    def encode_label_for_sample(self, name, category, setting):
        key_name = _clean_text(name)
        key_cat = _clean_text(category or "")
        cache_key = f"{key_name}||{key_cat}||{setting}"
        if cache_key in self._store:
            self._touch(cache_key)
            return self._store[cache_key]
        texts = []
        if setting in ("part_only", "ensemble") or not key_cat:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(key_name) if tpl.count("{}") == 1 else key_name)
        if not texts:
            texts = [key_name]
        toks = self.tokenizer(texts).to(self.device)
        feat = self.clip_model.encode_text(toks)
        feat = F.normalize(feat, dim=-1)
        out = F.normalize(feat.mean(dim=0, keepdim=True), dim=-1).squeeze(0).float().cpu()
        self._store[cache_key] = out
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return out

    def encode_labels_for_sample(self, names, category, setting):
        if not names:
            return torch.empty(0, self.text_dim, device=self.device)
        vecs = [self.encode_label_for_sample(n, category, setting) for n in names]
        return torch.stack(vecs, dim=0).to(self.device)


class BankTextCache:
    """Uses offline text banks saved as torch files (keys, emb)."""

    def __init__(self, po, strict=False, fallback=None):
        self.po = po
        self.strict = bool(strict)
        self.fallback = fallback
        self.text_dim = int(po.dim) if po is not None else (fallback.text_dim if fallback else 512)

    def _lookup_po(self, name):
        if self.po is None:
            return None
        key = _clean_text(name)
        idx = self.po.key_to_idx.get(key, None)
        if idx is None:
            return None
        return self.po.emb[idx].to(self.po.emb.device)

    def has_key_for_sample(self, name, category, setting):
        if setting not in ("part_only", "ensemble"):
            return False
        n = _clean_text(name)
        if not n:
            return False
        return (self.po is not None) and (n in self.po.key_to_idx)

    def encode_labels_for_sample(self, names, category, setting):
        if not names:
            return torch.empty(0, self.text_dim, device=self.po.emb.device if self.po else "cpu")
        vecs = []
        for n in names:
            v = self._lookup_po(n)
            if v is None and self.fallback is not None:
                v = self.fallback.encode_label_for_sample(n, category, setting)
            if v is None:
                if self.strict:
                    raise KeyError(f"Label not found in part-only bank: name='{n}'")
                v = torch.zeros(self.text_dim, device=self.po.emb.device if self.po else "cpu")
            vecs.append(v)
        return torch.stack(vecs, dim=0)


class LoadedTextBank:
    def __init__(self, keys, emb, key_to_idx, dim, meta):
        self.keys = keys
        self.emb = emb
        self.key_to_idx = key_to_idx
        self.dim = dim
        self.meta = meta


def _load_text_bank_file(path):
    obj = torch.load(path, map_location="cpu")
    meta = dict(obj.get("meta", {}))
    keys = list(obj["keys"])
    emb = obj["emb"]
    if emb.dtype != torch.float32:
        emb = emb.float()
    dim = int(meta.get("text_dim", emb.shape[1]))
    key_to_idx = {k: i for i, k in enumerate(keys)}
    return LoadedTextBank(keys=keys, emb=emb.contiguous(), key_to_idx=key_to_idx, dim=dim, meta=meta)


def try_load_text_banks(bank_dir):
    if not bank_dir:
        return None
    root = Path(bank_dir)
    if not root.exists():
        return None
    merged_po = sorted(root.glob("textbank_part_only_*_merged.pt"))
    if merged_po:
        return _load_text_bank_file(merged_po[-1])
    return None


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


def compute_patch_targets_vector(point_labels, patch_idx, num_labels):
    G, M = patch_idx.shape
    gathered = point_labels.gather(0, patch_idx.reshape(-1)).view(G, M)
    lbl = (gathered + 1).clamp_min(0)
    K = int(num_labels) + 1
    one_hot = F.one_hot(lbl.clamp_max(K - 1), num_classes=K).sum(dim=1)
    one_hot[:, 0] = 0
    has_any = one_hot.sum(dim=1) > 0
    preds = one_hot.argmax(dim=1) - 1
    out = torch.full((G,), -1, dtype=torch.long, device=point_labels.device)
    out[has_any] = preds[has_any]
    return out


def compute_patch_label_distribution(label_masks, patch_idx):
    """Per-patch multi-label distribution from label masks.
    label_masks: (L,N) bool, patch_idx: (G,M) long -> (G,L)
    """
    L, N = label_masks.shape
    G, M = patch_idx.shape
    idx = patch_idx.reshape(1, -1).expand(L, -1)
    gathered = label_masks.gather(1, idx).view(L, G, M)
    dist = gathered.float().mean(dim=-1).transpose(0, 1).contiguous()
    return dist


def assign_points_from_patches(points_xyz, patch_centers, patch_logits, patch_idx, mode="nearest"):
    B, _, N = points_xyz.shape
    K = patch_logits.shape[-1]
    if mode == "membership":
        point_logits = torch.zeros(B, N, K, device=points_xyz.device)
        counts = torch.zeros(B, N, 1, device=points_xyz.device)
        for b in range(B):
            idx = patch_idx[b].reshape(-1)
            src = patch_logits[b].unsqueeze(1).expand_as(patch_idx[b].unsqueeze(-1)).reshape(-1, K)
            point_logits[b].index_add_(0, idx, src)
            ones = torch.ones(idx.shape[0], 1, device=points_xyz.device)
            counts[b].index_add_(0, idx, ones)
        return point_logits / counts.clamp_min(1.0)
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)
    _, nearest = knn(patch_centers.transpose(1, 2).contiguous(), points_xyz.transpose(1, 2).contiguous())
    nearest = nearest.squeeze(-1)
    point_logits = patch_logits.gather(1, nearest.unsqueeze(-1).expand(-1, -1, K))
    return point_logits


def _euler_to_matrix(rx, ry, rz):
    B = rx.shape[0]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rz = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)], dim=-1),
            torch.stack([sz, cz, torch.zeros_like(cz)], dim=-1),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)], dim=-1),
        ],
        dim=1,
    )
    Ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
        ],
        dim=1,
    )
    Rx = torch.stack(
        [
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
            torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx), sx, cx], dim=-1),
        ],
        dim=1,
    )
    return Rz @ Ry @ Rx


def build_augmentor(enabled, prob, rot_deg, translate, scale_low, scale_high, jitter_sigma, jitter_clip, has_normals=False):
    if not enabled:
        return None
    rot_rad = float(rot_deg) * np.pi / 180.0

    def _augment(points):
        if prob < 1.0 and torch.rand(()) > prob:
            return points
        x = points
        B, C, N = x.shape
        xyz = x[:, :3, :]
        if rot_rad > 0:
            rx = (torch.rand(B, device=x.device) * 2 - 1) * rot_rad
            ry = (torch.rand(B, device=x.device) * 2 - 1) * rot_rad
            rz = (torch.rand(B, device=x.device) * 2 - 1) * rot_rad
            R = _euler_to_matrix(rx, ry, rz)
            xyz = R @ xyz
            if has_normals and C >= 6:
                nrm = x[:, 3:6, :]
                nrm = R @ nrm
                nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-12)
                x = torch.cat([xyz, nrm, x[:, 6:, :]], dim=1) if C > 6 else torch.cat([xyz, nrm], dim=1)
            else:
                x = torch.cat([xyz, x[:, 3:, :]], dim=1) if C > 3 else xyz
        if scale_high > 0 and scale_low > 0:
            s = torch.empty(B, device=x.device).uniform_(float(scale_low), float(scale_high)).view(B, 1, 1)
            x[:, :3, :] = x[:, :3, :] * s
        if translate > 0:
            t = (torch.rand(B, 3, device=x.device) * 2 - 1) * float(translate)
            x[:, :3, :] = x[:, :3, :] + t.unsqueeze(-1)
        if jitter_sigma > 0 and jitter_clip > 0:
            noise = torch.clamp(torch.randn_like(x[:, :3, :]) * float(jitter_sigma), -float(jitter_clip), float(jitter_clip))
            x[:, :3, :] = x[:, :3, :] + noise
        return x

    return _augment


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


def freeze_encoder_except_last_block(model):
    for p in model.parameters():
        p.requires_grad = False
    last_block = None
    if hasattr(model, "blocks"):
        inner = getattr(model.blocks, "blocks", None)
        if inner and len(inner) > 0:
            last_block = inner[-1]
    if last_block is not None:
        for p in last_block.parameters():
            p.requires_grad = True
    if hasattr(model, "norm"):
        for p in model.norm.parameters():
            p.requires_grad = True
    if hasattr(model, "reduce_dim"):
        for p in model.reduce_dim.parameters():
            p.requires_grad = True
    if hasattr(model, "cls_token"):
        model.cls_token.requires_grad = True
    if hasattr(model, "cls_pos"):
        model.cls_pos.requires_grad = True


def load_stage1(model, proj, path, device):
    if not path:
        return
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "proj" in ckpt and proj is not None:
        proj.load_state_dict(ckpt["proj"], strict=False)


def train_epoch(model, proj, clip_model, tokenizer, loader, optimizer, device, temp_module, text_cache, text_setting, exclude_category_label=False, drop_labels_not_in_bank=False, augment=None):
    model.train()
    proj.train()
    if clip_model is not None:
        clip_model.eval()
    total_loss_sum = 0.0
    total_loss_batches = 0
    total_patch_correct = 0
    total_patches = 0
    for batch in tqdm(loader, total=len(loader), desc="Train data", smoothing=0.9):
        points_b = []
        point_labels_b = []
        label_names_b = []
        item_ids_b = []
        label_masks_b = []
        for sample in batch:
            points_b.append(prepare_points(sample, device))
            point_labels_b.append(sample["point_labels"].to(device, non_blocking=True))
            label_names_b.append(sample["label_names"])
            item_ids_b.append(sample.get("item_id", ""))
            lm = sample.get("label_masks", None)
            label_masks_b.append(lm if isinstance(lm, torch.Tensor) else torch.as_tensor(lm) if lm is not None else None)
        if not points_b:
            continue
        points = torch.cat(points_b, dim=0)
        if augment is not None:
            points = augment(points)

        optimizer.zero_grad()
        patch_emb, patch_centers, patch_idx = model.forward_patches(points)
        patch_feat = proj(patch_emb)
        B, G, D = patch_feat.shape

        batch_loss = 0.0
        batch_valid = 0
        scale = temp_module()

        for b in range(B):
            names = label_names_b[b]
            if not names:
                continue
            K_full = len(names)
            item_id = item_ids_b[b]
            category = item_id.rsplit("_", 1)[0] if "_" in item_id else item_id
            keep_indices = list(range(K_full))
            if exclude_category_label and item_id:
                key_cat = category.replace("_", " ").strip().lower()
                for i, nm in enumerate(names):
                    if _clean_text(nm) == key_cat:
                        keep_indices = [k for k in keep_indices if k != i]
                        break
            if drop_labels_not_in_bank and hasattr(text_cache, "has_key_for_sample"):
                filtered = []
                for i in keep_indices:
                    if text_cache.has_key_for_sample(names[i], category, text_setting):
                        filtered.append(i)
                keep_indices = filtered
            if len(keep_indices) == 0:
                continue
            kept_names = [names[i] for i in keep_indices]
            text_feats = text_cache.encode_labels_for_sample(kept_names, category, text_setting).to(device)
            if text_feats.numel() == 0:
                continue
            Y_patch_full = compute_patch_label_distribution(label_masks_b[b].to(device), patch_idx[b])
            Y_patch = Y_patch_full[:, keep_indices]
            logits = (patch_feat[b] @ text_feats.t()) * scale
            K_keep = text_feats.size(0)
            pos_w = torch.full((K_keep,), max(1.0, float(K_keep - 1)), device=text_feats.device)
            loss_b = F.binary_cross_entropy_with_logits(logits, Y_patch, pos_weight=pos_w)
            # Patch accuracy via majority label from point labels
            lab_full = compute_patch_targets_vector(point_labels_b[b], patch_idx[b], K_full)
            keep_map = {old: new for new, old in enumerate(keep_indices)}
            mapped = torch.full_like(lab_full, -1)
            for old, new in keep_map.items():
                mapped[lab_full == old] = new
            pred_p = logits.argmax(dim=-1)
            valid_p = mapped >= 0
            total_patch_correct += (pred_p[valid_p] == mapped[valid_p]).sum().item()
            total_patches += valid_p.sum().item()

            batch_loss = batch_loss + loss_b
            batch_valid += 1

        if batch_valid > 0:
            (batch_loss / batch_valid).backward()
            optimizer.step()
            total_loss_sum += (batch_loss.item() / batch_valid)
            total_loss_batches += 1

    avg_loss = total_loss_sum / max(total_loss_batches, 1)
    patch_acc = total_patch_correct / max(total_patches, 1)
    return avg_loss, patch_acc, temp_module().item()


@torch.no_grad()
def eval_epoch(model, proj, clip_model, tokenizer, loader, device, temp_module, text_cache, text_setting, exclude_category_label=False, drop_labels_not_in_bank=False):
    model.eval()
    proj.eval()
    if clip_model is not None:
        clip_model.eval()
    total_loss_sum = 0.0
    total_loss_batches = 0
    total_patch_correct = 0
    total_patches = 0
    for batch in tqdm(loader, total=len(loader), desc="Val data", smoothing=0.9):
        points_b = []
        point_labels_b = []
        label_names_b = []
        item_ids_b = []
        label_masks_b = []
        for sample in batch:
            points_b.append(prepare_points(sample, device))
            point_labels_b.append(sample["point_labels"].to(device, non_blocking=True))
            label_names_b.append(sample["label_names"])
            item_ids_b.append(sample.get("item_id", ""))
            lm = sample.get("label_masks", None)
            label_masks_b.append(lm if isinstance(lm, torch.Tensor) else torch.as_tensor(lm) if lm is not None else None)
        if not points_b:
            continue
        points = torch.cat(points_b, dim=0)
        patch_emb, patch_centers, patch_idx = model.forward_patches(points)
        patch_feat = proj(patch_emb)
        B, G, D = patch_feat.shape

        batch_loss = 0.0
        batch_valid = 0
        scale = temp_module()

        for b in range(B):
            names = label_names_b[b]
            if not names:
                continue
            K_full = len(names)
            item_id = item_ids_b[b]
            category = item_id.rsplit("_", 1)[0] if "_" in item_id else item_id
            keep_indices = list(range(K_full))
            if exclude_category_label and item_id:
                key_cat = category.replace("_", " ").strip().lower()
                for i, nm in enumerate(names):
                    if _clean_text(nm) == key_cat:
                        keep_indices = [k for k in keep_indices if k != i]
                        break
            if drop_labels_not_in_bank and hasattr(text_cache, "has_key_for_sample"):
                filtered = []
                for i in keep_indices:
                    if text_cache.has_key_for_sample(names[i], category, text_setting):
                        filtered.append(i)
                keep_indices = filtered
            if len(keep_indices) == 0:
                continue
            kept_names = [names[i] for i in keep_indices]
            text_feats = text_cache.encode_labels_for_sample(kept_names, category, text_setting).to(device)
            if text_feats.numel() == 0:
                continue
            Y_patch_full = compute_patch_label_distribution(label_masks_b[b].to(device), patch_idx[b])
            Y_patch = Y_patch_full[:, keep_indices]
            logits = (patch_feat[b] @ text_feats.t()) * scale
            K_keep = text_feats.size(0)
            pos_w = torch.full((K_keep,), max(1.0, float(K_keep - 1)), device=text_feats.device)
            loss_b = F.binary_cross_entropy_with_logits(logits, Y_patch, pos_weight=pos_w)
            lab_full = compute_patch_targets_vector(point_labels_b[b], patch_idx[b], K_full)
            keep_map = {old: new for new, old in enumerate(keep_indices)}
            mapped = torch.full_like(lab_full, -1)
            for old, new in keep_map.items():
                mapped[lab_full == old] = new
            pred_p = logits.argmax(dim=-1)
            valid_p = mapped >= 0
            total_patch_correct += (pred_p[valid_p] == mapped[valid_p]).sum().item()
            total_patches += valid_p.sum().item()

            batch_loss = batch_loss + loss_b
            batch_valid += 1

        if batch_valid > 0:
            total_loss_sum += (batch_loss.item() / batch_valid)
            total_loss_batches += 1

    avg_loss = total_loss_sum / max(total_loss_batches, 1)
    patch_acc = total_patch_correct / max(total_patches, 1)
    return avg_loss, patch_acc, temp_module().item()


def parse_args():
    p = argparse.ArgumentParser("Stage 2: CLIP patch-text alignment on training data")
    # Model & CLIP
    p.add_argument("--arch", type=str, default="pointtransformer")
    p.add_argument("--clip_model", type=str, default="ViT-bigG-14")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
    p.add_argument("--clip_tau", type=float, default=0.07)
    p.add_argument("--text_setting", type=str, default="part_only", choices=["part_only"])
    p.add_argument("--text_cache", type=int, default=20000)
    p.add_argument("--num_group", type=int, default=128)
    p.add_argument("--group_size", type=int, default=32)
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--enc_learning_rate", type=float, default=1e-4)
    p.add_argument("--decay_rate", type=float, default=5e-2)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train_last_block_only", action="store_true", default=False)
    p.add_argument("--train_encoder", action="store_true", default=True)
    p.add_argument("--init_stage1", type=str, default="", help="Path to a Stage 1 checkpoint to init encoder.")
    # Data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--val_list", type=str, required=True)
    p.add_argument("--npoint", type=int, default=2048)
    p.add_argument("--min_per_label", type=int, default=32)
    p.add_argument("--random_sample_train", action="store_true", default=False)
    # No color/normal channels in this setup
    # ShapeNetPart eval during training
    p.add_argument("--shapenet_root", type=str, default="", help="If set, run ShapeNetPart eval every eval_shapenet_every epochs.")
    p.add_argument("--eval_shapenet_every", type=int, default=0, help="Run ShapeNetPart eval every N epochs (0 to disable).")
    p.add_argument("--shapenet_batch_size", type=int, default=16)
    p.add_argument("--shapenet_workers", type=int, default=4)
    p.add_argument("--shapenet_assign", type=str, default="nearest", choices=["nearest", "membership"])
    p.add_argument("--shapenet_use_normal", action="store_true", default=False)
    # Text bank options
    p.add_argument("--text_bank_dir", type=str, default=None, help="Directory with merged text banks (defaults to <data_root>/labeled/text_banks if exists)")
    p.add_argument("--text_bank_require", action="store_true", default=False)
    p.add_argument("--drop_labels_not_in_bank", action="store_true", default=False)
    p.add_argument("--exclude_category_label", action="store_true", default=False)
    # Augmentation
    p.add_argument("--aug_enable", action="store_true", default=False)
    p.add_argument("--aug_prob", type=float, default=0.9)
    p.add_argument("--aug_rot_deg", type=float, default=20)
    p.add_argument("--aug_translate", type=float, default=0.02)
    p.add_argument("--aug_scale_low", type=float, default=0.9)
    p.add_argument("--aug_scale_high", type=float, default=1.1)
    p.add_argument("--aug_jitter_sigma", type=float, default=0.01)
    p.add_argument("--aug_jitter_clip", type=float, default=0.05)
    # Logging
    p.add_argument("--log_dir", type=str, default="logs/stage2")
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
    exp_dir = Path(args.log_dir) / (run_name if run_name else f"stage2_{timestr}")
    ckpt_dir = exp_dir / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage2_clip")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(exp_dir / "log.txt"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    def log_string(s):
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

    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_dim = int(getattr(clip_model, "text_projection", None).shape[1] if hasattr(clip_model, "text_projection") else 512)

    # Text cache (bank -> fallback to on-the-fly)
    default_bank_dir = Path(args.data_root) / "labeled" / "text_banks"
    bank_dir = args.text_bank_dir or (str(default_bank_dir) if default_bank_dir.exists() else None)
    po = try_load_text_banks(bank_dir)
    lru_cache = LRUTextCache(device=device, capacity=args.text_cache, clip_model=clip_model, tokenizer=tokenizer, text_dim=text_dim)
    if po is not None:
        text_cache = BankTextCache(po=po, strict=args.text_bank_require, fallback=lru_cache)
        log_string(f"Loaded text bank from {bank_dir}")
    else:
        text_cache = lru_cache
        if bank_dir:
            log_string(f"No text bank found under {bank_dir}; using on-the-fly CLIP encoding.")

    shapenet_eval_enabled = bool(args.shapenet_root and args.eval_shapenet_every and args.eval_shapenet_every > 0)
    shapenet_text_feats = None
    if shapenet_eval_enabled:
        sn_ds = PartNormalDataset(root=args.shapenet_root, npoints=2048, split="test", normal_channel=args.shapenet_use_normal)
        sn_seg_classes = sn_ds.seg_classes
        sn_id2cat = {v: k for k, v in sn_ds.classes.items()}
        shapenet_text_feats = eval_tools.encode_text_from_part_names(
            sn_seg_classes,
            sn_id2cat,
            device=device,
            setting=args.text_setting,
            clip_model=clip_model,
            tokenizer=tokenizer,
        )
        log_string(f"ShapeNetPart eval enabled every {args.eval_shapenet_every} epochs.")

    model = build_model(args, device)
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)
    load_stage1(model, proj=None, path=args.init_stage1, device=device)
    if args.train_last_block_only:
        freeze_encoder_except_last_block(model)
    elif not args.train_encoder:
        for p in model.parameters():
            p.requires_grad = False

    params = []
    enc_params = [p for p in model.parameters() if p.requires_grad]
    if enc_params:
        params.append({"params": enc_params, "lr": args.enc_learning_rate})
    params.append({"params": proj.parameters(), "lr": args.learning_rate})
    temp_module = LearnableTemp(init_tau=args.clip_tau)
    params.append({"params": temp_module.parameters(), "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.decay_rate)

    augment = build_augmentor(
        enabled=args.aug_enable,
        prob=args.aug_prob,
        rot_deg=args.aug_rot_deg,
        translate=args.aug_translate,
        scale_low=args.aug_scale_low,
        scale_high=args.aug_scale_high,
        jitter_sigma=args.aug_jitter_sigma,
        jitter_clip=args.aug_jitter_clip
    )

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
        torch.save(
            {
                "epoch": epoch_idx,
                "model": model.state_dict(),
                "proj": proj.state_dict(),
                "temp": temp_module.state_dict(),
                "args": vars(args),
            },
            path,
        )
        log_string(f"[ckpt] saved to {path}")

    for epoch in range(1, args.epoch + 1):
        loss_tr, patch_acc_tr, cur_scale = train_epoch(
            model,
            proj,
            clip_model,
            tokenizer,
            dl_tr,
            optimizer,
            device,
            temp_module,
            text_cache=text_cache,
            text_setting=args.text_setting,
            exclude_category_label=args.exclude_category_label,
            drop_labels_not_in_bank=args.drop_labels_not_in_bank,
            augment=augment,
        )
        log_string(f"Epoch {epoch:03d} | train loss {loss_tr:.4f} | patch acc {patch_acc_tr:.4f} | scale {cur_scale:.3f}")
        if wandb_run:
            wandb_run.log({"train/loss": loss_tr, "train/patch_acc": patch_acc_tr, "train/logit_scale": cur_scale, "epoch": epoch})

        if args.eval_every and (epoch % args.eval_every == 0):
            loss_va, patch_acc_va, cur_scale_va = eval_epoch(
                model,
                proj,
                clip_model,
                tokenizer,
                dl_va,
                device,
                temp_module,
                text_cache=text_cache,
                text_setting=args.text_setting,
                exclude_category_label=args.exclude_category_label,
                drop_labels_not_in_bank=args.drop_labels_not_in_bank,
            )
            log_string(f"[val] loss {loss_va:.4f} | patch acc {patch_acc_va:.4f} | scale {cur_scale_va:.3f}")
            if wandb_run:
                wandb_run.log({"val/loss": loss_va, "val/patch_acc": patch_acc_va, "val/logit_scale": cur_scale_va, "epoch": epoch})
        if shapenet_eval_enabled and args.eval_shapenet_every and (epoch % args.eval_shapenet_every == 0):
            cur_tau = 1.0 / max(float(temp_module().item()), 1e-6)
            metrics_sn = eval_tools.evaluate_shapenet(
                model,
                proj,
                shapenet_text_feats,
                cur_tau,
                device,
                args.shapenet_root,
                batch_size=args.shapenet_batch_size,
                num_workers=args.shapenet_workers,
                assign_mode=args.shapenet_assign,
                progress=False,
                use_normal=args.shapenet_use_normal,
            )
            log_string(
                f"[shapenet] point_acc {metrics_sn['point_acc']:.4f} | miou {metrics_sn['point_miou']:.4f} | ciou {metrics_sn['point_ciou']:.4f} | patch_acc {metrics_sn['patch_acc']:.4f}"
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "shapenet/point_acc": metrics_sn["point_acc"],
                        "shapenet/point_miou": metrics_sn["point_miou"],
                        "shapenet/point_ciou": metrics_sn["point_ciou"],
                        "shapenet/patch_acc": metrics_sn["patch_acc"],
                        "epoch": epoch,
                    }
                )
        if args.save_every and (epoch % args.save_every == 0):
            save_checkpoint(epoch, f"epoch_{epoch:03d}")
    save_checkpoint(args.epoch, "last")
    if wandb_run:
        wandb_run.finish()
    log_string("Training finished.")


if __name__ == "__main__":
    main()
