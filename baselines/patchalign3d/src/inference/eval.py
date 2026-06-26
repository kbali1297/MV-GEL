"""Evaluation script for PatchAlign3D checkpoints on ShapeNetPart.

This mirrors the behavior of Point-BERT's segmentation/tools/eval_cli.py but
only depends on PatchAlign3D code (datasets + point_transformer).
"""

import argparse
import os
import re
from pathlib import Path

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from easydict import EasyDict

from patchalign3d.datasets.shapenet import PartNormalDataset
from patchalign3d.models import point_transformer

import open_clip


# Canonical part-name candidates (global ShapeNetPart ids)
PART_NAME_CANDIDATES = {
    "Airplane": ["body", "wing", "tail", "engine or frame"],
    "Chair": ["back", "seat", "leg", "arm"],
    "Car": ["roof", "hood", "wheel", "body"],
    "Table": ["desktop", "leg or support", "drawer"],
    "Lamp": ["base", "lampshade", "fixing bracket", "pole"],
    "Guitar": ["headstock", "neck", "body"],
    "Rocket": ["body", "fin", "nose"],
    "Pistol": ["barrel", "handle or grip", "trigger and guard"],
    "Skateboard": ["wheel", "deck", "belt for foot"],
    "Bag": ["handle", "body"],
    "Cap": ["crown", "brim"],
    "Laptop": ["keyboard", "screen"],
    "Mug": ["handle", "cup"],
    "Knife": ["blade", "handle"],
    "Earphone": ["earcup", "headband", "data wire"],
    "Motorbike": ["gas tank", "seat", "wheel", "handles or handlebars", "headlight", "engine or frame"],
}

PART_ONLY_TEMPLATES = [
    "{}",
    "a {}",
    "{} part",
]

PART_PLUS_CAT_TEMPLATES = [
    "a {} of a {}",
    "the {} of a {}",
    "{} of {}",
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


def build_model(num_group, group_size, use_color, use_normal, device):
    cfg = EasyDict(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        cls_dim=50,
        num_heads=6,
        group_size=group_size,
        num_group=num_group,
        encoder_dims=256,
        color=bool(use_color or use_normal),
        num_classes=16,
    )
    model = point_transformer.get_model(cfg).to(device)
    return model


def prepare_points(points):
    if points.ndim != 3:
        raise ValueError(f"Expected (B,N,C) points, got shape {points.shape}")
    pts = points.transpose(2, 1).contiguous()
    pts[:, [1, 2], :] = pts[:, [2, 1], :]
    return pts


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
    return patch_logits.gather(1, nearest.unsqueeze(-1).expand(-1, -1, K))


def compute_point_metrics(point_pred, target, label, seg_classes, id2cat):
    """Returns accuracy, per-instance IoUs, and per-category IoUs (instance-averaged).

    Note: If a part is absent in GT for that sample, it is skipped (no IoU added),
    to avoid inflating IoU when union is zero.
    """
    B, N = target.shape
    acc = (point_pred == target).float().mean().item()
    inst_ious = []
    cat_to_ious = {cat: [] for cat in seg_classes}
    for b in range(B):
        cat = id2cat[int(label[b].item())]
        valid = seg_classes[cat]
        preds = point_pred[b]
        gts = target[b]
        ious = []
        for pid in valid:
            pred_mask = preds == pid
            gt_mask = gts == pid
            if gt_mask.sum().item() == 0:
                # part not present in GT -> skip to avoid over-estimating IoU
                continue
            inter = (pred_mask & gt_mask).sum().item()
            union = (pred_mask | gt_mask).sum().item()
            if union == 0:
                continue
            iou = inter / union
            ious.append(iou)
        if ious:
            inst = sum(ious) / len(ious)
            inst_ious.append(inst)
            cat_to_ious[cat].append(inst)
    return dict(
        acc=acc,
        inst_ious=inst_ious,
        cat_to_ious=cat_to_ious,
        per_cat_iou={k: (sum(v) / len(v)) for k, v in cat_to_ious.items() if v},
    )


def compute_point_metrics_generic(point_pred, target, num_labels):
    """Instance-weighted IoU for datasets without category structure (e.g., FAUST)."""
    B, N = target.shape
    acc = (point_pred == target).float().mean().item()
    inst_ious = []
    part_to_ious = {i: [] for i in range(num_labels)}
    for b in range(B):
        preds = point_pred[b]
        gts = target[b]
        ious = []
        for pid in range(num_labels):
            pred_mask = preds == pid
            gt_mask = gts == pid
            inter = (pred_mask & gt_mask).sum().item()
            union = (pred_mask | gt_mask).sum().item()
            iou = 1.0 if union == 0 else inter / union
            ious.append(iou)
            part_to_ious[pid].append(iou)
        if ious:
            inst_ious.append(sum(ious) / len(ious))
    return dict(
        acc=acc,
        inst_ious=inst_ious,
        part_to_ious={k: (sum(v) / len(v)) for k, v in part_to_ious.items() if v},
    )


@torch.no_grad()
def encode_texts(names, category, setting, clip_model, tokenizer, device):
    texts = []
    cname = _clean_text(category or "")
    for nm in names:
        nm = _clean_text(nm)
        if setting in ("part_plus_cat", "ensemble") and cname:
            for tpl in PART_PLUS_CAT_TEMPLATES:
                slots = tpl.count("{}")
                if slots == 2:
                    texts.append(tpl.format(nm, cname))
                elif slots == 1:
                    texts.append(tpl.format(f"{cname} {nm}"))
                else:
                    texts.append(f"{cname} {nm}")
        if (setting in ("part_only", "ensemble")) or not cname:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(nm) if tpl.count("{}") == 1 else nm)
    if not texts:
        return torch.zeros(clip_model.text_projection.shape[1], device=device)
    toks = tokenizer(texts).to(device)
    feat = clip_model.encode_text(toks)
    feat = F.normalize(feat, dim=-1)
    return F.normalize(feat.mean(dim=0, keepdim=True), dim=-1).squeeze(0)


@torch.no_grad()
def encode_text_from_part_names(seg_classes, id2cat, device, setting, clip_model, tokenizer):
    bank = torch.zeros(50, clip_model.text_projection.shape[1], device=device)
    filled = torch.zeros(50, dtype=torch.bool, device=device)
    for cid, cat in id2cat.items():
        if cat not in seg_classes:
            continue
        gids = list(sorted(seg_classes[cat]))
        cand = PART_NAME_CANDIDATES.get(cat, [])
        for idx, gid in enumerate(gids):
            pname = cand[idx] if idx < len(cand) else f"part{idx}"
            bank[gid] = encode_texts([pname], category=cat if setting != "part_only" else None, setting=setting, clip_model=clip_model, tokenizer=tokenizer, device=device)
            filled[gid] = True
    if (~filled).any():
        mean_vec = F.normalize(bank[filled].mean(dim=0, keepdim=True), dim=-1) if filled.any() else F.normalize(torch.randn(1, bank.shape[1], device=device), dim=-1)
        bank = torch.where(filled.unsqueeze(-1), bank, mean_vec.expand_as(bank))
    return bank


def evaluate_shapenet(model, proj, text_feats_50, tau, device, shapenet_root, batch_size, num_workers, assign_mode, progress, use_normal):
    dataset = PartNormalDataset(root=shapenet_root, npoints=2048, split="test", normal_channel=use_normal)
    seg_classes = dataset.seg_classes
    id2cat = {v: k for k, v in dataset.classes.items()}
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    model.eval()
    proj.eval()
    tot_patch = tot_patch_correct = 0
    pt_acc = 0.0
    inst_iou_sum = 0.0
    inst_count = 0
    cat_to_ious = {cat: [] for cat in seg_classes}
    iterator = tqdm(dl, total=len(dl), desc="ShapeNetPart", smoothing=0.9) if progress else dl
    with torch.no_grad():
        for points, label, target in iterator:
            points = points.to(device).float()
            label = label.to(device).long()
            target = target.to(device).long()
            points = prepare_points(points)
            xyz = points[:, :3, :]
            pe, pc, pi = model.forward_patches(points)
            x = proj(pe)
            logits = (x @ text_feats_50.t()) / max(tau, 1e-6)
            valid = torch.zeros_like(logits, dtype=torch.bool)
            for b in range(points.size(0)):
                cat = id2cat[int(label[b].item())]
                allow = seg_classes.get(cat, [])
                if allow:
                    valid[b, :, allow] = True
            logits = logits.masked_fill(~valid, -1e4)
            patch_pred = logits.argmax(dim=-1)
            for b in range(points.size(0)):
                patch_gt = compute_patch_targets_vector(target[b], pi[b], 50)
                present = patch_gt >= 0
                tot_patch += present.sum().item()
                tot_patch_correct += (patch_pred[b][present] == patch_gt[present]).sum().item()
            point_logits = assign_points_from_patches(xyz, pc, logits, pi, mode=assign_mode)
            pred = point_logits.argmax(dim=-1)
            metrics_batch = compute_point_metrics(pred, target, label, seg_classes, id2cat)
            pt_acc += metrics_batch["acc"]
            inst_iou_sum += sum(metrics_batch["inst_ious"])
            inst_count += len(metrics_batch["inst_ious"])
            for cat, v in metrics_batch["per_cat_iou"].items():
                cat_to_ious[cat].append(v)
    num_batches = len(dl)
    per_cat_iou = {k: sum(v) / len(v) for k, v in cat_to_ious.items() if v}
    ciou_vals = list(per_cat_iou.values())
    metrics = {
        "patch_acc": tot_patch_correct / max(tot_patch, 1),
        "point_acc": pt_acc / max(num_batches, 1),
        "point_miou": inst_iou_sum / max(inst_count, 1),
        "point_ciou": (sum(ciou_vals) / len(ciou_vals)) if ciou_vals else 0.0,
        "per_cat_iou": per_cat_iou,
    }
    return metrics


class FaustNpzDataset(torch.utils.data.Dataset):
    """Loads FAUST-style files with keys: points (N,3), labels (N,), label_names (K)."""

    def __init__(self, paths, npoints=2048):
        files = []
        for p in paths:
            cand = []
            if any(ch in p for ch in "*?[]"):
                cand = glob.glob(p)
            else:
                cand = [p]
            for c in cand:
                path = Path(c)
                if path.is_dir():
                    files.extend(sorted(path.glob("*.npz")))
                elif path.suffix == ".npz" and path.exists():
                    files.append(path)
        if not files:
            raise RuntimeError("No NPZ files found for FAUST evaluation.")
        self.files = files
        self.npoints = int(npoints)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        pts = np.asarray(data["points"], dtype=np.float32)
        labels = np.asarray(data["labels"]).reshape(-1).astype(np.int64)
        names = [str(x) for x in data.get("label_names", [])]
        if names and labels.max(initial=-1) >= len(names):
            labels = np.clip(labels, 0, len(names) - 1)
        if self.npoints > 0 and pts.shape[0] != self.npoints:
            N = pts.shape[0]
            replace = N < self.npoints
            sel = np.random.choice(N, size=self.npoints, replace=replace)
            pts = pts[sel]
            labels = labels[sel]
        return {"points": pts[:, :3], "labels": labels, "label_names": names, "slug": path.stem}


def collate_faust(batch):
    return batch


def evaluate_faust(model, proj, clip_model, tokenizer, tau, device, npz_paths, batch_size, num_workers, assign_mode, progress, npoints):
    dataset = FaustNpzDataset(npz_paths, npoints=npoints)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate_faust)
    model.eval()
    proj.eval()
    tot_patch = tot_patch_correct = 0
    acc_sum = 0.0
    sample_count = 0
    inst_iou_sum = 0.0
    inst_count = 0
    part_to_ious = {}
    iterator = tqdm(dl, total=len(dl), desc="FAUST", smoothing=0.9) if progress else dl
    with torch.no_grad():
        for batch in iterator:
            points_list = []
            labels_list = []
            names_list = []
            for sample in batch:
                pts = torch.as_tensor(sample["points"], dtype=torch.float32, device=device).unsqueeze(0)  # (1,N,3)
                labels = torch.as_tensor(sample["labels"], dtype=torch.long, device=device)
                points_list.append(pts)
                labels_list.append(labels)
                names_list.append(sample.get("label_names", []))
            points = torch.cat(points_list, dim=0)  # (B,N,3)
            labels = labels_list
            B, N, _ = points.shape
            points_pt = prepare_points(points)  # (B,3,N)
            pe, pc, pi = model.forward_patches(points_pt)
            x = proj(pe)
            for b in range(B):
                names = names_list[b]
                if not names:
                    continue
                # Encode each part name separately to get (K,D)
                text_feats = torch.stack(
                    [encode_texts([nm], category=None, setting="part_only", clip_model=clip_model, tokenizer=tokenizer, device=device) for nm in names],
                    dim=0,
                )
                logits = (x[b] @ text_feats.t()) / max(tau, 1e-6)  # (G,K)
                patch_pred = logits.argmax(dim=-1)
                patch_gt = compute_patch_targets_vector(labels[b], pi[b], len(names))
                present = patch_gt >= 0
                tot_patch += present.sum().item()
                tot_patch_correct += (patch_pred[present] == patch_gt[present]).sum().item()
                # Point assignment
                point_logits = assign_points_from_patches(points_pt[b : b + 1, :3, :], pc[b : b + 1], logits.unsqueeze(0), pi[b : b + 1], mode=assign_mode)
                pred = point_logits.argmax(dim=-1).squeeze(0)
                metrics_b = compute_point_metrics_generic(pred.unsqueeze(0), labels[b].unsqueeze(0), len(names))
                acc_sum += metrics_b["acc"]
                sample_count += 1
                inst_iou_sum += sum(metrics_b["inst_ious"])
                inst_count += len(metrics_b["inst_ious"])
                for pid, v in metrics_b["part_to_ious"].items():
                    part_to_ious.setdefault(pid, []).append(v)
    per_part_iou = {k: sum(v) / len(v) for k, v in part_to_ious.items() if v}
    metrics = {
        "patch_acc": tot_patch_correct / max(tot_patch, 1),
        "point_acc": acc_sum / max(sample_count, 1),
        "point_miou": inst_iou_sum / max(inst_count, 1),
        "per_part_iou": per_part_iou,
    }
    return metrics


def parse_args():
    p = argparse.ArgumentParser("Evaluate PatchAlign3D checkpoint (ShapeNetPart, FAUST)")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--shapenet_root", type=str, required=False, default="", help="Path to ShapeNetPart root")
    p.add_argument("--faust_npz", type=str, nargs="*", default=[], help="FAUST NPZ file(s) or directory(ies)")
    p.add_argument("--faust_npoints", type=int, default=2048)
    p.add_argument("--text_setting", type=str, default="part_only", choices=["part_only", "part_plus_cat", "ensemble"])
    p.add_argument("--assign", type=str, default="nearest", choices=["nearest", "membership"])
    p.add_argument("--clip_model", type=str, default="ViT-bigG-14")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
    p.add_argument("--clip_tau", type=float, default=0.07)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_group", type=int, default=128)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--use_color", action="store_true", default=False)
    p.add_argument("--use_normal", action="store_true", default=False)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no_progress", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_dim = int(getattr(clip_model, "text_projection", None).shape[1] if hasattr(clip_model, "text_projection") else 512)

    model = build_model(args.num_group, args.group_size, use_color=args.use_color, use_normal=args.use_normal, device=device)
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        res = model.load_state_dict(ckpt["model"], strict=False)
        missing = getattr(res, "missing_keys", res[0] if isinstance(res, (list, tuple)) else [])
        unexpected = getattr(res, "unexpected_keys", res[1] if isinstance(res, (list, tuple)) else [])
        print(f"[ckpt] model loaded: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)
    if "proj" in ckpt:
        res = proj.load_state_dict(ckpt["proj"], strict=False)
        missing = getattr(res, "missing_keys", res[0] if isinstance(res, (list, tuple)) else [])
        unexpected = getattr(res, "unexpected_keys", res[1] if isinstance(res, (list, tuple)) else [])
        print(f"[ckpt] proj loaded: missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing)
        if unexpected:
            print("  unexpected:", unexpected)

    tau = float(args.clip_tau)

    if args.shapenet_root:
        dataset_tmp = PartNormalDataset(root=args.shapenet_root, split="test", normal_channel=args.use_normal)
        seg_classes = dataset_tmp.seg_classes
        id2cat = {v: k for k, v in dataset_tmp.classes.items()}
        text_feats_50 = encode_text_from_part_names(seg_classes, id2cat, device=device, setting=args.text_setting, clip_model=clip_model, tokenizer=tokenizer)

        metrics_sh = evaluate_shapenet(
            model,
            proj,
            text_feats_50,
            tau,
            device,
            args.shapenet_root,
            batch_size=args.batch_size,
            num_workers=args.workers,
            assign_mode=args.assign,
            progress=not args.no_progress,
            use_normal=args.use_normal,
        )
        print("ShapeNetPart metrics:", metrics_sh)

    if args.faust_npz:
        metrics_faust = evaluate_faust(
            model,
            proj,
            clip_model,
            tokenizer,
            tau,
            device,
            args.faust_npz,
            batch_size=args.batch_size,
            num_workers=args.workers,
            assign_mode=args.assign,
            progress=not args.no_progress,
            npoints=args.faust_npoints,
        )
        print("FAUST metrics:", metrics_faust)


if __name__ == "__main__":
    main()
