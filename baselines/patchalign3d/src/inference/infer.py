#!/usr/bin/env python3
"""
Minimal inference for PatchAlign3D stage-2 checkpoints.

Inputs:
  - .npz with 'points' (N,3) [optional 'label_names']
  - .ply point cloud
Outputs:
  - per-point predictions saved to .npz
  - optional rendered PNG with legend
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from patchalign3d.models import point_transformer

import open_clip


PART_ONLY_TEMPLATES = ["{}", "a {}", "{} part"]
PART_PLUS_CAT_TEMPLATES = ["a {} of a {}", "the {} of a {}", "{} of {}", "a {} part of a {}"]


def _clean_text(s):
    s = s.strip().lower().replace("_", " ")
    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


def prepare_points(points):
    if points.ndim != 3:
        raise ValueError(f"Expected (B,N,C), got {points.shape}")
    pts = points.transpose(2, 1).contiguous()
    pts[:, [1, 2], :] = pts[:, [2, 1], :]
    return pts


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


def encode_texts(names, setting, clip_model, tokenizer, device):
    texts = []
    for nm in names:
        nm = _clean_text(nm)
        if setting == "part_plus_cat":
            for tpl in PART_PLUS_CAT_TEMPLATES:
                texts.append(tpl.format(nm, "object"))
        else:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(nm) if tpl.count("{}") == 1 else nm)
    toks = tokenizer(texts).to(device)
    feat = clip_model.encode_text(toks)
    feat = F.normalize(feat, dim=-1)
    per_label = []
    if setting == "part_plus_cat":
        prompts_per_label = len(PART_PLUS_CAT_TEMPLATES)
    else:
        prompts_per_label = len(PART_ONLY_TEMPLATES)
    for i in range(len(names)):
        chunk = feat[i * prompts_per_label : (i + 1) * prompts_per_label]
        per_label.append(F.normalize(chunk.mean(dim=0, keepdim=True), dim=-1))
    return torch.cat(per_label, dim=0)


def load_points(path):
    ext = path.suffix.lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        if "points" in data:
            pts = data["points"]
        elif "xyz" in data:
            pts = data["xyz"]
        else:
            raise ValueError("NPZ must contain 'points' or 'xyz'")
        pts = np.asarray(pts, dtype=np.float32)
        return pts[:, :3]
    if ext == ".ply":
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.size == 0:
            raise ValueError("PLY contains no points")
        return pts[:, :3]
    raise ValueError("Unsupported input format (use .npz or .ply)")


def load_labels(path, labels_str, npz_path):
    if path:
        p = Path(path)
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if labels_str:
        return [x.strip() for x in labels_str.split(",") if x.strip()]
    if npz_path.suffix.lower() == ".npz":
        data = np.load(npz_path, allow_pickle=True)
        if "label_names" in data:
            return [str(x) for x in data["label_names"].tolist()]
    return []


def render_points(points, labels, label_names, out_path):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = np.array([cmap(int(i) % 20)[:3] for i in labels], dtype=np.float32)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=4, linewidths=0)
    ax.set_axis_off()
    handles = []
    for i, name in enumerate(label_names):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=name, markerfacecolor=cmap(i % 20), markersize=6))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser("PatchAlign3D inference (stage-2 ckpt)")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="Path to .npz or .ply")
    p.add_argument("--label_list", type=str, default=None, help="Text file with one label per line")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated labels")
    p.add_argument("--text_setting", type=str, default="part_only", choices=["part_only", "part_plus_cat"])
    p.add_argument("--clip_model", type=str, default="ViT-bigG-14")
    p.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
    p.add_argument("--clip_tau", type=float, default=0.07)
    p.add_argument("--assign", type=str, default="nearest", choices=["nearest", "membership"])
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--num_group", type=int, default=128)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--out", type=str, default=None, help="Output .npz path")
    p.add_argument("--render", action="store_true", default=False, help="Save a PNG visualization")
    p.add_argument("--render_path", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_path = Path(args.input)
    points = load_points(in_path)
    label_names = load_labels(args.label_list, args.labels, in_path)
    if not label_names:
        raise RuntimeError("No labels provided. Use --label_list or --labels (or include label_names in NPZ).")

    clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_dim = int(getattr(clip_model, "text_projection", None).shape[1] if hasattr(clip_model, "text_projection") else 512)

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
    proj = PatchToTextProj(in_dim=384, out_dim=text_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"], strict=False)

    print("[info] Points are expected in original coordinates; Y/Z are swapped to match training.")
    pts = torch.as_tensor(points, dtype=torch.float32).unsqueeze(0)  # (1,N,3)
    pts = prepare_points(pts).to(device)

    with torch.no_grad():
        patch_emb, patch_centers, patch_idx = model.forward_patches(pts)
        patch_feat = proj(patch_emb)
        text_feats = encode_texts(label_names, args.text_setting, clip_model, tokenizer, device)
        logits = (patch_feat @ text_feats.t()) / max(float(args.clip_tau), 1e-6)
        point_logits = assign_points_from_patches(pts[:, :3, :], patch_centers, logits, patch_idx, mode=args.assign)
        pred = point_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(np.int64)

    out_path = Path(args.out) if args.out else in_path.with_suffix("").with_name(in_path.stem + "_pred.npz")
    np.savez(out_path, points=points, pred=pred, label_names=np.array(label_names, dtype=object))
    print(f"[saved] {out_path}")

    if args.render:
        render_path = Path(args.render_path) if args.render_path else out_path.with_suffix(".png")
        render_points(points, pred, label_names, render_path)
        print(f"[saved] {render_path}")


if __name__ == "__main__":
    main()
