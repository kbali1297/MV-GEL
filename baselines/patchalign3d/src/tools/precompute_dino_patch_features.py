#!/usr/bin/env python3
"""
Precompute DINOv2 patch features.

Requires COPS (https://github.com/jianglongye/cops) cloned into PatchAlign3D/cops:
  python -m patchalign3d.tools.precompute_dino_patch_features --root <data_root> --split train

Outputs:
  <root>/labeled/rendered/<item_id>/oriented/patch_dino/patch_features.pt
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math

_repo = Path(__file__).resolve().parents[2]
cops_root = _repo / "cops"
source_dir = cops_root / "source"
if not source_dir.exists():
    raise RuntimeError("Clone COPS into PatchAlign3D/cops (it must contain source/).")
sys.path.append(str(cops_root))
sys.path.append(str(source_dir))
from point_cloud_utils.feature_interpolation import interpolate_feature_map, interpolate_point_cloud
from point_cloud_utils.backprojection import backproject


def build_preprocess(size):
    import torchvision.transforms as T
    bic = getattr(T.InterpolationMode, 'BICUBIC', 3)
    return T.Compose([
        T.Resize(size, interpolation=bic),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dinov2(hf_id, device, force_size=518):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(device).eval()
    size = int(getattr(model.config, 'image_size', force_size or 518))
    preprocess = build_preprocess(size)
    dim = int(getattr(model.config, 'hidden_size', 768))
    return model, preprocess, dim, size


def _trim_register_tokens_to_grid(last_hidden_state, model):
    """Ensure sequence is CLS + (H*W) patch tokens by dropping register tokens.

    - Some ViTs (DINOv2/v3) add a small number of register tokens after the patch tokens.
    - This trims those extra tokens so that (#tokens_without_cls) is a perfect square.

    Returns a new last_hidden_state with shape (B, 1 + H*W, D).
    """
    if last_hidden_state.ndim != 3 or last_hidden_state.size(1) < 2:
        return last_hidden_state
    cls = last_hidden_state[:, :1, :]
    tokens = last_hidden_state[:, 1:, :]
    cfg = getattr(model, 'config', None)
    nreg = 0
    if cfg is not None:
        # Try common names for register tokens
        nreg = int(getattr(cfg, 'num_register_tokens', 0) or getattr(cfg, 'num_reg_tokens', 0) or 0)
    if nreg > 0 and tokens.shape[1] > nreg:
        tokens_wo_reg = tokens[:, :-nreg, :]
    else:
        tokens_wo_reg = tokens
    L = tokens_wo_reg.shape[1]
    # Floor to nearest square if still not square
    ws = int(math.isqrt(L))
    grid = ws * ws
    if grid <= 0:
        return last_hidden_state
    if grid != L:
        tokens_wo_reg = tokens_wo_reg[:, :grid, :]
    return torch.cat([cls, tokens_wo_reg], dim=1)


def oriented_paths(root, item_id):
    base = root / 'labeled' / 'rendered' / item_id / 'oriented'
    imgs = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        if (base / 'imgs').exists():
            imgs += list((base / 'imgs').rglob(ext))
        imgs += list(base.rglob(ext))
    uniq, seen = [], set()
    for p in imgs:
        if not p.is_file():
            continue
        name = p.name.lower()
        if ('mask' in name and 'overlay' not in name):
            continue
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    pix2face = base / 'pix2face.pt'
    uid = item_id.split('_')[-1]
    p2f = root / 'labeled' / 'points' / uid / 'point2face.pt'
    pts_path = root / 'labeled' / 'points' / uid / 'points.pt'
    return base, uniq, pix2face, p2f, pts_path


def view_index_from_name(path):
    stem = path.stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits.isdigit() else -1


def build_face2points(point2face):
    # representative point per face id (first occurrence)
    f2p = {}
    f = point2face.view(-1).cpu().numpy()
    for p, fid in enumerate(f):
        if fid < 0:
            continue
        fid = int(fid)
        if fid not in f2p:
            f2p[fid] = p
    return f2p


def face_map_to_point_map(face_map, face2pts):
    if face_map.size == 0:
        return np.empty_like(face_map, dtype=np.int64)
    if not face2pts:
        return np.full(face_map.shape, -1, dtype=np.int64)
    max_f = int(max(face2pts.keys()))
    lut = np.full(max_f + 1, -1, dtype=np.int64)
    for fid, pid in face2pts.items():
        if fid >= 0:
            if fid >= lut.size:
                # grow LUT if necessary
                lut = np.pad(lut, (0, fid - lut.size + 1), constant_values=-1)
            lut[fid] = pid
    pm = np.full(face_map.shape, -1, dtype=np.int64)
    mask = (face_map >= 0) & (face_map < lut.size)
    pm[mask] = lut[face_map[mask]]
    return pm


def fps_np(xyz, k, seed=0):
    N = xyz.shape[0]
    k = min(k, N)
    rng = np.random.RandomState(seed)
    idx = np.zeros(k, dtype=np.int64)
    farthest = rng.randint(0, N)
    dist = np.full(N, np.inf, dtype=np.float64)
    for i in range(k):
        idx[i] = farthest
        centroid = xyz[farthest:farthest + 1]
        d = np.sum((xyz - centroid) ** 2, axis=1)
        dist = np.minimum(dist, d)
        farthest = int(np.argmax(dist))
    return idx


@torch.no_grad()
def process_item(root, item_id, model, preprocess, img_size, d_dim, device,
                 num_views, G, M, out_dir_name='patch_dino', view_batch=4,
                 resume=True, verbose=True):
    base, views, pix2face_path, p2f_path, pts_path = oriented_paths(root, item_id)
    out_dir = base / out_dir_name
    out_path = out_dir / 'patch_features.pt'
    if resume and out_path.exists():
        if verbose:
            print(f"[skip-exist] {item_id}", flush=True)
        return True
    if not (pix2face_path.exists() and p2f_path.exists() and pts_path.exists()):
        print(f"[skip-missing] {item_id}", flush=True)
        return False
    pix2face = torch.load(pix2face_path, map_location='cpu').numpy()
    point2face = torch.load(p2f_path, map_location='cpu')
    pts = torch.load(pts_path, map_location='cpu').float()
    if pts.ndim != 2:
        pts = pts.reshape(-1, pts.shape[-1])
    if pts.shape[1] >= 3:
        y = pts[:, 1].clone(); pts[:, 1] = pts[:, 2]; pts[:, 2] = y

    views = [p for p in views if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    views = sorted(views)[:max(1, num_views)]
    if not views:
        print(f"[skip-noviews] {item_id}", flush=True)
        return False

    if verbose:
        print(f"[item] {item_id} | views={len(views)} | points={pts.shape[0]}", flush=True)

    face2pts = build_face2points(point2face)

    vb = max(1, int(view_batch))
    feat_sum = torch.zeros((pts.shape[0], d_dim), dtype=torch.float32, device=device)
    feat_cnt = torch.zeros((pts.shape[0],), dtype=torch.float32, device=device)

    for i in range(0, len(views), vb):
        chunk = views[i:i + vb]
        imgs_b, maps_b = [], []
        for vp in chunk:
            vidx = view_index_from_name(vp)
            if not (0 <= vidx < pix2face.shape[0]):
                continue
            face_map = pix2face[vidx]
            H0, W0 = face_map.shape
            s = img_size / min(H0, W0)
            newH, newW = int(round(H0 * s)), int(round(W0 * s))
            img_face = Image.fromarray(face_map.astype(np.int32), mode='I')
            img_face = img_face.resize((newW, newH), resample=Image.NEAREST)
            arr = np.array(img_face)
            top = max(0, (newH - img_size) // 2); left = max(0, (newW - img_size) // 2)
            face_rc = arr[top:top + img_size, left:left + img_size]
            point_map = face_map_to_point_map(face_rc, face2pts)
            maps_b.append(torch.from_numpy(point_map).long())
            img = Image.open(vp).convert('RGB')
            imgs_b.append(preprocess(img))
        if not imgs_b:
            continue
        X = torch.stack(imgs_b, dim=0).to(device)
        out = model(pixel_values=X)
        last = out.last_hidden_state
        # Trim any register tokens so the patch length forms a square grid
        last = _trim_register_tokens_to_grid(last, model)
        feat_pix = interpolate_feature_map(last, width=img_size, height=img_size, mode='bicubic')  # (B,H,W,D)
        for j in range(feat_pix.shape[0]):
            pmap = maps_b[j].to(device)
            pf = feat_pix[j].to(device)
            pcd_feat = backproject(pmap, pts[:, :3].to(device), pf, device=device.type)
            valid = ~torch.all(pcd_feat == 0.0, dim=-1)
            feat_sum[valid] += pcd_feat[valid]
            feat_cnt[valid] += 1.0
        if verbose:
            print(f"  [views {i}-{i+len(imgs_b)-1}] processed", flush=True)

    cnt = feat_cnt.clamp_min(1.0)
    feat_points = (feat_sum / cnt.unsqueeze(-1)).float().cpu()
    if (feat_cnt == 0).any():
        feat_points = interpolate_point_cloud(pts[:, :3].cpu(), feat_points.cpu(), neighbors=20)
    feat_points = F.normalize(feat_points.float(), dim=-1)

    # Patches
    xyz = pts[:, :3].cpu().numpy()
    centers_idx = fps_np(xyz, k=G, seed=0)
    D = torch.cdist(torch.from_numpy(xyz[centers_idx]).float(), torch.from_numpy(xyz).float()).cpu().numpy()
    patch_idx = np.argpartition(D, M, axis=1)[:, :M]
    point2center = D.argmin(axis=0).astype(np.int64)

    patch_feats = torch.zeros((patch_idx.shape[0], d_dim), dtype=torch.float32)
    for g in range(patch_idx.shape[0]):
        ids = torch.from_numpy(patch_idx[g])
        patch_feats[g] = F.normalize(feat_points[ids].mean(dim=0), dim=-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'item_id': item_id,
        'feature_dim': int(d_dim),
        'img_size': int(img_size),
        'centers_idx': torch.from_numpy(centers_idx).long(),
        'centers_xyz': torch.from_numpy(xyz[centers_idx]).float(),
        'patch_idx': torch.from_numpy(patch_idx).long(),
        'point2center': torch.from_numpy(point2center).long(),
        'patch_feats': patch_feats.half(),
    }, out_path)
    if verbose:
        print(f"[done] {item_id} → {str(out_path)}", flush=True)
    return True


def read_split(root, split):
    f = root / 'labeled' / 'split' / f'{split}.txt'
    if f.exists():
        with f.open('r', encoding='utf-8') as fh:
            return [s.strip() for s in fh if s.strip()]
    rendered = root / 'labeled' / 'rendered'
    return sorted([p.name for p in rendered.iterdir() if p.is_dir()])


def main():
    ap = argparse.ArgumentParser('Precompute DINO patch features (single-process)')
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', default='train', help="split name (uses labeled/split/<name>.txt) or 'all'")
    ap.add_argument('--hf_id', default='facebook/dinov2-base')
    ap.add_argument('--force_size', type=int, default=518)
    ap.add_argument('--num_views', type=int, default=4)
    ap.add_argument('--view_batch', type=int, default=4)
    ap.add_argument('--G', type=int, default=128)
    ap.add_argument('--M', type=int, default=32)
    ap.add_argument('--out_dir_name', type=str, default='patch_dino', help='Subfolder name to save patch features')
    ap.add_argument('--device', type=str, default='cuda:0', help="e.g., 'cuda:0' or 'cpu'")
    ap.add_argument('--resume', action='store_true', default=True)
    ap.add_argument('--verbose', action='store_true', default=True)
    args = ap.parse_args()

    root = Path(args.root)

    def log_line(prefix, msg):
        print(f"[{prefix}] {msg}", flush=True)
    if args.split == 'all':
        item_ids = read_split(root, 'all')
        if not item_ids:
            item_ids = list(dict.fromkeys(read_split(root, 'train') + read_split(root, 'val')))
    else:
        item_ids = read_split(root, args.split)

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    model, preprocess, d_dim, img_size = load_dinov2(args.hf_id, device, force_size=args.force_size)
    log_line('start', f"total={len(item_ids)} device={device}")
    done = 0; skipped = 0; t0 = time.time()
    for item_id in item_ids:
        ok = process_item(root, item_id, model=model, preprocess=preprocess, img_size=img_size, d_dim=d_dim,
                          device=device, num_views=args.num_views, G=args.G, M=args.M,
                          out_dir_name=args.out_dir_name, view_batch=args.view_batch,
                          resume=args.resume, verbose=args.verbose)
        if ok:
            done += 1
        else:
            skipped += 1
        dt = max(1e-6, time.time() - t0)
        rate = (done + skipped) / dt
        rem = len(item_ids) - (done + skipped)
        eta = time.strftime('%H:%M:%S', time.gmtime(int(rem / rate)))
        log_line('progress', f"done={done} skip={skipped} rate={rate:.2f}/s eta={eta}")
    log_line('finish', f"done={done} skip={skipped}")


if __name__ == '__main__':
    main()
