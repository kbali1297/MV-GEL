from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Default root can be overridden by CLI
DATA_ROOT = "/path/to/data_root"


def _gather_item_ids(root):
    rendered = root / "labeled" / "rendered"
    if not rendered.exists():
        raise FileNotFoundError(f"Rendered directory not found: {rendered}")
    return sorted([p.name for p in rendered.iterdir() if p.is_dir()])


def _read_split_file(root, split):
    """Read labeled/split/<split>.txt if it exists."""
    path = root / "labeled" / "split" / f"{split}.txt"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _paths_for_item(root, item_id):
    """Return (points_dir, mask2points, mask_labels) for an item id (folder name under labeled/rendered)."""
    uid = item_id.split('_')[-1]
    pts_dir = root / "labeled" / "points" / uid
    mdir = root / "labeled" / "rendered" / item_id / "oriented" / "masks" / "merged"
    return pts_dir, mdir / "mask2points.pt", mdir / "mask_labels.txt"


def _filter_existing(root, item_ids):
    """Filter item ids to those with required files present (points + merged masks)."""
    keep = []
    for item_id in item_ids:
        pts_dir, m2p, labels = _paths_for_item(root, item_id)
        if pts_dir.exists() and (pts_dir / "points.pt").exists() and m2p.exists() and labels.exists():
            keep.append(item_id)
    return keep


def _load_mask_labels(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _deterministic_split(items, split, ratio):
    ratio = max(0.0, min(1.0, ratio))
    n = len(items)
    pivot = int(ratio * n)
    if split == "train":
        return list(items[:pivot])
    if split == "val":
        return list(items[pivot:])
    return list(items)


def _fps_cpu(xyz, k, gen):
    xyz = xyz.detach().cpu()
    N = xyz.shape[0]
    if k >= N:
        return torch.arange(N, dtype=torch.long)
    idx = torch.zeros(k, dtype=torch.long)
    start = torch.randint(0, N, (1,), generator=gen).item()
    idx[0] = start
    dist = torch.full((N,), float('inf'))
    for i in range(1, k):
        p = xyz[idx[i - 1]]
        d = ((xyz - p) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        idx[i] = torch.argmax(dist).item()
    return idx


def _mask_aware_indices(points, label_masks, total, min_per_label, seed):
    pts = points.detach().cpu()
    labels = label_masks.detach().cpu()
    total = min(total, pts.shape[0])
    chosen = torch.zeros(pts.shape[0], dtype=torch.bool)
    gen = torch.Generator().manual_seed(seed)

    for mask in labels:
        idx_mask = torch.nonzero(mask, as_tuple=True)[0]
        if idx_mask.numel() == 0:
            continue
        take = min(min_per_label, idx_mask.numel())
        if take <= 0:
            continue
        sel = _fps_cpu(pts[idx_mask], take, gen)
        chosen[idx_mask[sel]] = True

    chosen_idx = torch.nonzero(chosen, as_tuple=True)[0]
    if chosen_idx.numel() < total:
        remaining = torch.nonzero(~chosen, as_tuple=True)[0]
        need = min(total - chosen_idx.numel(), remaining.numel())
        if need > 0:
            sel = _fps_cpu(pts[remaining], need, gen)
            chosen_idx = torch.cat([chosen_idx, remaining[sel]])

    if chosen_idx.numel() > total:
        sel = _fps_cpu(pts[chosen_idx], total, gen)
        chosen_idx = chosen_idx[sel]

    return torch.sort(chosen_idx).values


class TrainingSetDataset(Dataset):
    """Loads training-set objects and returns subsampled point clouds with labels."""

    def __init__(self,
                 root=None,
                 split="train",
                 id_list_file=None,
                 npoints=2048,
                 train_ratio=0.9,
                 min_per_label=32,
                 seed=42,
                 filter_missing=True,
                 random_subsample=False):
        self.root = Path(root) if root is not None else Path(DATA_ROOT)
        if id_list_file:
            with open(id_list_file, 'r', encoding='utf-8') as f:
                item_ids = [line.strip() for line in f if line.strip()]
        else:
            from_split = _read_split_file(self.root, split)
            if from_split is not None:
                item_ids = from_split
            else:
                item_ids = _gather_item_ids(self.root)
        if not item_ids:
            raise RuntimeError("No training-set objects found")
        if id_list_file or _read_split_file(self.root, split) is not None:
            chosen = item_ids
        else:
            chosen = _deterministic_split(item_ids, split, train_ratio)
        self._raw_count = len(chosen)
        self.samples = _filter_existing(self.root, chosen) if filter_missing else list(chosen)
        self._dropped = self._raw_count - len(self.samples)
        if not self.samples:
            raise RuntimeError(f"No samples for split {split}")
        self.npoints = npoints
        self.min_per_label = min_per_label
        self.seed = seed
        self.random_subsample = random_subsample

    def __len__(self):
        return len(self.samples)

    def _load_points(self, uid):
        pts_path = self.root / "labeled" / "points" / uid / "points.pt"
        pts = torch.load(pts_path, map_location='cpu').float()
        if pts.ndim == 2:
            return pts
        return pts.reshape(-1, pts.shape[-1])

    def __getitem__(self, idx: int):
        item_id = self.samples[idx]
        uid = item_id.split('_')[-1]
        base = self.root / "labeled" / "rendered" / item_id / "oriented"
        merged_dir = base / "masks" / "merged"

        mask2points = torch.load(merged_dir / "mask2points.pt", map_location='cpu').bool()
        label_rows = _load_mask_labels(merged_dir / "mask_labels.txt")
        if mask2points.shape[0] != len(label_rows):
            m = min(mask2points.shape[0], len(label_rows))
            mask2points = mask2points[:m]
            label_rows = label_rows[:m]

        label_to_idx = OrderedDict()
        union_masks = []
        for mask, name in zip(mask2points, label_rows):
            if not name:
                continue
            if name in label_to_idx:
                union_masks[label_to_idx[name]] |= mask
            else:
                label_to_idx[name] = len(union_masks)
                union_masks.append(mask.clone())
        if not union_masks:
            raise RuntimeError(f"No valid labels for {item_id}")

        label_masks = torch.stack(union_masks)
        label_names = list(label_to_idx.keys())

        points = self._load_points(uid)
        point_feat = points

        if self.random_subsample:
            N = point_feat.shape[0]
            if self.npoints >= N:
                idx_sel = torch.arange(N, dtype=torch.long)
            else:
                perm = torch.randperm(N)
                idx_sel = perm[: self.npoints].long()
        else:
            idx_sel = _mask_aware_indices(points, label_masks, self.npoints,
                                          self.min_per_label, self.seed + idx)
        pts_sel = point_feat[idx_sel]
        label_masks_sel = label_masks[:, idx_sel]
        point_labels = torch.full((idx_sel.numel(),), -1, dtype=torch.long)
        for lid, mask in enumerate(label_masks_sel):
            point_labels[mask] = lid

        sample = {
            'item_id': item_id,
            'idx_sel': idx_sel.clone(),
            'points': pts_sel,
            'point_labels': point_labels,
            'label_masks': label_masks_sel,
            'label_names': label_names
        }
        return sample


def collate_trainset(batch):
    return batch
