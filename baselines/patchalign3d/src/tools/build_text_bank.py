#!/usr/bin/env python3
"""
Build offline text feature banks for training-set part labels.

Outputs (by default):
  <data_root>/labeled/text_banks/textbank_part_only_<stamp>.pt
  <data_root>/labeled/text_banks/textbank_part_plus_cat_<stamp>.pt  (when --mode includes part_plus_cat)
"""

import argparse
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip

from patchalign3d.datasets import trainset as ds

# Prompt templates
PART_ONLY_TEMPLATES = ["{}", "a {}", "{} part"]
PART_PLUS_CAT_TEMPLATES = ["a {} of a {}", "the {} of a {}", "{} of {}", "a {} part of a {}"]


def clean_text(s):
    s = s.strip().lower().replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_item_category(item_id):
    return item_id.rsplit("_", 1)[0] if "_" in item_id else item_id


def read_list_file(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"List file not found: {p}")
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def enumerate_items(data_root, train_list, val_list, filter_missing=True):
    items = []
    for lst in [train_list, val_list]:
        if lst:
            items.extend(read_list_file(lst))
    if not items:
        pool = []
        for sp in ["train", "val"]:
            from_split = ds._read_split_file(data_root, sp)
            if from_split is not None:
                pool.extend(from_split)
        if not pool:
            pool = ds._gather_item_ids(data_root)
        items = pool
    if filter_missing:
        items = ds._filter_existing(data_root, items)
    return sorted(set(items))


def collect_unique_labels(data_root, item_ids):
    seen_names = []
    seen_pairs = []
    seen_name_set = set()
    seen_pair_set = set()
    for item_id in item_ids:
        _, _, labels_path = ds._paths_for_item(data_root, item_id)
        if not labels_path.exists():
            continue
        label_rows = ds._load_mask_labels(labels_path)
        cat = clean_text(parse_item_category(item_id))
        names_in_obj = [clean_text(n) for n in label_rows if n and n.strip()]
        for nm in names_in_obj:
            if nm and nm not in seen_name_set:
                seen_name_set.add(nm)
                seen_names.append(nm)
            pair = (nm, cat)
            if nm and pair not in seen_pair_set:
                seen_pair_set.add(pair)
                seen_pairs.append(pair)
    return seen_names, seen_pairs


def build_prompts_part_only(names):
    prompts = []
    owners = []
    for i, nm in enumerate(names):
        for tpl in PART_ONLY_TEMPLATES:
            txt = tpl.format(nm) if tpl.count("{}") == 1 else nm
            prompts.append(txt)
            owners.append(i)
    return prompts, owners


def build_prompts_part_plus_cat(pairs):
    prompts = []
    owners = []
    for i, (nm, cat) in enumerate(pairs):
        for tpl in PART_PLUS_CAT_TEMPLATES:
            slots = tpl.count("{}")
            if slots == 2:
                txt = tpl.format(nm, cat)
            elif slots == 1:
                txt = tpl.format(f"{cat} {nm}")
            else:
                txt = f"{cat} {nm}"
            prompts.append(txt)
            owners.append(i)
    return prompts, owners


def encode_prompts(prompts, tokenizer, model, device, batch):
    feats = []
    for i in tqdm(range(0, len(prompts), batch), desc="encode", ncols=80):
        chunk = prompts[i:i + batch]
        tokens = tokenizer(chunk).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        feats.append(emb.cpu())
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser("Build offline text feature banks (part-only / part+cat)")
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--train_list", type=str, default=None)
    ap.add_argument("--val_list", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="Defaults to <data_root>/labeled/text_banks")
    ap.add_argument("--mode", type=str, default="both", choices=["part_only", "part_plus_cat", "both"])
    ap.add_argument("--clip_model", type=str, default="ViT-B-16")
    ap.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    ap.add_argument("--batch_texts", type=int, default=256)
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cuda:0', or 'cpu'")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, _, _ = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    proj = getattr(model, "text_projection", None)
    text_dim = int(proj.shape[1]) if proj is not None and hasattr(proj, "shape") else 512

    root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "labeled" / "text_banks"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids = enumerate_items(root, args.train_list, args.val_list, filter_missing=True)
    print(f"[info] items={len(item_ids)} | mode={args.mode}")

    names, pairs = collect_unique_labels(root, item_ids)
    print(f"[info] unique part-only labels={len(names)} | part+cat pairs={len(pairs)}")

    if args.mode in ("part_only", "both"):
        prompts, owners = build_prompts_part_only(names)
        feats = encode_prompts(prompts, tokenizer, model, device, args.batch_texts)
        owners_t = torch.tensor(owners, dtype=torch.long)
        pooled = []
        for i in range(len(names)):
            mask = owners_t == i
            pooled.append(F.normalize(feats[mask].mean(dim=0, keepdim=True), dim=-1))
        bank = torch.cat(pooled, dim=0)
        path_po = out_dir / "textbank_part_only.pt"
        torch.save(
            {"keys": names, "emb": bank, "meta": {"text_dim": int(bank.shape[1]), "backend": "clip", "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained}},
            path_po,
        )
        print(f"[saved] {path_po}")

    if args.mode in ("part_plus_cat", "both"):
        prompts, owners = build_prompts_part_plus_cat(pairs)
        feats = encode_prompts(prompts, tokenizer, model, device, args.batch_texts)
        owners_t = torch.tensor(owners, dtype=torch.long)
        pooled = []
        keys = []
        for i, (nm, cat) in enumerate(pairs):
            mask = owners_t == i
            pooled.append(F.normalize(feats[mask].mean(dim=0, keepdim=True), dim=-1))
            keys.append(f"{nm}||{cat}")
        bank = torch.cat(pooled, dim=0)
        path_pc = out_dir / "textbank_part_plus_cat.pt"
        torch.save(
            {"keys": keys, "emb": bank, "meta": {"text_dim": int(bank.shape[1]), "backend": "clip", "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained}},
            path_pc,
        )
        print(f"[saved] {path_pc}")


if __name__ == "__main__":
    main()
