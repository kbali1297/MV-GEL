#!/usr/bin/env python
"""
Aggregate the 3 zero-shot baselines (PartSLIP, Find3D, PatchAlign3D), each in its
Default and Top-PCD variant, over the corrected 1535-entity test set.

Per config: gather all per-shard localization_metrics_*.log entity dicts, dedup by
(cad_file, feature, feature_idx), restrict to the 1535 allowlist, and average
IoU / Precision / Recall / F1 separately for face and edge.

Emits a CSV (means_baselines_1535.csv) and prints LaTeX-ready rows.
"""
import os
import ast
import glob
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
# Baseline run outputs (GeLoM_*) are written next to each vendored baseline repo.
PARTSLIP_DIR = os.path.join(ROOT, "baselines", "partslip")
FIND3D_DIR = os.path.join(ROOT, "baselines", "find3d")
PATCHALIGN_DIR = os.path.join(ROOT, "baselines", "patchalign3d")
ALLOWLIST = os.path.join(ROOT, "configs", "val_dataset_1535_entities.txt")

# (label, list of globs of per-shard logs)
CONFIGS = [
    ("PartSLIP (Default)", [
     os.path.join(PARTSLIP_DIR, "GeLoM_PartSLIP_default_1535_shard*", "localization_metrics_partslip.log")]),
    ("PartSLIP (Top-PCD)", [
     os.path.join(PARTSLIP_DIR, "GeLoM_PartSLIP_toppcd_1535_shard*", "localization_metrics_partslip.log")]),
    ("Find3D (Default)", [
     os.path.join(FIND3D_DIR, "GeLoM_Find3D_default_1535_shard*", "localization_metrics_find3d.log"),
     os.path.join(FIND3D_DIR, "GeLoM_Find3D_default_1535_recover", "localization_metrics_find3d.log")]),
    ("Find3D (Top-PCD)", [
     os.path.join(FIND3D_DIR, "GeLoM_Find3D_toppcd_1535_shard*", "localization_metrics_find3d.log"),
     os.path.join(FIND3D_DIR, "GeLoM_Find3D_toppcd_1535_recover", "localization_metrics_find3d.log")]),
    ("PatchAlign3D (Default)", [
     os.path.join(PATCHALIGN_DIR, "GeLoM_PatchAlign3D_default_1535_shard*", "localization_metrics_patchalign3d.log"),
     os.path.join(PATCHALIGN_DIR, "GeLoM_PatchAlign3D_default_1535_recover", "localization_metrics_patchalign3d.log")]),
    ("PatchAlign3D (Top-PCD)", [
     os.path.join(PATCHALIGN_DIR, "GeLoM_PatchAlign3D_toppcd_1535_shard*", "localization_metrics_patchalign3d.log"),
     os.path.join(PATCHALIGN_DIR, "GeLoM_PatchAlign3D_toppcd_1535_recover", "localization_metrics_patchalign3d.log")]),
]

METRICS = ("iou", "precision", "recall", "F1")


def load_allowlist():
    keys = set()
    with open(ALLOWLIST) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            c, ft, idx = ln.split(",")
            keys.add((c.strip(), ft.strip(), int(idx)))
    return keys


def parse_config(patterns, allow):
    """Return dict: key -> metric dict, deduped, restricted to allowlist."""
    by_key = {}
    files = []
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{") or "'feature'" not in line:
                    continue
                try:
                    d = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
                key = (str(d["cad_file"]).strip(), d["feature"].strip(), int(d["feature_idx"]))
                if allow and key not in allow:
                    continue
                # last write wins (shards are disjoint so collisions are rare)
                by_key[key] = d
    return by_key, len(files)


def averages(by_key):
    acc = {"face": {m: 0.0 for m in METRICS}, "edge": {m: 0.0 for m in METRICS}}
    cnt = {"face": 0, "edge": 0}
    for (cad, feat, idx), d in by_key.items():
        if feat not in acc:
            continue
        cnt[feat] += 1
        for m in METRICS:
            acc[feat][m] += float(d[m])
    for feat in acc:
        if cnt[feat]:
            for m in METRICS:
                acc[feat][m] /= cnt[feat]
    return acc, cnt


def main():
    allow = load_allowlist()
    n_face_target = sum(1 for k in allow if k[1] == "face")
    n_edge_target = sum(1 for k in allow if k[1] == "edge")
    print(f"[allowlist] {len(allow)} entities ({n_face_target} face, {n_edge_target} edge)\n")

    rows = []
    for label, pattern in CONFIGS:
        by_key, nfiles = parse_config(pattern, allow)
        acc, cnt = averages(by_key)
        rows.append((label, acc, cnt))
        print(f"=== {label} ===")
        print(f"  shards={nfiles}  matched={len(by_key)}/{len(allow)}  "
              f"(face {cnt['face']}/{n_face_target}, edge {cnt['edge']}/{n_edge_target})")
        for feat in ("face", "edge"):
            a = acc[feat]
            print(f"  {feat:4s}: IoU={a['iou']:.4f}  P={a['precision']:.4f}  "
                  f"R={a['recall']:.4f}  F1={a['F1']:.4f}")
        print()

    # CSV
    out_csv = os.path.join(ROOT, "means_baselines_1535.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "feature", "n", "IoU", "Precision", "Recall", "F1"])
        for label, acc, cnt in rows:
            for feat in ("face", "edge"):
                a = acc[feat]
                w.writerow([label, feat, cnt[feat],
                            f"{a['iou']:.4f}", f"{a['precision']:.4f}",
                            f"{a['recall']:.4f}", f"{a['F1']:.4f}"])
    print(f"[saved] {out_csv}\n")

    # LaTeX-ready rows: method & face(IoU P R F1) & edge(IoU P R F1)
    print("% LaTeX rows (face: IoU P R F1 | edge: IoU P R F1)")
    for label, acc, cnt in rows:
        fa, ed = acc["face"], acc["edge"]
        print(f"{label} & "
              f"{fa['iou']:.3f} & {fa['precision']:.3f} & {fa['recall']:.3f} & {fa['F1']:.3f} & "
              f"{ed['iou']:.3f} & {ed['precision']:.3f} & {ed['recall']:.3f} & {ed['F1']:.3f} \\\\")


if __name__ == "__main__":
    main()
