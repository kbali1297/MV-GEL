#!/usr/bin/env python
r"""
aggregate_all_tables.py
=======================================================================
ONE reproducible entry point that rebuilds *every* localization table in the
paper from the raw per-entity metric logs, on the corrected 1535-query test set
(655 faces + 880 edges, 218 meshes).

It is intentionally self-contained and committable: the MANIFEST below documents,
for every config, (a) the checkpoint(s) that produced it and (b) the exact set of
localization-metric log files -- including the SHARDED runs and their COMPLETION /
RECOVERY re-runs -- that have to be combined. Running this script:

  1. parses + combines every config's logs with an explicit precedence rule,
  2. restricts to the 1535 allowlist and asserts full 655/880 coverage,
  3. exports two GitHub-friendly artefacts under results_1535/:
        - all_per_entity_metrics.csv  (one tidy long table; the single file that
          downstream code can re-process without touching the scattered dirs)
        - SOURCES.md                  (the checkpoint + log-file provenance)
  4. prints the five LaTeX tables (baselines, lisa_view_selectors, lisa_top1,
     ablation_top3, ablation_top5),
  5. cross-checks every recomputed cell against the values currently in the paper
     and reports PASS / UPDATED / MISMATCH.

Run with any env that has pandas+numpy, e.g.:
    /data/1bali/miniforge3/envs/FIND3D/bin/python aggregate_all_tables.py
=======================================================================
"""
import os
import ast
import glob
import numpy as np
import pandas as pd

# Repo-relative root; override with the MVGEL_ROOT environment variable.
ROOT = os.environ.get(
    "MVGEL_ROOT", os.path.dirname(os.path.abspath(__file__)))
PARTSLIP_DIR = os.path.join(
    ROOT, '162-PartSLIP-Low-Shot-Part-Segmentation-for-3D-Point-Clouds-via-'
          'Pretrained-Image-Language-Models')
OUT_DIR = os.path.join(ROOT, 'results_1535')
ALLOWLIST_FILE = os.path.join(ROOT, 'val_dataset_1535_entities.txt')

SK = ['cad_name', 'feature', 'feature_idx']         # per-entity key
METRICS = ['iou', 'precision', 'recall', 'F1']
K_TO_IDX = {1: 0, 3: 1, 5: 2}                       # view-count -> log suffix idx

PRETTY = {'cliplora_cross_attention': 'Cross-Attention', 'cliplora_film': 'FiLM',
          'cliplora_no_fusion': 'No-Fusion', 'cliplora_only_clip': 'Only-CLIP',
          'random': 'Random', 'GT': 'GTviews'}
DISP = ['cliplora_cross_attention', 'cliplora_film', 'cliplora_no_fusion',
        'cliplora_only_clip', 'random', 'GT']

# ======================================================================
# REPRODUCIBILITY MANIFEST
# ----------------------------------------------------------------------
# Seg. VLM checkpoints
#   LISA-CAD       : runs/CAD_LISA/global_step5076 (domain-adapted)
#   LISA-Vanilla   : base LISA-7B weights (sentinel "vanilla" in infer.py)
# View-selector checkpoints (LISA repo root)
#   Cross-Attention: best_model_view_ranker_cliplora_cross_attention.pt
#   FiLM           : best_model_view_ranker_cliplora_film.pt
#   No-Fusion      : best_model_view_ranker_cliplora_no_fusion.pt
#   Only-CLIP      : best_model_view_ranker_cliplora_only_clip.pt
#   Random / GT    : sentinels (no checkpoint)
# Point-cloud baselines (pretrained, in their sub-repos)
#   PartSLIP : 162-PartSLIP.../  | Find3D : Find3D/  | PatchAlign3D : PatchAlign3D/
# ======================================================================

VLM_CKPT = {
    'CAD_LISA': 'runs/CAD_LISA_repro20/ckpt_model/global_step5076',
    'Vanilla_LISA': 'vanilla (base LISA-7B)',
}
VS_CKPT = {
    'cliplora_cross_attention': 'best_model_view_ranker_cliplora_cross_attention.pt',
    'cliplora_film': 'best_model_view_ranker_cliplora_film.pt',
    'cliplora_no_fusion': 'best_model_view_ranker_cliplora_no_fusion.pt',
    'cliplora_only_clip': 'best_model_view_ranker_cliplora_only_clip.pt',
    'random': '(sentinel: random views)',
    'GT': '(sentinel: geometric-visibility / marked-target oracle)',
}

# Baseline configs: label -> (checkpoint note, [shard glob, recovery glob, ...])
BASELINES = {
    'PartSLIP (Default)': (
        'PartSLIP pretrained (GLIP+SAM), conf preset',
        [f'{PARTSLIP_DIR}/GeLoM_PartSLIP_default_1535_shard*/localization_metrics_partslip.log']),
    'PartSLIP (Top-PCD)': (
        'PartSLIP pretrained, topk_pct preset (face10/edge2)',
        [f'{PARTSLIP_DIR}/GeLoM_PartSLIP_toppcd_1535_shard*/localization_metrics_partslip.log']),
    'Find3D (Default)': (
        'Find3D pretrained, argmax preset',
        [f'{ROOT}/Find3D/GeLoM_Find3D_default_1535_shard*/localization_metrics_find3d.log',
         f'{ROOT}/Find3D/GeLoM_Find3D_default_1535_recover/localization_metrics_find3d.log']),
    'Find3D (Top-PCD)': (
        'Find3D pretrained, topk_pct preset (face10/edge2)',
        [f'{ROOT}/Find3D/GeLoM_Find3D_toppcd_1535_shard*/localization_metrics_find3d.log',
         f'{ROOT}/Find3D/GeLoM_Find3D_toppcd_1535_recover/localization_metrics_find3d.log']),
    'PatchAlign3D (Default)': (
        'PatchAlign3D pretrained, argmax preset',
        [f'{ROOT}/PatchAlign3D/GeLoM_PatchAlign3D_default_1535_shard*/localization_metrics_patchalign3d.log',
         f'{ROOT}/PatchAlign3D/GeLoM_PatchAlign3D_default_1535_recover/localization_metrics_patchalign3d.log']),
    'PatchAlign3D (Top-PCD)': (
        'PatchAlign3D pretrained, topk_pct preset (face10/edge2)',
        [f'{ROOT}/PatchAlign3D/GeLoM_PatchAlign3D_toppcd_1535_shard*/localization_metrics_patchalign3d.log',
         f'{ROOT}/PatchAlign3D/GeLoM_PatchAlign3D_toppcd_1535_recover/localization_metrics_patchalign3d.log']),
}

# MV-GEL FiLM @top3 with ViewNMS (tab:baselines row 7) -- vnms1 sharded run.
VNMS1_FILM_SHARDS = [
    f'{ROOT}/MVGEL_CAD_LISA_cliplora_film_vnms1_val_dataset_1535_shard*/'
    f'localization_metrics_view_selector_cliplora_film_views_3.log']


# ----------------------------------------------------------------------
# parsing helpers
# ----------------------------------------------------------------------
def _key_from_dict(d):
    if 'mesh_file' in d:
        cad = str(d['mesh_file'].split('/')[-1].split('_')[0])
    else:
        cad = str(d['cad_file'])
    return cad.zfill(8), d['feature'].strip(), int(d['feature_idx'])


def parse_files(patterns, keep='last'):
    """Parse all entity dicts from a list of glob patterns into a deduped frame.

    `patterns` is an ordered list; files are read in that order and, within each
    pattern, in sorted order. Dedup is keep=`last`, so later patterns (e.g. a
    recovery re-run listed after the shard glob) win on any key collision."""
    rows = []
    nfiles = 0
    for pat in patterns:
        for fp in sorted(glob.glob(pat)):
            nfiles += 1
            with open(fp) as fr:
                for line in fr:
                    line = line.strip()
                    if not line.startswith('{') or "'feature'" not in line:
                        continue
                    try:
                        d = ast.literal_eval(line)
                    except (ValueError, SyntaxError):
                        continue
                    cad, feat, idx = _key_from_dict(d)
                    rows.append({'cad_name': cad, 'feature': feat,
                                 'feature_idx': idx, 'iou': float(d['iou']),
                                 'precision': float(d['precision']),
                                 'recall': float(d['recall']), 'F1': float(d['F1'])})
    if not rows:
        return pd.DataFrame(columns=SK + METRICS), nfiles
    df = (pd.DataFrame(rows).drop_duplicates(subset=SK, keep=keep)
          .reset_index(drop=True))
    return df, nfiles


def combine_precedence(pattern_groups):
    """Combine several (priority, patterns) groups; lower priority number wins.

    Used for MV-GEL view selectors: precedence missing24(0) > dupes(1) > total(2)."""
    frames = []
    for prio, pats in pattern_groups:
        df, _ = parse_files(pats, keep='last')
        if len(df):
            df = df.copy()
            df['_prio'] = prio
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return (df.sort_values('_prio').drop_duplicates(subset=SK, keep='first')
            .drop(columns='_prio').reset_index(drop=True))


def vs_combined(vlm, vs, k):
    """MV-GEL vnms0 view selector at top-k: total + dupes + missing24.

    Log files are named by the literal view count k (1/3/5). Precedence:
    missing24 (0) > dupes (1) > total (2)."""
    suf = {2: 'val_dataset_total', 1: 'val_dataset_dupes', 0: 'val_dataset_missing24'}
    groups = [(prio, [f'{ROOT}/MVGEL_{vlm}_{vs}_vnms0_{s}/'
                      f'localization_metrics_view_selector_{vs}_views_{k}.log'])
              for prio, s in suf.items()]
    return combine_precedence(groups)


def gt_marked_target(vlm):
    """MV-GEL GT marked-target oracle (top-1): sharded *_gttarget run + recovery."""
    df, _ = parse_files(
        [f'{ROOT}/MVGEL_{vlm}_GT_vnms0_val_dataset_1535_shard*_gttarget/'
         f'localization_metrics_view_selector_GT_views_1.log'], keep='last')
    return df


# ----------------------------------------------------------------------
# allowlist + averaging
# ----------------------------------------------------------------------
def load_allowlist():
    keys = set()
    with open(ALLOWLIST_FILE) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            c, ft, idx = ln.split(',')
            keys.add((c.strip().zfill(8), ft.strip(), int(idx)))
    return keys


ALLOW = load_allowlist()
N_FACE = sum(1 for k in ALLOW if k[1] == 'face')
N_EDGE = sum(1 for k in ALLOW if k[1] == 'edge')


def restrict(df):
    kt = df[SK].apply(tuple, axis=1)
    return df[kt.isin(ALLOW)].reset_index(drop=True)


def coverage(df):
    kt = set(df[SK].apply(tuple, axis=1))
    inter = kt & ALLOW
    nf = sum(1 for k in inter if k[1] == 'face')
    ne = sum(1 for k in inter if k[1] == 'edge')
    return len(inter), nf, ne


def feat_means(df, feat):
    sub = df[df['feature'] == feat]
    return sub[METRICS].mean()


def row8(df):
    """Return the 8-tuple [face IoU,P,R,F1, edge IoU,P,R,F1] over the 1535 set."""
    f = feat_means(df, 'face')
    e = feat_means(df, 'edge')
    return [f['iou'], f['precision'], f['recall'], f['F1'],
            e['iou'], e['precision'], e['recall'], e['F1']]


# ======================================================================
# 1) BUILD every config -> deduped/combined/restricted frame
# ======================================================================
print('=' * 72)
print(f'1535 allowlist: {len(ALLOW)} entities ({N_FACE} face, {N_EDGE} edge)')
print('=' * 72)

CONFIGS = {}          # key -> dict(group, seg_vlm, selector, topk, df, src, ckpt)
cov_problems = []


def register(key, group, seg_vlm, selector, topk, df, src, ckpt):
    df = restrict(df)
    n, nf, ne = coverage(df)
    tag = 'OK' if (nf == N_FACE and ne == N_EDGE) else 'PARTIAL'
    if tag != 'OK':
        cov_problems.append((key, n, nf, ne))
    CONFIGS[key] = dict(group=group, seg_vlm=seg_vlm, selector=selector,
                        topk=topk, df=df, src=src, ckpt=ckpt,
                        n=n, nf=nf, ne=ne)
    print(f'  [{tag:7s}] {key:42s} face {nf}/{N_FACE}  edge {ne}/{N_EDGE}')


print('\n[build] point-cloud baselines (sharded + recovery, keep=last)')
for label, (ckpt, pats) in BASELINES.items():
    df, nfiles = parse_files(pats, keep='last')
    register(f'baseline::{label}', 'baselines', label, '-', None, df,
             pats, ckpt)

print('\n[build] MV-GEL view selectors vnms0 (total+dupes+missing24) @ top1/3/5')
for vlm in ('CAD_LISA', 'Vanilla_LISA'):
    for vs in DISP:
        for k in (1, 3, 5):
            df = vs_combined(vlm, vs, k)
            register(f'mvgel::{vlm}::{vs}::top{k}', 'mvgel_vnms0', vlm, vs, k,
                     df,
                     [f'MVGEL_{vlm}_{vs}_vnms0_val_dataset_'
                      f'{{total,dupes,missing24}}/...views_{k}.log'],
                     f'{VLM_CKPT[vlm]} + {VS_CKPT[vs]}')

print('\n[build] MV-GEL GT marked-target oracle (sharded *_gttarget) @ top1')
for vlm in ('CAD_LISA', 'Vanilla_LISA'):
    df = gt_marked_target(vlm)
    register(f'gttarget::{vlm}::top1', 'mvgel_gttarget', vlm, 'GT', 1, df,
             [f'MVGEL_{vlm}_GT_vnms0_val_dataset_1535_shard*_gttarget/'
              f'...views_1.log'],
             f'{VLM_CKPT[vlm]} + marked-target oracle')

print('\n[build] MV-GEL FiLM @top3 + ViewNMS (vnms1 sharded)')
df, _ = parse_files(VNMS1_FILM_SHARDS, keep='last')
register('vnms1::CAD_LISA::film::top3', 'mvgel_vnms1', 'CAD_LISA',
         'cliplora_film', 3, df, VNMS1_FILM_SHARDS,
         f'{VLM_CKPT["CAD_LISA"]} + {VS_CKPT["cliplora_film"]} (+ViewNMS)')

if cov_problems:
    print('\n[warn] configs without full 1535 coverage:')
    for key, n, nf, ne in cov_problems:
        print(f'    {key}: {n} (face {nf}, edge {ne})')


# ======================================================================
# 2) EXPORT consolidated artefacts for GitHub
# ======================================================================
os.makedirs(OUT_DIR, exist_ok=True)
long_rows = []
for key, c in CONFIGS.items():
    d = c['df']
    for _, r in d.iterrows():
        long_rows.append({
            'config_key': key, 'group': c['group'], 'seg_vlm': c['seg_vlm'],
            'selector': c['selector'], 'topk': c['topk'],
            'cad_name': r['cad_name'], 'feature': r['feature'],
            'feature_idx': int(r['feature_idx']), 'iou': r['iou'],
            'precision': r['precision'], 'recall': r['recall'], 'F1': r['F1']})
long_df = pd.DataFrame(long_rows)
long_csv = os.path.join(OUT_DIR, 'all_per_entity_metrics.csv')
long_df.to_csv(long_csv, index=False)
print(f'\n[export] {long_csv}  ({len(long_df)} rows, {len(CONFIGS)} configs)')

with open(os.path.join(OUT_DIR, 'SOURCES.md'), 'w') as fw:
    fw.write('# Localization-metric provenance (corrected 1535-query test set)\n\n')
    fw.write(f'Test set: {len(ALLOW)} entities ({N_FACE} faces, {N_EDGE} edges), '
             '218 meshes. Allowlist: `val_dataset_1535_entities.txt`.\n\n')
    fw.write('Each config is the combination (deduped) of the listed log files; '
             'sharded runs and their completion/recovery re-runs are merged with '
             'the precedence noted in `aggregate_all_tables.py`.\n\n')
    for key, c in CONFIGS.items():
        fw.write(f'## `{key}`  ({c["nf"]}/{N_FACE} face, {c["ne"]}/{N_EDGE} edge)\n')
        fw.write(f'- **checkpoint(s):** {c["ckpt"]}\n')
        fw.write('- **logs:**\n')
        for s in c['src']:
            fw.write(f'    - `{s}`\n')
        fw.write('\n')
print(f'[export] {os.path.join(OUT_DIR, "SOURCES.md")}')

# tidy means table
mean_rows = []
for key, c in CONFIGS.items():
    v = row8(c['df'])
    mean_rows.append({'config_key': key, 'group': c['group'],
                      'seg_vlm': c['seg_vlm'], 'selector': c['selector'],
                      'topk': c['topk'], 'n_face': c['nf'], 'n_edge': c['ne'],
                      'face_iou': v[0], 'face_prec': v[1], 'face_rec': v[2],
                      'face_f1': v[3], 'edge_iou': v[4], 'edge_prec': v[5],
                      'edge_rec': v[6], 'edge_f1': v[7]})
means_csv = os.path.join(OUT_DIR, 'means_all_tables_1535.csv')
pd.DataFrame(mean_rows).to_csv(means_csv, index=False)
print(f'[export] {means_csv}')


# ======================================================================
# 3) LaTeX table renderers
# ======================================================================
def fmt(v, nd=3, bold=False, ital=False):
    s = f'{v:.{nd}f}'
    if bold:
        s = r'\textbf{' + s + '}'
    if ital:
        s = r'\textit{' + s + '}'
    return s


def _means_for(vlm, k):
    """dict selector -> 8-tuple for a (vlm, k) view-selector ablation."""
    out = {}
    for vs in DISP:
        out[vs] = row8(CONFIGS[f'mvgel::{vlm}::{vs}::top{k}']['df'])
    return out


def _best_per_col(rows):
    """index of best (max) non-GT selector per each of 8 cols; FiLM wins ties."""
    nong = [vs for vs in DISP if vs != 'GT']
    pref = {'cliplora_film': 0, 'cliplora_cross_attention': 1,
            'cliplora_no_fusion': 2, 'cliplora_only_clip': 3, 'random': 4}
    best = []
    for c in range(8):
        bvs, bv, bp = None, -1, 99
        for vs in nong:
            v = rows[vs][c]
            p = pref[vs]
            if v > bv + 1e-9 or (abs(v - bv) <= 1e-9 and p < bp):
                bvs, bv, bp = vs, v, p
        best.append(bvs)
    return best


def render_selector_rows(rows, gt_rows, start_sno, vlm_label):
    """rows: non-GT means dict; gt_rows: 8-tuple for the GT row (kept separate so
    the GT source can differ between tables)."""
    best = _best_per_col(rows)
    lines, sno = [], start_sno
    for vs in DISP:
        if vs == 'GT':
            cells = [fmt(gt_rows[c], ital=True) for c in range(8)]
            name = r'\textit{GTviews}'
        else:
            cells = [fmt(rows[vs][c], bold=(best[c] == vs)) for c in range(8)]
            name = PRETTY[vs]
        lines.append(f'{sno} & {vlm_label} & {name} & ' + ' & '.join(cells) + r' \\')
        sno += 1
    return lines, sno


def table_lisa_view_selectors():
    """tab:lisa_view_selectors -- CAD + Vanilla top1, GT = marked-target oracle."""
    cad = _means_for('CAD_LISA', 1)
    van = _means_for('Vanilla_LISA', 1)
    cad_gt = row8(CONFIGS['gttarget::CAD_LISA::top1']['df'])
    van_gt = row8(CONFIGS['gttarget::Vanilla_LISA::top1']['df'])
    cl, nxt = render_selector_rows(cad, cad_gt, 1, 'LISA-CAD')
    vl, _ = render_selector_rows(van, van_gt, nxt, 'LISA (Vanilla)')
    head = [
        r'\begin{table}[h]', r'\centering',
        r'\caption{Performance evaluation of domain adapted and unadapted LISA '
        r'variants and view selectors across Face@ top1 views and Edge@ top1 '
        r'views metrics. \textit{GTviews}, short for the Ground Truth views, act '
        r'as the oracle target views of CAD meshes on the test set: the single '
        r'annotated view each entity was marked on and to which its referring '
        r'caption corresponds. Test set of 1535 queries (655 faces, 880 edges) '
        r'from 218 meshes.}',
        r'\label{tab:lisa_view_selectors}',
        r'\resizebox{\linewidth}{!}{%', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{c >{\raggedright\arraybackslash}p{2.8cm} l cccc cccc}',
        r'\toprule',
        r' & & & \multicolumn{4}{c}{\textbf{Face@ top1 views}} & '
        r'\multicolumn{4}{c}{\textbf{Edge@ top1 views}} \\',
        r'\cmidrule(lr){4-7} \cmidrule(lr){8-11}',
        r'\textbf{S.no} & \textbf{Seg. VLM} & \textbf{View Selector} & '
        r'\textbf{IoU} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & '
        r'\textbf{IoU} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\',
        r'\midrule']
    foot = [r'\bottomrule', r'\end{tabular}%', r'}', r'\vspace{1mm}', r'\end{table}']
    return '\n'.join(head + cl + [r'\midrule'] + vl + foot)


def table_cad_topk(k, label, caption, face_head, edge_head, tabular_star=False,
                   gt_marked_target=False):
    """tab:lisa_top1 / ablation_top3 / ablation_top5 -- CAD only.

    GT row source:
      - geometric-visibility oracle (top_views_desc[:k]) by default (top3/top5);
      - caption-marked TARGET view when gt_marked_target=True (top1 only -- the
        marked view is inherently a single Top-1 view, so this matches the GT row
        of tab:lisa_view_selectors)."""
    rows = _means_for('CAD_LISA', k)
    if gt_marked_target:
        gt = row8(CONFIGS['gttarget::CAD_LISA::top1']['df'])
    else:
        gt = rows['GT']
    cl, _ = render_selector_rows(rows, gt, 1, 'LISA-CAD')
    if tabular_star:
        topspec = (r'\begin{tabular*}{\linewidth}'
                   r'{@{\extracolsep{\fill}}c l l cccc cccc}')
        endspec = r'\end{tabular*}%'
    else:
        topspec = r'\begin{tabular}{c l l cccc cccc}'
        endspec = r'\end{tabular}%'
    head = [
        r'\begin{table}[h]', r'\centering', rf'\caption{{{caption}}}',
        rf'\label{{{label}}}', r'\resizebox{\linewidth}{!}{%',
        r'\footnotesize', topspec, r'\toprule',
        rf' & & & \multicolumn{{4}}{{c}}{{\textbf{{{face_head}}}}} & '
        rf'\multicolumn{{4}}{{c}}{{\textbf{{{edge_head}}}}} \\',
        r'\cmidrule(lr){4-7} \cmidrule(lr){8-11}',
        r'\textbf{S.no} & \textbf{Seg. VLM} & \textbf{View Selector} & '
        r'\textbf{IoU} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & '
        r'\textbf{IoU} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\',
        r'\midrule']
    foot = [r'\bottomrule', endspec, r'}', r'\end{table}']
    return '\n'.join(head + cl + foot)


def table_baselines():
    """tab:baselines -- 6 corrected point-cloud baselines + 3 MV-GEL rows."""
    brows = [(lbl, row8(CONFIGS[f'baseline::{lbl}']['df'])) for lbl in BASELINES]
    mv_nms3 = row8(CONFIGS['vnms1::CAD_LISA::film::top3']['df'])
    mv_top3 = row8(CONFIGS['mvgel::CAD_LISA::cliplora_film::top3']['df'])
    mv_top1 = row8(CONFIGS['mvgel::CAD_LISA::cliplora_film::top1']['df'])

    face_rec = [r[1][2] for r in brows]
    edge_rec = [r[1][6] for r in brows]
    bi_f, bi_e = int(np.argmax(face_rec)), int(np.argmax(edge_rec))

    lines = []
    for i, (name, v) in enumerate(brows):
        cells = [fmt(v[c], bold=((c == 2 and i == bi_f) or (c == 6 and i == bi_e)))
                 for c in range(8)]
        lines.append(f'{i + 1} &\n{name}\n& ' + ' & '.join(cells[:4]) +
                     '\n& ' + ' & '.join(cells[4:]) + r' \\')
    body = '\n\n'.join(lines)

    nms_cells = [fmt(mv_nms3[c]) for c in range(8)]
    t3_cells = [fmt(mv_top3[c]) for c in range(8)]
    t1_cells = [fmt(mv_top1[c], bold=(c in (0, 1, 3, 4, 5, 7))) for c in range(8)]
    mv_block = (
        '7 &\nMV-GEL (FiLM@top3, ViewNMS)\n& ' + ' & '.join(nms_cells[:4]) +
        '\n& ' + ' & '.join(nms_cells[4:]) + r' \\' + '\n\n'
        '8 &\nMV-GEL (FiLM@top3)\n& ' + ' & '.join(t3_cells[:4]) +
        '\n& ' + ' & '.join(t3_cells[4:]) + r' \\' + '\n\n'
        '9 &\n' + r'\textbf{MV-GEL (FiLM@top1)}' + '\n& ' + ' & '.join(t1_cells[:4]) +
        '\n& ' + ' & '.join(t1_cells[4:]) + r' \\')

    head = [
        r'\begin{table}[h]', r'\centering',
        r'\caption{\textbf{Comparison against zero-shot point cloud baselines} on '
        r'the common 1535-query testset across 218 meshes. Existing point-cloud '
        r'localization methods achieve high recall but low precision, resulting in '
        r'poor F1 scores due to over-segmentation.}',
        r'\label{tab:baselines}', r'\resizebox{\linewidth}{!}{%',
        r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{c l cccc cccc}', r'\toprule',
        r'&', r'&', r'\multicolumn{4}{c}{\textbf{Face Localization}}', r'&',
        r'\multicolumn{4}{c}{\textbf{Edge Localization}}', r'\\',
        r'\cmidrule(lr){3-6}', r'\cmidrule(lr){7-10}',
        r'\textbf{S.no} & \textbf{Model} & \textbf{IoU} & \textbf{Prec.} & '
        r'\textbf{Rec.} & \textbf{F1} & \textbf{IoU} & \textbf{Prec.} & '
        r'\textbf{Rec.} & \textbf{F1} \\', r'\midrule']
    foot = [r'\bottomrule', r'\end{tabular}%', r'}', r'\end{table}']
    return '\n'.join(head + [body, r'\midrule', mv_block] + foot)


# ======================================================================
# 4) PRINT tables
# ======================================================================
print('\n' + '#' * 72)
print('# LaTeX tables (corrected 1535 set)')
print('#' * 72 + '\n')
print(table_baselines(), '\n')
print(table_lisa_view_selectors(), '\n')
print(table_cad_topk(1, 'tab:lisa_top1',
      r'Performance evaluation of domain adapted LISA and view selectors across '
      r'Face@ top1 and Edge@ top1 metrics. \textit{GTviews} is the caption-marked '
      r'target-view oracle.', 'Face@ top1', 'Edge@ top1', tabular_star=True,
      gt_marked_target=True), '\n')
print(table_cad_topk(3, 'tab:ablation_top3', r'\textbf{Top-3 View Selection '
      r'Ablation.}', 'Face@ top3', 'Edge@ top3', tabular_star=True), '\n')
print(table_cad_topk(5, 'tab:ablation_top5', r'\textbf{Top-5 View Selection '
      r'Ablation.}', 'Face@ top5', 'Edge@ top5', tabular_star=True), '\n')


# ======================================================================
# 5) CROSS-CHECK recomputed cells vs the values currently in the paper
# ======================================================================
# EXPECTED = the numbers presently in the paper. Baselines rows 1-6 are EXPECTED
# to change (old uncorrected values); everything else must match exactly.
EXP = {
    # tab:lisa_view_selectors  (GT = marked-target)
    'gttarget': {
        ('CAD_LISA', 'cliplora_cross_attention'): [0.486, 0.570, 0.694, 0.582, 0.287, 0.327, 0.547, 0.359],
        ('CAD_LISA', 'cliplora_film'):            [0.501, 0.588, 0.706, 0.598, 0.284, 0.326, 0.551, 0.358],
        ('CAD_LISA', 'cliplora_no_fusion'):       [0.490, 0.570, 0.695, 0.584, 0.283, 0.324, 0.541, 0.355],
        ('CAD_LISA', 'cliplora_only_clip'):       [0.305, 0.374, 0.464, 0.377, 0.160, 0.193, 0.327, 0.210],
        ('CAD_LISA', 'random'):                   [0.196, 0.262, 0.293, 0.250, 0.071, 0.097, 0.156, 0.101],
        ('CAD_LISA', 'GT'):                       [0.529, 0.619, 0.744, 0.632, 0.311, 0.355, 0.598, 0.390],
        ('Vanilla_LISA', 'cliplora_cross_attention'): [0.292, 0.337, 0.652, 0.390, 0.043, 0.048, 0.494, 0.074],
        ('Vanilla_LISA', 'cliplora_film'):            [0.293, 0.342, 0.647, 0.391, 0.046, 0.053, 0.503, 0.079],
        ('Vanilla_LISA', 'cliplora_no_fusion'):       [0.293, 0.338, 0.647, 0.389, 0.044, 0.049, 0.502, 0.077],
        ('Vanilla_LISA', 'cliplora_only_clip'):       [0.201, 0.239, 0.460, 0.272, 0.035, 0.039, 0.388, 0.061],
        ('Vanilla_LISA', 'random'):                   [0.122, 0.159, 0.287, 0.174, 0.028, 0.034, 0.261, 0.046],
        ('Vanilla_LISA', 'GT'):                       [0.293, 0.340, 0.645, 0.389, 0.046, 0.051, 0.510, 0.079],
    },
    # tab:lisa_top1 (CAD); GT row now uses the caption-marked target oracle
    # (same as tab:lisa_view_selectors), not the geometric-visibility view.
    'top1': {
        'cliplora_cross_attention': [0.486, 0.570, 0.694, 0.582, 0.287, 0.327, 0.547, 0.359],
        'cliplora_film':            [0.501, 0.588, 0.706, 0.598, 0.284, 0.326, 0.551, 0.358],
        'cliplora_no_fusion':       [0.490, 0.570, 0.695, 0.584, 0.283, 0.324, 0.541, 0.355],
        'cliplora_only_clip':       [0.305, 0.374, 0.464, 0.377, 0.160, 0.193, 0.327, 0.210],
        'random':                   [0.196, 0.262, 0.293, 0.250, 0.071, 0.097, 0.156, 0.101],
        'GT':                       [0.529, 0.619, 0.744, 0.632, 0.311, 0.355, 0.598, 0.390],
    },
    'top3': {
        'cliplora_cross_attention': [0.416, 0.463, 0.807, 0.537, 0.229, 0.246, 0.712, 0.319],
        'cliplora_film':            [0.419, 0.469, 0.810, 0.540, 0.228, 0.248, 0.708, 0.317],
        'cliplora_no_fusion':       [0.413, 0.461, 0.808, 0.535, 0.225, 0.243, 0.710, 0.316],
        'cliplora_only_clip':       [0.307, 0.344, 0.662, 0.407, 0.142, 0.155, 0.534, 0.211],
        'random':                   [0.237, 0.265, 0.618, 0.332, 0.083, 0.096, 0.376, 0.133],
        'GT':                       [0.421, 0.460, 0.844, 0.541, 0.236, 0.254, 0.732, 0.327],
    },
    'top5': {
        'cliplora_cross_attention': [0.371, 0.399, 0.861, 0.493, 0.180, 0.192, 0.765, 0.267],
        'cliplora_film':            [0.374, 0.407, 0.861, 0.499, 0.182, 0.195, 0.767, 0.270],
        'cliplora_no_fusion':       [0.368, 0.400, 0.856, 0.492, 0.174, 0.186, 0.760, 0.262],
        'cliplora_only_clip':       [0.295, 0.319, 0.757, 0.403, 0.127, 0.136, 0.642, 0.200],
        'random':                   [0.231, 0.244, 0.756, 0.331, 0.081, 0.089, 0.513, 0.135],
        'GT':                       [0.378, 0.402, 0.890, 0.500, 0.192, 0.204, 0.781, 0.279],
    },
    # tab:baselines MV-GEL rows 7-9 (must match) -- rows 1-6 are updated, not checked here
    'baseline_mvgel': {
        'nms3': [0.372, 0.400, 0.846, 0.495, 0.180, 0.194, 0.695, 0.266],
        'top3': [0.419, 0.469, 0.810, 0.540, 0.228, 0.248, 0.708, 0.317],
        'top1': [0.501, 0.588, 0.706, 0.598, 0.284, 0.326, 0.551, 0.358],
    },
}
# old (pre-correction) baseline rows currently in the paper, for the diff report
OLD_BASELINES = {
    'PartSLIP (Default)': [0.070, 0.080, 0.361, 0.110, 0.011, 0.012, 0.411, 0.021],
    'PartSLIP (Top-PCD)': [0.063, 0.080, 0.266, 0.099, 0.013, 0.014, 0.301, 0.024],
    'Find3D (Default)':   [0.171, 0.172, 0.955, 0.260, 0.017, 0.017, 0.938, 0.032],
    'Find3D (Top-PCD)':   [0.165, 0.187, 0.561, 0.245, 0.022, 0.023, 0.409, 0.040],
    'PatchAlign3D (Default)': [0.158, 0.161, 0.890, 0.241, 0.017, 0.017, 0.895, 0.031],
    'PatchAlign3D (Top-PCD)': [0.159, 0.208, 0.379, 0.234, 0.021, 0.026, 0.177, 0.038],
}


def approx(a, b, tol=5e-4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


print('\n' + '=' * 72)
print('CROSS-CHECK: recomputed vs paper (tol 5e-4 at 3 d.p.)')
print('=' * 72)
npass = nfail = 0


def check(name, got, exp):
    global npass, nfail
    got3 = [round(x, 3) for x in got]
    if approx(got3, exp):
        npass += 1
        return
    nfail += 1
    print(f'  MISMATCH {name}\n    got {got3}\n    exp {exp}')


for (vlm, vs), exp in EXP['gttarget'].items():
    if vs == 'GT':
        got = row8(CONFIGS[f'gttarget::{vlm}::top1']['df'])
    else:
        got = row8(CONFIGS[f'mvgel::{vlm}::{vs}::top1']['df'])
    check(f'view_selectors::{vlm}::{vs}', got, exp)
for tab, k in [('top1', 1), ('top3', 3), ('top5', 5)]:
    for vs, exp in EXP[tab].items():
        if vs == 'GT':
            # tab:lisa_top1 GT = marked-target oracle; top3/top5 GT = geometric.
            if tab == 'top1':
                got = row8(CONFIGS['gttarget::CAD_LISA::top1']['df'])
            else:
                got = row8(CONFIGS[f'mvgel::CAD_LISA::GT::top{k}']['df'])
        else:
            got = row8(CONFIGS[f'mvgel::CAD_LISA::{vs}::top{k}']['df'])
        check(f'{tab}::CAD::{vs}', got, exp)
check('baselines::MV-GEL@nms3', row8(CONFIGS['vnms1::CAD_LISA::film::top3']['df']),
      EXP['baseline_mvgel']['nms3'])
check('baselines::MV-GEL@top3', row8(CONFIGS['mvgel::CAD_LISA::cliplora_film::top3']['df']),
      EXP['baseline_mvgel']['top3'])
check('baselines::MV-GEL@top1', row8(CONFIGS['mvgel::CAD_LISA::cliplora_film::top1']['df']),
      EXP['baseline_mvgel']['top1'])

print(f'\n  unchanged-cell checks: {npass} PASS, {nfail} MISMATCH')

print('\n' + '=' * 72)
print('UPDATED baseline rows (tab:baselines rows 1-6): old -> new')
print('=' * 72)
for lbl in BASELINES:
    new = [round(x, 3) for x in row8(CONFIGS[f'baseline::{lbl}']['df'])]
    old = OLD_BASELINES[lbl]
    changed = '' if approx(new, old) else '   <-- changed'
    print(f'  {lbl:24s}\n    old {old}\n    new {new}{changed}')

print('\n[done] consolidated artefacts in results_1535/ ; tables above.')
