#!/usr/bin/env python
"""Check whether the caption-MARKED target view falls within the geometric
top_views_desc[:k] ranking for each of the 1535 test entities.

If the marked view is within top-k, then the GT@top-k oracle (which uses
top_views_desc[:k]) genuinely includes the marked target view as one of its
legitimate views, justifying it as the oracle for the Top-3 / Top-5 tables.
"""
import os
import re
import ast
import glob

ECCV = '/data/1bali/Other_LLM_projects/ECCV_2026'
ROOT = f'{ECCV}/LISA'
ALLOWLIST = f'{ROOT}/val_dataset_1535_entities.txt'
DATASETS = [f'{ECCV}/ABC_CAD_Dataset_small2', f'{ECCV}/ABC_CAD_Dataset_small3']

ELAZ = re.compile(r'view_e(-?\d+)_a(-?\d+)')


def elaz(path):
    m = ELAZ.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def load_allow():
    keys = set()
    with open(ALLOWLIST) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            c, ft, idx = ln.split(',')
            keys.add((c.strip().zfill(8), ft.strip(), int(idx)))
    return keys


ALLOW = load_allow()

# rank of the marked view within top_views_desc, per entity
ranks = {}        # (cad,feat,idx) -> rank (0-based) or None if not present
for ds in DATASETS:
    for log in glob.glob(f'{ds}/*/views_and_ques_*_augmented_corrected.log'):
        cad = os.path.basename(os.path.dirname(log)).zfill(8)
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = ast.literal_eval(line)
                mk = d['marked_image']
                bn = os.path.basename(mk)
                # ..._marked_{feature}[{idx}].png
                feat = bn.split('_marked_')[1].split('[')[0]
                idx = int(bn.split('[')[1].split(']')[0])
                key = (cad, feat, idx)
                if key not in ALLOW:
                    continue
                m_elaz = elaz(mk)
                tv_elaz = [elaz(v) for v in d['top_views_desc']]
                try:
                    r = tv_elaz.index(m_elaz)
                except ValueError:
                    r = None
                ranks[key] = r

found = [k for k in ALLOW if k in ranks]
missing = [k for k in ALLOW if k not in ranks]
not_in_tv = [k for k, r in ranks.items() if r is None]


def pct(n):
    return f'{n}/{len(ALLOW)} ({100.0 * n / len(ALLOW):.1f}%)'


def within(kmax, feat=None):
    n = 0
    for k, r in ranks.items():
        if r is None:
            continue
        if feat and k[1] != feat:
            continue
        if r < kmax:
            n += 1
    return n


print(f'allowlist entities         : {len(ALLOW)}')
print(f'entities matched in logs   : {pct(len(found))}')
if missing:
    print(f'  [warn] {len(missing)} allowlist entities not found in caption logs')
if not_in_tv:
    print(f'  [warn] {len(not_in_tv)} entities whose marked (el,az) is NOT in top_views_desc')

for feat in (None, 'face', 'edge'):
    tag = 'ALL ' if feat is None else feat
    tot = len(ALLOW) if feat is None else sum(1 for k in ALLOW if k[1] == feat)
    w1 = within(1, feat)
    w3 = within(3, feat)
    w5 = within(5, feat)

    def p(n):
        return f'{n}/{tot} ({100.0 * n / tot:.1f}%)'
    print(f'\n[{tag}] marked view rank in geometric top_views_desc')
    print(f'   within top-1 : {p(w1)}')
    print(f'   within top-3 : {p(w3)}')
    print(f'   within top-5 : {p(w5)}')

# rank histogram (0..9, then 10+)
hist = {}
for r in ranks.values():
    if r is None:
        hist['none'] = hist.get('none', 0) + 1
    else:
        b = r if r < 10 else '10+'
        hist[b] = hist.get(b, 0) + 1
print('\nrank histogram (0-based):')
for b in list(range(10)) + ['10+', 'none']:
    if b in hist:
        print(f'   rank {b:>4}: {hist[b]}')
