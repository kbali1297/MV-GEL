# MV-GEL: Multi-View Geometric Entity Localization on CAD Meshes

MV-GEL localizes a natural-language–referred geometric entity (a **face** or an
**edge**) on a 3D CAD mesh by (1) selecting the most informative rendered views
with a lightweight **CLIP-LoRA view selector**, (2) segmenting the entity in those
views with a domain-adapted segmentation VLM (**LISA-CAD**), and (3) back-projecting
the 2D masks onto the mesh to produce a 3D localization, scored against the
ground-truth entity geometry (IoU / Precision / Recall / F1).

This repository contains the **code** (training, inference, utilities, baselines,
and analysis) and the **per-entity result tables** for the corrected 1535-query
test set (655 faces + 880 edges across 218 meshes). Large artefacts — model
weights, rendered views, and CAD meshes — are intentionally **not** committed
(see [Data & checkpoints](#data--checkpoints)).

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── val_dataset_1535.log            # test split: 218 mesh folders
│   ├── val_dataset_1535_entities.txt   # 1535 allowlist keys "cad,feature,idx"
│   └── means_baselines_1535.csv        # baseline means (reference)
│
│  ── Training ───────────────────────────────────────────────
├── train_ds_GLISA.py            # train LISA-CAD (DeepSpeed, LoRA, optional depth fusion)
├── train_ds.py                  # base LISA-CAD training entry (no depth branch)
├── train_views_selector.py      # train the CLIP-LoRA view selector (fusion ablations)
│
│  ── View precompute / inference ────────────────────────────
├── precompute_top_views.py      # cache the view selector's top-k views per entity
├── precompute_top_views_batch.py# batched variant of the above
├── infer_views_selector.py      # standalone view-selector inference / scoring
├── inference.py                 # single-sample end-to-end localization demo
│
│  ── Evaluation (MV-GEL) ────────────────────────────────────
├── GeoLocLM_exp.py              # core per-entity evaluation (view select → LISA → back-project)
├── run_GeoLocLM_all.py          # multi-GPU supervisor: all (Seg.VLM × selector × shard) jobs
│
│  ── Zero-shot point-cloud baselines ───────────────────────
├── baselines/                   # vendored upstream repos (code only) + wrappers
│   ├── README.md                # per-baseline env + weight-download setup
│   ├── partslip/                # Colin97/PartSLIP  + partslip_geolocalization.py
│   ├── find3d/                  # ziqi-ma/Find3D    + find3d_geolocalization.py
│   └── patchalign3d/            # PatchAlign3D       + patchalign3d_geolocalization.py
├── run_baselines_1535.py        # sharded baseline sweep supervisor
├── aggregate_baselines_1535.py  # combine baseline shards/recovery → means
│
│  ── Dataset generation ─────────────────────────────────────
├── render_views.py              # render multi-view RGB images of CAD meshes
├── render_depth_maps.py         # render aligned depth maps (for the depth branch)
├── augment_question.py          # paraphrase / augment referring captions
├── generate_dataset_parallel.py # build the VQA-style caption dataset (LLM-assisted)
├── split_dataset.py             # train/val split utilities
│
│  ── Analysis / tables ──────────────────────────────────────
├── aggregate_all_tables.py      # ONE entry point: rebuild every paper table from logs
├── check_marked_view_rank.py    # verify the marked target view ∈ geometric top-k
│
│  ── Shared modules ─────────────────────────────────────────
├── cad_utils.py                 # CAD I/O, entity→mesh-face extraction, back-projection
├── eval_utils.py                # IoU/P/R/F1 metrics, mask utilities
├── _gt_features_helper.py       # subprocess GT-feature worker (FreeCAD env)
├── model/                       # model code (see below)
├── utils/                       # datasets, conversation, metric helpers
│
└── results/
    └── results_1535/
        ├── all_per_entity_metrics.csv   # tidy long table, all configs (single re-processable file)
        ├── means_all_tables_1535.csv    # per-config means
        └── SOURCES.md                   # checkpoint + log-file provenance per config
```

### `model/`
- `LISA.py` — LISA segmentation VLM (LLaVA + SAM mask decoder) used for entity segmentation.
- `GLISA.py` — LISA-CAD variant with an optional **depth encoder** fusion branch (`DepthEncoder`).
- `part_views.py` — the **CLIP-LoRA view selector** (`LoraCLIPViewSelector_Ablation`) with the four fusion variants.
- `Point_MAE.py` — Point-MAE point-cloud encoder (used by point-cloud baselines / ablations).
- `llava/`, `segment_anything/` — vendored LLaVA and SAM model code (weights excluded).

### `utils/`
- `dataset_.py` — the central datasets:
  - `CAD_VQA_dataset` / `cad_collate_fn` — caption-conditioned segmentation training data for LISA-CAD.
  - `CAD_ViewRank_Dataset` / `views_collate_fn` — view-ranking data for the view selector. Reads
    `views_and_ques_{edge,face}_augmented_corrected.log` per mesh; supports `caption_variant`
    and an `entity_allowlist` for restricting evaluation to a fixed key set.
  - `extract_el_az_from_view_desc` — parse `view_e{EL}_a{AZ}` filenames into elevation/azimuth.
- `utils.py` — token constants, `AverageMeter`, `intersectionAndUnionGPU`, distributed helpers.
- `conversation.py`, `cad_vqa_dataset.py`, `data_processing.py`, plus standard LISA seg-dataset loaders.

---

## Installation

The pipeline uses **four** conda environments because the baselines pull in
mutually incompatible `transformers` / CUDA stacks. Only the first is needed for
MV-GEL itself.

| Env | Purpose | Python |
|-----|---------|--------|
| MV-GEL (main) | LISA-CAD + view selector training & eval | 3.10 |
| `LISA_multi_view` | GT entity-geometry extraction (FreeCAD) — called as a subprocess | 3.10 |
| `FIND3D`, `PARTSLIP`, `patchalign3d` | the three point-cloud baselines | 3.10 / 3.9 |

```bash
conda create -n mvgel python=3.10 -y && conda activate mvgel
pip install -r requirements.txt
# FreeCAD env for ground-truth entity geometry (used by GeoLocLM_exp.py / baselines)
conda create -n LISA_multi_view python=3.10 -y && conda install -n LISA_multi_view -c conda-forge freecad -y
```

Secrets are read from the environment (none are committed):

```bash
export OPENROUTER_API_KEY=...   # only for LLM-assisted caption generation
export HF_TOKEN=...             # only if pulling gated HF weights
```

See `configs/tokens.config.example`.

---

## Data & checkpoints

Not committed (regenerate or download separately):

- **CAD meshes + rendered views** — `ABC_CAD_Dataset_small{2,3}/<cad>/` with
  `mesh_views_corrected/` (RGB), `target_CDviews/` (GT masks), and per-mesh
  `views_and_ques_{edge,face}_augmented_corrected.log` caption files.
- **View-selector checkpoints** (~0.6 GB each):
  `best_model_view_ranker_cliplora_{cross_attention,film,no_fusion,only_clip}.pt`.
- **LISA-CAD checkpoint**: `runs/CAD_LISA_repro20/ckpt_model/global_step5076`.
- **Base weights**: LISA-7B, SAM ViT-H, CLIP — under `model/load_files*`.

`configs/val_dataset_1535*.{log,txt}` (the test split + entity allowlist) **are**
included so the metrics are reproducible once the data/weights are in place.

> Scripts use absolute paths from the original workspace
> (`/data/1bali/.../LISA`). Update the `ROOT` / default-path constants near the
> top of each entry script to your checkout location before running.

---

## Training

### 1. LISA-CAD (segmentation VLM)
Domain-adapts LISA to CAD renders with LoRA, under DeepSpeed. `train_ds_GLISA.py`
adds an optional depth-encoder fusion branch; `train_ds.py` is the RGB-only entry.

```bash
deepspeed --num_gpus=4 train_ds_GLISA.py \
    --version model/load_files\&weights/LISA-7B-v1-explanatory \
    --dataset_dir ./dataset \
    --log_base_dir ./runs \
    --lora_r 8 --image_size 1024 --model_max_length 512
```
Produces a DeepSpeed checkpoint under `runs/<exp>/ckpt_model/` (merge LoRA at eval
time — `GeoLocLM_exp.py` handles `global_step*` checkpoints and LoRA merging).

### 2. View selector (CLIP-LoRA)
Ranks the rendered views for a referring caption. Train one model per fusion
variant; the variant string also names the output checkpoint
(`best_model_view_ranker_cliplora_<fusion>.pt`).

```bash
python train_views_selector.py --modality_fusion film            --batch_size 4
python train_views_selector.py --modality_fusion cross_attention --batch_size 4
python train_views_selector.py --modality_fusion no_fusion       --batch_size 4
python train_views_selector.py --modality_fusion only_clip       --batch_size 4
```
Fusion variants map to the paper's view-selector rows
(FiLM / Cross-Attention / No-Fusion / Only-CLIP). `random` and `GTviews` are
sentinels evaluated directly (no checkpoint).

---

## Inference & evaluation

### Core per-entity evaluation
`GeoLocLM_exp.py` runs the full MV-GEL pipeline for each entity: select top-k views
→ LISA segment → back-project → score vs GT geometry. Ground-truth entity geometry
is computed in the `LISA_multi_view` (FreeCAD) env via a hard-timeout subprocess.

Key flags: `--view_selector_model_path`, `--LISA_model_path` (a non-CAD path ⇒
vanilla LISA), `--num_top_views {1,3,5}`, `--view_nms {0,1}`,
`--val_dataset_log`, `--entity_allowlist`, `--gt_timeout`,
`--gt_target_view 1` (use the caption-marked target view as the GT oracle),
`--gt_view_offset N` (rank-offset oracle ablation). Output lands in
`MVGEL_<VLM>_<selector>_vnms<N>_<tag>/localization_metrics_*.log`.

### Multi-GPU sweep
`run_GeoLocLM_all.py` fans every `(Seg.VLM × view-selector × shard)` job across
GPUs, with auto-resume from each run's `progress.txt`:

```bash
python run_GeoLocLM_all.py \
    --num-top-views 1 --num-gpus 4 \
    --val_dataset_log configs/val_dataset_1535.log \
    --entity_allowlist configs/val_dataset_1535_entities.txt \
    --num-shards 6 --gt-timeout 120
# marked-target GT oracle:
python run_GeoLocLM_all.py --selectors GT --gt-target-view 1 --num-shards 6 ...
```

### Precompute top views (optional speedup)
```bash
python precompute_top_views_batch.py   # cache selector top-k per entity
```

---

## Baselines (zero-shot point-cloud)

Wrappers around PartSLIP, Find3D, and PatchAlign3D, each in **Default** and
**Top-PCD** presets. Each baseline is a **vendored copy of its (MIT-licensed)
upstream repo** under `baselines/<name>/` plus our wrapper. They are **not**
runnable from a bare clone — each needs its own conda env and downloaded weights.
See **[`baselines/README.md`](baselines/README.md)** for the full per-baseline
environment setup and weight-download instructions.

Once the envs + weights are in place, orchestrate + aggregate from the repo root:

```bash
export PARTSLIP_PY=$(conda run -n PARTSLIP which python)
export FIND3D_PY=$(conda run -n find3d which python)
export PATCHALIGN_PY=$(conda run -n patchalign3d which python)

python run_baselines_1535.py --configs all --shards 6 --gpus 0,1,2,3
python aggregate_baselines_1535.py        # → configs/means_baselines_1535.csv + LaTeX rows
```
Each wrapper shells GT extraction to the `LISA_multi_view` (FreeCAD) env
(`--gt_timeout` controls the per-entity timeout; heavy meshes were recovered with
`--gt_timeout 600` into `*_recover` dirs, which the aggregator merges).

---

## Reproducing the paper tables

`aggregate_all_tables.py` is the single entry point. It documents (in its
`MANIFEST`) every checkpoint and log-file set, **combines sharded runs with their
completion/recovery re-runs** under an explicit precedence, restricts to the 1535
allowlist, asserts full 655/880 coverage, exports the consolidated artefacts under
`results/results_1535/`, and prints all five LaTeX tables — with a built-in
cross-check against the values in the paper.

```bash
python aggregate_all_tables.py
```

Tables produced:
- `tab:baselines` — 6 point-cloud baselines + 3 MV-GEL rows (FiLM @top1/@top3/@top3+ViewNMS).
- `tab:lisa_view_selectors` — LISA-CAD vs vanilla LISA × 6 selectors, Top-1, GT = caption-marked target view.
- `tab:lisa_top1` — LISA-CAD view-selector ablation @top1 (GT = marked target view).
- `tab:ablation_top3`, `tab:ablation_top5` — Top-3 / Top-5 ablations (GT = geometric-visibility oracle).

`check_marked_view_rank.py` verifies that the caption-marked target view lies
within the geometric top-3 (hence top-5) for **100%** of the 1535 entities,
justifying the geometric oracle in the Top-3/Top-5 tables.

The consolidated `results/results_1535/all_per_entity_metrics.csv` is a single
tidy file you can re-process without touching the scattered run directories;
`results/results_1535/SOURCES.md` lists the checkpoint + log provenance per config.

---

## Notes
- All hardcoded API keys/tokens were removed; secrets come from environment
  variables only.
- Background `nohup` launches must use explicit interpreter paths (not a `$VAR`
  set earlier on the same compound line), which otherwise expands empty.
