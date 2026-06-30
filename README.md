# MV-GEL: Multi-View Geometric Entity Localization on CAD Meshes

MV-GEL localizes a natural-language–referred geometric entity (a **face** or an
**edge**) on a 3D CAD mesh by (1) selecting the most informative rendered views
with a lightweight **CLIP-LoRA view selector**, (2) segmenting the entity in those
views with a domain-adapted segmentation VLM (**LISA-CAD**), and (3) back-projecting
the 2D masks onto the mesh to produce a 3D localization, scored against the
ground-truth entity geometry (IoU / Precision / Recall / F1).

This repository contains the **code** (training, inference, utilities, baselines,
and analysis) and the **per-entity result tables** for the corrected 1535-query
test set (655 faces + 880 edges across 218 meshes).

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── configs/
│   ├── train_dataset.log               # train split: 5685 CAD folders (paths only)
│   ├── val_dataset.log                 # val/test split: 218 CAD folders (paths only)
│   ├── val_dataset_1535_entities.txt   # 1535-query eval allowlist "cad,feature,idx"
│   ├── means_baselines_1535.csv        # baseline means (reference)
│   └── tokens.config.example           # env-var template for secrets
│
│  ── Training ───────────────────────────────────────────────
├── train_ds.py                  # train LISA-CAD (DeepSpeed, LoRA) — main MV-GEL trainer
├── train_views_selector.py      # train the CLIP-LoRA view selector (fusion ablations)
├── train.py                     # upstream-LISA generic reason-seg trainer (reference; needs upstream datasets)
│
│  ── Inference ──────────────────────────────────────────────
├── infer.py                     # core per-entity evaluation (view select → LISA → back-project)
├── run_MVGEL.py                 # multi-GPU supervisor: all (Seg.VLM × selector × shard) jobs
├── infer_views_selector.py      # standalone view-selector inference / scoring
├── inference.py                 # single-sample LISA segmentation demo
│
│  ── Zero-shot point-cloud baselines ───────────────────────
├── baselines/                   # vendored upstream repos (code only) + wrappers
│   ├── README.md                # per-baseline env + weight-download setup
│   ├── partslip/                # Colin97/PartSLIP
│   ├── find3d/                  # ziqi-ma/Find3D
│   └── patchalign3d/            # PatchAlign3D
├── run_baselines.py             # sharded baseline sweep supervisor
├── aggregate_baselines.py       # combine baseline shards/recovery → means
│
│  ── Dataset generation ─────────────────────────────────────
├── render_views.py              # render multi-view RGB images of CAD meshes
├── render_depth_maps.py         # render aligned depth maps
├── generate_dataset_parallel.py # build the VQA-style caption dataset (LLM-assisted)
├── split_dataset.py             # train/val split utility
│
│  ── Analysis / tables ──────────────────────────────────────
├── aggregate_all_tables.py      # ONE entry point: rebuild every paper table from logs
├── check_marked_view_rank.py    # verify the marked target view ∈ geometric top-k
│
│  ── Shared modules ─────────────────────────────────────────
├── cad_utils.py                 # CAD I/O, entity→mesh-face extraction, back-projection
├── eval_utils.py                # IoU/P/R/F1 metrics, mask utilities
├── inspect_masks.py             # mask visualization helper
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
- `LISA.py` — LISA segmentation VLM (LLaVA + SAM mask decoder, `LISAForCausalLM`) used for entity segmentation.
- `part_views.py` — the **CLIP-LoRA view selector** (`LoraCLIPViewSelector_Ablation`) with the four fusion variants.
- `llava/`, `segment_anything/` — vendored LLaVA and SAM model code (weights excluded).

### `utils/`
- `dataset_.py` — the central datasets:
  - `CAD_VQA_dataset` / `cad_collate_fn` — caption-conditioned segmentation training data for LISA-CAD.
  - `CAD_ViewRank_Dataset` / `views_collate_fn` — view-ranking data for the view selector. Reads
    `views_and_ques_{edge,face}_augmented_corrected.log` per mesh; supports `caption_variant`
    and an `entity_allowlist` for restricting evaluation to a fixed key set.
  - `extract_el_az_from_view_desc` — parse `view_e{EL}_a{AZ}` filenames into elevation/azimuth.
- `utils.py` — token constants, `AverageMeter`, `intersectionAndUnionGPU`, distributed helpers.
- `conversation.py`, `cad_vqa_dataset.py`, `data_processing.py`, plus the upstream LISA seg-dataset
  loaders (`dataset.py`, `reason_seg_dataset.py`, `refer_seg_dataset.py`, `sem_seg_dataset.py`,
  `vqa_dataset.py`, `refer.py`, `grefer.py`, `grefcoco.py`) used only by `train.py`.

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
# FreeCAD env for ground-truth entity geometry (used by infer.py / baselines)
conda create -n LISA_multi_view python=3.10 -y && conda install -n LISA_multi_view -c conda-forge freecad -y
```

Secrets are read from the environment (none are committed):

```bash
export OPENROUTER_API_KEY=...   # only for LLM-assisted caption generation
export HF_TOKEN=...             # only if you hit Hub rate limits while downloading
```

See `configs/tokens.config.example`.

---

## Data & checkpoints

The **code and the split configs are committed** to this repo. The split logs
[`configs/train_dataset.log`](configs/train_dataset.log) (5685 CAD folders) and
[`configs/val_dataset.log`](configs/val_dataset.log) (218 CAD folders) each list
**CAD folder paths only**; [`configs/val_dataset_1535_entities.txt`](configs/val_dataset_1535_entities.txt)
is the 1535-query eval allowlist over the val split. The large artefacts (meshes,
rendered views, weights) live on the Hugging Face Hub:

| Artefact | Hub repo | Size |
|----------|----------|------|
| **Dataset** — CAD meshes, rendered views, GT masks, caption logs | [`datasets/kbali1297/MV-GEL`](https://huggingface.co/datasets/kbali1297/MV-GEL) | ~15.6 GB |
| **Checkpoints** — 4 view-selectors, LISA-CAD DeepSpeed ckpt, SAM ViT-H | [`kbali1297/MV-GEL`](https://huggingface.co/kbali1297/MV-GEL) | ~36 GB |
| **Base LISA-7B** — init weights for training only | [`xinlai/LISA-7B-v1-explanatory`](https://huggingface.co/xinlai/LISA-7B-v1-explanatory) | ~16 GB |

### Where everything goes

Download into the repo so the default repo-relative paths resolve out of the box
(set `MVGEL_ROOT` if you keep the big files elsewhere — see below):

```
MVGEL_release/                                   # = ROOT = DATA_ROOT (default)
├── best_model_view_ranker_cliplora_film.pt            ┐
├── best_model_view_ranker_cliplora_cross_attention.pt │ view selectors
├── best_model_view_ranker_cliplora_no_fusion.pt       │ (from kbali1297/MV-GEL)
├── best_model_view_ranker_cliplora_only_clip.pt       ┘
├── runs/
│   └── CAD_LISA/                                # LISA-CAD DeepSpeed ckpt
│       ├── global_step5076/                     #   (from kbali1297/MV-GEL)
│       └── latest
├── model/
│   ├── segment_anything/load_files&weights/
│   │   └── sam_vit_h_4b8939.pth                 # SAM ViT-H (from kbali1297/MV-GEL)
│   └── load_files&weights/
│       └── LISA-7B-v1-explanatory/             # base LISA-7B (training only)
└── ABC_CAD_Dataset_central/                     # unpacked meshes + views + captions
    └── <cad_id>/ …                              #   (from datasets/kbali1297/MV-GEL;
                                                 #    referenced by configs/*_dataset.log)
```

The dataset can live anywhere — the committed `configs/{train,val}_dataset.log`
list absolute CAD-folder paths, so after unpacking just retarget them to your
location (see step 2).

### 1. Checkpoints (needed for inference and training)

```bash
pip install -U "huggingface_hub[cli]"
cd MVGEL_release

# View-selectors + LISA-CAD ckpt (runs/CAD_LISA/) -> repo root
huggingface-cli download kbali1297/MV-GEL --local-dir .

# Put SAM ViT-H where train_ds.py --vision_pretrained expects it
mkdir -p "model/segment_anything/load_files&weights"
mv sam_vit_h_4b8939.pth "model/segment_anything/load_files&weights/"
```

With this in place you can **run inference immediately** — `infer.py` /
`run_MVGEL.py` load `runs/CAD_LISA/global_step5076` and the view-selector `.pt`
files by their default names.

### 2. Dataset (needed for training, eval, and dataset (re)generation)

```bash
huggingface-cli download kbali1297/MV-GEL --repo-type dataset --local-dir hf_data
# Unpack the 24 mesh shards into ABC_CAD_Dataset_central/
DATASET_DIR="$PWD/ABC_CAD_Dataset_central"
mkdir -p "$DATASET_DIR"
for f in hf_data/cads_*.tar.gz; do tar -xzf "$f" -C "$DATASET_DIR"; done

# Retarget the committed split logs (+ the per-CAD caption logs) at your path
OLD=/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_central
sed -i "s#${OLD}#${DATASET_DIR}#g" configs/train_dataset.log configs/val_dataset.log
find "$DATASET_DIR" -name 'views_and_ques_*_augmented_corrected.log' \
    -exec sed -i "s#${OLD}#${DATASET_DIR}#g" {} +
```
The committed `configs/{train,val}_dataset.log` list one `ABC_CAD_Dataset_central/<cad_id>`
folder per line (CAD paths only). The dataset repo also ships
`train_dataset_central.log` / `val_dataset_central.log` (identical to the committed
splits) and `cad_index_mapping.csv`.

### 3. Base LISA-7B (training only)

Inference uses the released `runs/CAD_LISA` checkpoint directly. To **train**
LISA-CAD from scratch you also need the base LISA-7B init weights:

```bash
huggingface-cli download xinlai/LISA-7B-v1-explanatory \
    --local-dir "model/load_files&weights/LISA-7B-v1-explanatory"
```
(`train_ds.py --version` defaults to that path under `DATA_ROOT`.)

> **Paths are repo-relative.** Every entry script resolves code paths from its own
> location and data paths from a configurable root:
>
> ```python
> ROOT      = os.path.dirname(os.path.abspath(__file__))   # the code (this repo)
> DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)           # weights / datasets / runs
> ```
>
> By default `DATA_ROOT == ROOT`, i.e. weights, `runs/`, and `{train,val}_dataset.log`
> are expected **inside** the checkout. If your large artefacts live elsewhere, point
> `MVGEL_ROOT` at that directory:
>
> ```bash
> export MVGEL_ROOT=/path/to/data_root          # weights, runs/, *_dataset.log
> export MVGEL_CAD_DATASET=/path/to/ABC_CAD_Dataset_small2   # only for dataset generation
> export FREECAD_LIB=$(python -c "import sys;print(sys.prefix)")/lib  # auto-detected by default
> ```

---

## Training

### 1. LISA-CAD (segmentation VLM)
Domain-adapts LISA to CAD renders with LoRA, under DeepSpeed. `train_ds.py` is the
MV-GEL trainer used for the paper checkpoint (`runs/CAD_LISA/`).

```bash
export MVGEL_ROOT=/path/to/data_root          # holds runs/, *_dataset.log, model weights
deepspeed --include localhost:0,1,2,3 train_ds.py \
    --exp_name CAD_LISA \
    --batch_size 96 \
    --epochs 60
```
Key flags: `--version` / `--vision_pretrained` (base LISA-7B / SAM ViT-H weights,
default to `model/load_files&weights/...` under `DATA_ROOT`), `--no_eval` (skip
validation), `--resume <dir>` to continue, and `--reset_schedule` (with `--resume`)
to load **weights only** — skipping optimizer/LR state, which avoids a world-size
mismatch when resuming on a different GPU count. Produces a DeepSpeed checkpoint
under `runs/<exp_name>/global_step*/`.

**Reproduce the reported validation metrics** from the released checkpoint
(`runs/CAD_LISA/global_step5076`):

```bash
deepspeed --include localhost:0,1 train_ds.py \
    --eval_only --resume $MVGEL_ROOT/runs/CAD_LISA --reset_schedule --workers 2
# → giou ≈ 0.58, ciou ≈ 0.78  (paper: giou 0.5701, ciou 0.7838;
#   small giou variance comes from DistributedSampler sharding across GPU count
#   and per-pass random caption sampling — ciou is sampler-invariant.)
```

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
(FiLM / Cross-Attention / No-Fusion / Only-CLIP). `random` and `GT` are
sentinels evaluated directly (no checkpoint).

> `train.py` is the **upstream LISA** generic reason-segmentation trainer
> (sem_seg / refer_seg / vqa / reason_seg), kept for provenance. It is **not** part
> of the MV-GEL pipeline and requires the upstream LISA datasets, which are not
> shipped here.

---

## Inference & evaluation

### Core per-entity evaluation
`infer.py` runs the full MV-GEL pipeline for each entity: select top-k views
→ LISA segment → back-project → score vs GT geometry. Ground-truth entity geometry
is computed in the `LISA_multi_view` (FreeCAD) env via a hard-timeout subprocess.

```bash
export MVGEL_ROOT=/path/to/data_root
python infer.py \
    --view_selector_model_path best_model_view_ranker_cliplora_film.pt \
    --LISA_model_path runs/CAD_LISA/global_step5076 \
    --val_dataset_log configs/val_dataset.log \
    --entity_allowlist configs/val_dataset_1535_entities.txt \
    --num_top_views 1 --gt_timeout 120
```

Key flags: `--view_selector_model_path` (a bare `.pt` name, a path, or the sentinels
`random` / `GT`), `--LISA_model_path` (a non-CAD path / `vanilla` ⇒ vanilla LISA),
`--num_top_views {1,3,5}`, `--view_nms {0,1}`, `--val_dataset_log`,
`--entity_allowlist`, `--gt_timeout`, `--gt_target_view 1` (use the caption-marked
target view as the GT oracle), `--gt_view_offset N` (rank-offset oracle ablation).
Bare checkpoint names and relative dataset logs are auto-resolved against
`DATA_ROOT` then the repo root. Output lands in
`MVGEL_<VLM>_<selector>_vnms<N>_<tag>/localization_metrics_*.log`.

### Multi-GPU sweep
`run_MVGEL.py` fans every `(Seg.VLM × view-selector × shard)` job across
GPUs, with auto-resume from each run's `progress.txt`:

```bash
python run_MVGEL.py \
    --num-top-views 1 --num-gpus 4 \
    --val-dataset-log configs/val_dataset.log \
    --entity-allowlist configs/val_dataset_1535_entities.txt \
    --num-shards 6 --gt-timeout 120
# marked-target GT oracle:
python run_MVGEL.py --selectors GT --gt-target-view 1 --num-shards 6 ...
```

### Standalone view-selector inference
```bash
python infer_views_selector.py   # score / inspect the view selector's top-k ranking
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

python run_baselines.py --configs all --shards 6 --gpus 0,1,2,3
python aggregate_baselines.py        # → configs/means_baselines_1535.csv + LaTeX rows
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
