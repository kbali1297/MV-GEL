# Point-cloud baselines (PartSLIP / Find3D / PatchAlign3D)

Each baseline is a **vendored, MIT-licensed copy of its upstream repo** (code only —
model weights and our run outputs are removed) plus one MV-GEL **wrapper script**
that runs the baseline on the 1535-query CAD test set and writes
`localization_metrics_*.log` files in the same format as MV-GEL.

```
baselines/
├── partslip/      ← github.com/Colin97/PartSLIP  (+ partslip_geolocalization.py)
├── find3d/        ← github.com/ziqi-ma/Find3D     (+ find3d_geolocalization.py)
└── patchalign3d/  ← PatchAlign3D                  (+ patchalign3d_geolocalization.py)
```

> **These do NOT run out of the box from a fresh clone.** Each baseline needs its
> own conda environment (mutually incompatible deps) **and** its pretrained
> weights downloaded. Follow the per-baseline steps below. The wrapper expects to
> sit at `baselines/<name>/` so it can import the MV-GEL `utils/` and
> `eval_utils.py` from two levels up; do not move it.

Common to all three: ground-truth entity geometry is extracted with **FreeCAD**,
which the wrappers shell out to via `conda run -n LISA_multi_view`. Create that
env once:

```bash
conda create -n LISA_multi_view python=3.10 -y
conda install -n LISA_multi_view -c conda-forge freecad -y
```

Run + aggregate everything with the supervisors at the repo root. Point the
per-env interpreters at your envs (or `conda activate` each before its config):

```bash
export PARTSLIP_PY=$(conda run -n PARTSLIP   which python)
export FIND3D_PY=$(conda run   -n find3d     which python)
export PATCHALIGN_PY=$(conda run -n patchalign3d which python)

python run_baselines.py --configs all --shards 6 --gpus 0,1,2,3
python aggregate_baselines.py          # → configs/means_baselines_1535.csv + LaTeX rows
```

---

## 1. PartSLIP — `baselines/partslip/`
Upstream: https://github.com/Colin97/PartSLIP (MIT). Uses GLIP for open-vocab 2D
detection over rendered point clouds.

```bash
cd baselines/partslip
conda env create -f environment.yml          # creates env "PARTSLIP"
conda activate PARTSLIP
# PyTorch3D (point-cloud rendering)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# GLIP (modified fork) — build it (the source is vendored under ./GLIP)
cd GLIP && python setup.py build develop --user && cd ..
# cut-pursuit superpoints
pip install git+https://github.com/loicland/superpoint_graph.git   # see upstream README
```

Weights: download `glip_large_model.pth` from
https://huggingface.co/datasets/minghua/PartSLIP/tree/main/models into
`baselines/partslip/models/`.

Run directly (or via the supervisor):
```bash
python partslip_geolocalization.py \
    --val_dataset_log ../../configs/val_dataset.log \
    --entity_allowlist ../../configs/val_dataset_1535_entities.txt \
    --preset conf --experiment_name GeLoM_PartSLIP_default_1535_shard00
# Top-PCD variant: --preset topk_pct
```

## 2. Find3D — `baselines/find3d/`
Upstream: https://github.com/ziqi-ma/Find3D (MIT). Point-transformer trained on
internet 3D assets; weights pulled from HuggingFace at runtime.

```bash
cd baselines/find3d/model
conda create -n find3d python=3.8 -y && conda activate find3d
pip install -r requirements.txt
# Pointcept point ops
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept/libs/pointops && python setup.py install && cd -
# FlashAttention (slow build, up to ~3h)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && MAX_JOBS=4 python setup.py install && cd -
```

Weights: pulled automatically via `Find3D.from_pretrained("ziqima/find3d-checkpt0")`
(set `HF_TOKEN` if rate-limited).

```bash
python find3d_geolocalization.py \
    --val_dataset_log ../../configs/val_dataset.log \
    --entity_allowlist ../../configs/val_dataset_1535_entities.txt \
    --n_points 5000 --preset default --experiment_name GeLoM_Find3D_default_1535_shard00
# Top-PCD variant: --preset topk
```

## 3. PatchAlign3D — `baselines/patchalign3d/`
Upstream: PatchAlign3D (MIT). Patch-aligned point transformer with open-vocab CLIP.

```bash
cd baselines/patchalign3d
conda create -n patchalign3d python=3.9 -y && conda activate patchalign3d
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

Weights: download the stage-2 encoder checkpoint from
https://huggingface.co/patchalign3d/patchalign3d-encoder into
`baselines/patchalign3d/ckpts/patchalign3d.pt`.

```bash
python patchalign3d_geolocalization.py \
    --val_dataset_log ../../configs/val_dataset.log \
    --entity_allowlist ../../configs/val_dataset_1535_entities.txt \
    --preset default --experiment_name GeLoM_PatchAlign3D_default_1535_shard00
# Top-PCD variant: --preset topk
```

---

### Notes
- Each vendored repo keeps its upstream `LICENSE` and `README.md` for full
  attribution and the canonical setup instructions.
- Run outputs land in `GeLoM_*` dirs inside each `baselines/<name>/`;
  `aggregate_baselines.py` globs those (merging `_recover` re-runs of heavy
  meshes) into the final baseline rows of `tab:baselines`.
- The presets map to the paper's **Default** (paper-faithful) and **Top-PCD**
  (top-confidence point-cloud subset) columns.
