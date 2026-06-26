"""
Run PartSLIP zero-shot localization on the same CAD validation dataset used by
GeoLocLM.py and evaluate with the same `return_localization_metrics`.

Pipeline per CAD sample:
    .obj mesh  ──▶  surface-sample point cloud (with face indices kept)
                ──▶  PartSLIP zero-shot (GLIP + bbox2seg) using the question
                    as the text prompt
                ──▶  point-level sem_seg
                ──▶  project labelled points back to mesh face indices
                ──▶  for `face`: predicted faces = those face indices
                    for `edge`: predicted edges = unique edges of predicted faces
                ──▶  return_localization_metrics(...)
"""

import os
import re
import sys
import ast
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, TimeoutError

# Run from this directory so PartSLIP's relative paths (GLIP/configs, models/) resolve
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)

# Make LISA repo importable for dataset and eval utilities — MUST come before
# PartSLIP imports because PartSLIP's src/bbox2seg.py adds its own src/ dir to
# sys.path and does `from utils import ...`, which would shadow LISA's
# `utils/` package with PartSLIP's `src/utils.py` module.
# This script lives at <repo_root>/baselines/partslip/, so the MV-GEL root
# (with utils/ and eval_utils.py) is two levels up.
LISA_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, LISA_ROOT)

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from transformers import CLIPImageProcessor

# Import LISA dataset/eval utilities FIRST so `utils` binds to LISA's package
from utils.dataset_ import CAD_ViewRank_Dataset, views_collate_fn  # noqa: E402
from eval_utils import (  # noqa: E402
    stitch_mesh_topology,
    return_localization_metrics,
    cad_entity_to_mesh_faces,
    visualize_entity_predictions,
)

# Now PartSLIP imports — they will append their src/ to sys.path internally,
# but `utils` is already bound in sys.modules to LISA's package. Inject the
# PartSLIP utility symbols into the `utils` package so PartSLIP's
# `from utils import save_colored_pc, get_iou` keeps working.
sys.path.insert(0, THIS_DIR)
import importlib.util as _ilu
import utils as _lisa_utils  # LISA's package, already imported above
_partslip_utils_path = os.path.join(THIS_DIR, "src", "utils.py")
_spec = _ilu.spec_from_file_location("_partslip_utils", _partslip_utils_path)
_partslip_utils = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_partslip_utils)
for _name in ("save_colored_pc", "get_iou", "normalize_pc"):
    if hasattr(_partslip_utils, _name):
        setattr(_lisa_utils, _name, getattr(_partslip_utils, _name))

from src.render_pc import render_pc  # noqa: E402
from src.glip_inference import glip_inference, glip_inference_batched, load_model  # noqa: E402
from src.gen_superpoint import gen_superpoint  # noqa: E402
from src.bbox2seg import bbox2seg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_pointcloud_from_mesh(mesh, n_points=10000):
    """Surface-sample a mesh and return (xyz, rgb, face_idx_per_point)."""
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    xyz = np.asarray(points, dtype=np.float32)
    # Normalize: center + scale to unit ball (matches PartSLIP normalize_pc)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz = xyz / scale
    rgb = np.full_like(xyz, 0.5, dtype=np.float32)  # neutral gray
    return xyz, rgb, np.asarray(face_idx, dtype=np.int64)


def faces_to_edges(faces, face_indices):
    """Return unique sorted (v0, v1) edge tuples for a set of face indices."""
    edges = set()
    for fi in face_indices:
        f = faces[int(fi)]
        edges.add(tuple(sorted((int(f[0]), int(f[1])))))
        edges.add(tuple(sorted((int(f[1]), int(f[2])))))
        edges.add(tuple(sorted((int(f[2]), int(f[0])))))
    return list(edges)

## MV_GEL has a flipped vocab, left means right for it and vice versa, so we apply the same L/R swap to the question prompt before feeding it to GLIP, to put both on the same footing.
## To put both on the same footing, we apply an L/R swap to the question prompt before feeding it to GLIP.
_LR_SWAP = {
    "left": "right", "right": "left",
    "Left": "Right", "Right": "Left",
    "LEFT": "RIGHT", "RIGHT": "LEFT",
    "leftmost": "rightmost", "rightmost": "leftmost",
    "Leftmost": "Rightmost", "Rightmost": "Leftmost",
    "left-most": "right-most", "right-most": "left-most",
    "leftward": "rightward", "rightward": "leftward",
    "leftwards": "rightwards", "rightwards": "leftwards",
}
_LR_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_LR_SWAP, key=len, reverse=True)) + r")\b"
)


def swap_left_right(text):
    """Apply the same L/R caption fix used to put GeoLocLM on real-world footing."""
    if not text:
        return text
    return _LR_PATTERN.sub(lambda m: _LR_SWAP[m.group()], text)


def _gt_worker(q, cad_file_path, vertices, faces, feature, feature_idx):
    try:
        result = cad_entity_to_mesh_faces(
            cad_file_path, vertices, faces,
            entity_type=feature, entity_index=feature_idx,
        )
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", repr(e)))


# Shell out to the LISA env which has FreeCAD installed. PARTSLIP env does not.
GT_HELPER_ENV = "LISA_multi_view"
GT_HELPER_SCRIPT = os.path.abspath(os.path.join(LISA_ROOT, "_gt_features_helper.py"))


def _ensure_gt_helper_script():
    if os.path.exists(GT_HELPER_SCRIPT):
        return
    helper_src = '''import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import cad_entity_to_mesh_faces

def _to_py(o):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (list, tuple)):
        return [_to_py(x) for x in o]
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    return o

# Persistent worker: read one JSON request per line on stdin, emit one
# JSON response per line on stdout. The parent process spawns this ONCE
# (paying FreeCAD/Python startup a single time) and reuses it for all
# samples, instead of `conda run` per sample (which costs ~5 s each).
sys.stdout.write("@@READY@@\\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
        vertices = np.array(data["vertices"], dtype=np.float64)
        faces = np.array(data["faces"], dtype=np.int64)
        out = cad_entity_to_mesh_faces(
            data["cad_file_path"], vertices, faces,
            entity_type=data["feature"], entity_index=data["feature_idx"],
        )
        out = _to_py(out)
        sys.stdout.write(json.dumps({"ok": True, "result": out}) + "\\n")
    except Exception as e:
        sys.stdout.write(json.dumps({"ok": False, "err": repr(e)}) + "\\n")
    sys.stdout.flush()
'''
    with open(GT_HELPER_SCRIPT, "w") as f:
        f.write(helper_src)


# ---------------- Persistent GT helper subprocess ----------------
_GT_HELPER_PROC = None


def _start_gt_helper():
    """Spawn the long-running FreeCAD helper subprocess once and return it."""
    import subprocess
    _ensure_gt_helper_script()
    cmd = ["conda", "run", "--no-capture-output", "-n", GT_HELPER_ENV,
           "python", "-u", GT_HELPER_SCRIPT]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    # Wait for ready handshake.
    for _ in range(120):  # up to ~2 min for first-time FreeCAD import
        ln = proc.stdout.readline()
        if "@@READY@@" in ln:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"GT helper exited before READY (rc={proc.returncode}); "
                f"stderr={proc.stderr.read()[-2000:]}"
            )
    raise RuntimeError("GT helper did not signal READY in time")


def _get_gt_helper():
    global _GT_HELPER_PROC
    if _GT_HELPER_PROC is None or _GT_HELPER_PROC.poll() is not None:
        _GT_HELPER_PROC = _start_gt_helper()
    return _GT_HELPER_PROC


def compute_gt_features(cad_file_path, vertices, faces, feature, feature_idx, timeout_s=10):
    import json
    import threading
    proc = _get_gt_helper()
    payload = json.dumps({
        "cad_file_path": cad_file_path,
        "vertices": np.asarray(vertices).tolist(),
        "faces": np.asarray(faces).tolist(),
        "feature": feature,
        "feature_idx": int(feature_idx),
    })

    # Send request.
    try:
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        # Worker died; restart on next call.
        global _GT_HELPER_PROC
        _GT_HELPER_PROC = None
        print(f"[GT error] worker pipe broken for {cad_file_path}; will restart")
        return None

    # Read response with timeout via a worker thread.
    container = {}

    def _reader():
        try:
            container["line"] = proc.stdout.readline()
        except Exception as e:
            container["err"] = repr(e)

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        # Helper hung on this sample (e.g. pathological FreeCAD geom).
        # Kill it so future samples get a fresh process.
        print(f"[GT timeout] {cad_file_path}")
        try:
            proc.kill()
        except Exception:
            pass
        _GT_HELPER_PROC = None
        return None

    line = container.get("line", "")
    if not line:
        # Worker exited unexpectedly.
        _GT_HELPER_PROC = None
        print(f"[GT error] {cad_file_path}: empty response (worker exited)")
        return None
    try:
        result = json.loads(line.strip())
    except json.JSONDecodeError as e:
        print(f"[GT error] {cad_file_path}: bad json ({e}); line={line[:200]!r}")
        return None
    if not result.get("ok"):
        print(f"[GT error] {cad_file_path}: {result.get('err')}")
        return None
    return result["result"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PartSLIP localization on CAD val set")
    parser.add_argument("--n_points", type=int, default=10000)
    parser.add_argument("--num_views", type=int, default=10,
                        help="Must match render_pc default (10).")
    parser.add_argument("--gt_timeout", type=int, default=20,
                        help="Timeout (seconds) for FreeCAD GT-feature extraction.")    
    parser.add_argument("--glip_conf", type=float, default=0.2,
                        help="GLIP detection confidence threshold (lower = more recall). "
                             "This is OPTION 1 (confidence sweep).")
    parser.add_argument("--prompt_mode", type=str, default="question",
                        choices=["question", "feature", "feature_question"],
                        help="What to feed GLIP as the text prompt: full question, "
                             "just the entity word ('edge'/'face'), or both as separate categories.")    
    # OPTION 2: top-K boxes per view (Find3D / PatchAlign3D analog).
    # If set (>0), AFTER GLIP runs at the threshold above we keep only the
    # top-K highest-scoring boxes per (view, category). Tunable per feature
    # type because faces typically need more boxes than edges.
    parser.add_argument("--topk_boxes", type=int, default=0,
                        help="If >0, override per-feature top-K and keep this "
                             "many top-scoring GLIP boxes per (view, category).")
    parser.add_argument("--topk_boxes_face", type=int, default=3,
                        help="Top-K GLIP boxes per view per category for FACE samples "
                             "(used when --topk_boxes==0). 0 disables filtering.")
    parser.add_argument("--topk_boxes_edge", type=int, default=1,
                        help="Top-K GLIP boxes per view per category for EDGE samples "
                             "(used when --topk_boxes==0). 0 disables filtering.")
    # Strip whole-object boxes BEFORE top-K. PartSLIP's bbox2seg already drops
    # boxes covering >=98% of the visible point cloud, but that happens AFTER
    # top-K, so a single high-confidence whole-object detection can wipe out
    # all kept boxes. This pre-filter uses the SAME visible-point-fraction
    # criterion as bbox2seg, applied per-view, before top-K.
    parser.add_argument("--whole_obj_pc_thr", type=float, default=0.98,
                        help="Drop GLIP boxes covering this fraction or more of "
                             "the visible 3D points in their view, BEFORE top-K. "
                             "Mirrors PartSLIP's internal 0.98 guard. Set to 1.0 to disable.")
    parser.add_argument("--glip_batch_size", type=int, default=10,
                        help="Number of rendered views processed by GLIP in a single "
                             "forward pass. 0 or negative reverts to per-view serial "
                             "inference.")
    parser.add_argument("--save_visualization", type=int, default=0,
                        help="If 1, render predicted-vs-GT mesh visualizations with "
                             "PyVista (slow, several seconds/sample). Default 0.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Process samples with dataloader index in [start_idx, end_idx). "
                             "Combined with --end_idx, lets you shard the work across "
                             "multiple parallel processes on the same/different GPUs.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Exclusive upper bound; -1 = use len(val_loader).")
    # OPTION 3: top-X% of ALL boxes by score across views (Find3D / PatchAlign3D analog).
    # Applied AFTER --glip_conf and AFTER (or instead of) Option 2.
    parser.add_argument("--topk_pct", type=float, default=0.0,
                        help="If >0, override per-feature top-X%% and keep this percentage "
                             "of the highest-scoring GLIP boxes across all views/categories.")
    parser.add_argument("--topk_pct_face", type=float, default=10.0,
                        help="Top-X%% of GLIP boxes for FACE samples (used when --topk_pct==0). "
                             "0 disables percentage filtering.")
    parser.add_argument("--topk_pct_edge", type=float, default=2.0,
                        help="Top-X%% of GLIP boxes for EDGE samples (used when --topk_pct==0). "
                             "0 disables percentage filtering.")
    parser.add_argument("--preset", type=str, default="conf",
                        choices=["conf", "topk", "topk_pct"],
                        help="'conf' = pure confidence-threshold mode (no top-K filtering). "
                             "'topk' = enable per-view top-K filtering with the "
                             "--topk_boxes_face / --topk_boxes_edge defaults. "
                             "'topk_pct' = keep top-X%% of all boxes (face=10, edge=2).")
    parser.add_argument("--experiment_name", type=str, default="GeLoM_PartSLIP_zeroshot")
    parser.add_argument("--config", type=str, default="GLIP/configs/glip_Swin_L.yaml")
    parser.add_argument("--weight", type=str, default="models/glip_large_model.pth")
    parser.add_argument("--val_dataset_log", type=str, default=None,
                        help="Optional dataset log restricting the val set (e.g. the 1535 set).")
    parser.add_argument("--entity_allowlist", type=str, default=None,
                        help="Optional file of 'cad,feature,idx' keys to restrict evaluation.")
    args = parser.parse_args()

    # Preset wiring
    if args.preset == "conf":
        # disable top-K filtering; rely entirely on --glip_conf
        args.topk_boxes = 0
        args.topk_boxes_face = 0
        args.topk_boxes_edge = 0
        args.topk_pct = 0.0
        args.topk_pct_face = 0.0
        args.topk_pct_edge = 0.0
        if args.experiment_name == "GeLoM_PartSLIP_zeroshot":
            args.experiment_name = f"GeLoM_PartSLIP_zeroshot_conf{args.glip_conf}"
    elif args.preset == "topk":
        # per-view top-K only; disable percentage
        args.topk_pct = 0.0
        args.topk_pct_face = 0.0
        args.topk_pct_edge = 0.0
        if args.experiment_name == "GeLoM_PartSLIP_zeroshot":
            args.experiment_name = (
                f"GeLoM_PartSLIP_zeroshot_topk_face{args.topk_boxes_face}"
                f"_edge{args.topk_boxes_edge}"
            )
    elif args.preset == "topk_pct":
        # percentage only; disable per-view K
        args.topk_boxes = 0
        args.topk_boxes_face = 0
        args.topk_boxes_edge = 0
        args.topk_pct = 0.0
        args.topk_pct_face = 10.0
        args.topk_pct_edge = 2.0
        if args.experiment_name == "GeLoM_PartSLIP_zeroshot":
            args.experiment_name = (
                f"GeLoM_PartSLIP_zeroshot_topkpct_face{int(args.topk_pct_face)}"
                f"_edge{int(args.topk_pct_edge)}"
            )
    print(f"[config] glip_conf={args.glip_conf} "
          f"topk_boxes={args.topk_boxes} "
          f"topk_boxes_face={args.topk_boxes_face} "
          f"topk_boxes_edge={args.topk_boxes_edge} "
          f"topk_pct={args.topk_pct} "
          f"topk_pct_face={args.topk_pct_face} "
          f"topk_pct_edge={args.topk_pct_edge} "
          f"experiment_name={args.experiment_name}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # ------------------------------------------------------------------
    # Dataset / loader (mirrors GeoLocLM.py)
    # ------------------------------------------------------------------
    entity_allowlist = None
    if args.entity_allowlist:
        _ap = args.entity_allowlist
        if not os.path.isabs(_ap):
            _ap = os.path.join(LISA_ROOT, _ap)
        entity_allowlist = set()
        with open(_ap) as _fh:
            for _ln in _fh:
                _ln = _ln.strip()
                if not _ln or _ln.startswith("#"):
                    continue
                _c, _ft, _fi = _ln.split(",")
                entity_allowlist.add((str(_c).strip(), _ft.strip(), int(_fi)))
        print(f"[entity_allowlist] {len(entity_allowlist)} keys from {_ap}")
    val_dataset = CAD_ViewRank_Dataset(
        clip_image_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16"),
        split="val",
        dataset_log_path=args.val_dataset_log,
        entity_allowlist=entity_allowlist,
    )
    total_n = len(val_dataset)
    end_idx_arg = total_n if args.end_idx < 0 else min(args.end_idx, total_n)
    start_idx_arg = max(0, args.start_idx)
    print(f"[shard] processing dataset indices [{start_idx_arg}, {end_idx_arg}) of {total_n}")
    shard_dataset = torch.utils.data.Subset(val_dataset, range(start_idx_arg, end_idx_arg))
    val_loader = torch.utils.data.DataLoader(
        shard_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=views_collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    # ------------------------------------------------------------------
    # Load GLIP once
    # ------------------------------------------------------------------
    print("[loading GLIP model...]")
    glip_demo = load_model(args.config, args.weight)

    experiment_name = args.experiment_name
    os.makedirs(experiment_name, exist_ok=True)
    log_filepath = f"{experiment_name}/localization_metrics_partslip.log"
    if not os.path.exists(log_filepath):
        open(log_filepath, "w").close()

    all_metrics = []

    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        cad_folder_path = batch["image_paths"][0][0].split("/mesh_views_corrected/")[0]

        # Locate .step and .obj
        cad_file_path, mesh_path = None, None
        for fname in os.listdir(cad_folder_path):
            if fname.endswith(".step"):
                cad_file_path = os.path.join(cad_folder_path, fname)
            if fname.endswith(".obj"):
                mesh_path = os.path.join(cad_folder_path, fname)
        if cad_file_path is None or mesh_path is None:
            print(f"[skip] missing .step/.obj in {cad_folder_path}")
            continue
        cad_file_name = os.path.basename(cad_folder_path)

        # Resolve question, feature, feature_idx by EXACT question match. The
        # feature CANNOT be inferred from the question text ('edge' in ques),
        # because face captions often mention bounding edges (and vice-versa),
        # which silently grabs the wrong entity. Search both corrected caption
        # logs and take the authoritative feature from the matched entry.
        chosen_ques = batch["question"][0]
        feature, part_dict = None, None
        for _feat_try in ("edge", "face"):
            _lp = f"{cad_folder_path}/views_and_ques_{_feat_try}_augmented_corrected.log"
            if not os.path.exists(_lp):
                continue
            with open(_lp, "r", encoding="utf-8") as f:
                for line in f:
                    _cand = ast.literal_eval(line.strip())
                    if _cand["question"] == chosen_ques:
                        part_dict = _cand
                        feature = _feat_try
                        break
            if part_dict is not None:
                break
        if part_dict is None:
            print(f"[skip] question not found in caption logs for {cad_file_name}")
            continue
        chosen_marked_view_path = part_dict["marked_image"]
        feature_idx = int(os.path.basename(chosen_marked_view_path).split("[")[1].split("]")[0])

        # Captions in *_augmented_corrected.log are already L/R-corrected, so no
        # swap is applied; feed the question verbatim to GLIP.
        chosen_ques_for_inference = chosen_ques

        # Build the GLIP prompt list. PartSLIP/GLIP grounds noun phrases best,
        # so a long relational sentence often returns 0 detections.
        if args.prompt_mode == "question":
            part_names_glip = [chosen_ques_for_inference]
        elif args.prompt_mode == "feature":
            part_names_glip = [feature]  # "edge" or "face"
        else:  # "feature_question"
            part_names_glip = [feature, chosen_ques_for_inference]

        # Load and stitch mesh
        mesh = trimesh.load(mesh_path, process=True)
        mesh.vertices, mesh.faces = stitch_mesh_topology(mesh.vertices, mesh.faces)

        # GT features (timeout-protected)
        gt_features = compute_gt_features(
            cad_file_path, mesh.vertices, mesh.faces, feature, feature_idx,
            timeout_s=args.gt_timeout,
        )
        if gt_features is None:
            continue

        # Sample point cloud, keeping per-point face index for back-projection
        xyz, rgb, face_idx_per_point = sample_pointcloud_from_mesh(mesh, n_points=args.n_points)

        sample_save_dir = os.path.join(experiment_name, f"{cad_file_name}_{feature}_{feature_idx}")
        os.makedirs(sample_save_dir, exist_ok=True)

        # ----- timed PartSLIP inference -----
        t0 = time.time()
        try:
            img_dir, pc_idx, screen_coords = render_pc(xyz, rgb, sample_save_dir, device)
            if args.glip_batch_size and args.glip_batch_size > 0:
                preds = glip_inference_batched(
                    glip_demo, sample_save_dir, part_names_glip,
                    num_views=args.num_views, confidence=args.glip_conf,
                    batch_size=args.glip_batch_size,
                    save_pred_img=bool(args.save_visualization),
                )
            else:
                preds = glip_inference(
                    glip_demo, sample_save_dir, part_names_glip,
                    num_views=args.num_views, confidence=args.glip_conf,
                    save_pred_img=bool(args.save_visualization),
                )
            n_preds_raw = len(preds)

            # ---- Strip whole-object boxes BEFORE top-K ----
            # Use the SAME per-view visible-point-fraction criterion that
            # bbox2seg uses internally (its 0.98 guard). Doing it here
            # ensures top-K does not pick boxes that bbox2seg would then
            # silently discard, which previously left us with 0 P/R when
            # K was small and GLIP's top boxes were whole-object detections.
            if args.whole_obj_pc_thr < 1.0 and len(preds) > 0:
                # Precompute per-view visible-points screen coordinates once.
                vis_screen_per_view = []
                for vi in range(args.num_views):
                    sc = screen_coords[vi]
                    pidx = pc_idx[vi]
                    visible_pts = np.unique(pidx)
                    visible_pts = visible_pts[visible_pts >= 0]
                    vis_screen_per_view.append(sc[visible_pts] if len(visible_pts) else None)
                kept = []
                for p in preds:
                    vs = vis_screen_per_view[p["image_id"]]
                    if vs is None or len(vs) == 0:
                        continue  # nothing visible in this view -> drop
                    x1, y1, w, h = p["bbox"]
                    x2, y2 = x1 + w, y1 + h
                    inside = ((vs[:, 0] > x1) & (vs[:, 0] < x2) &
                              (vs[:, 1] > y1) & (vs[:, 1] < y2))
                    if inside.mean() < args.whole_obj_pc_thr:
                        kept.append(p)
                preds = kept
            n_preds_after_whole_filter = len(preds)

            # ---- OPTION 2: top-K boxes per (view, category) ----
            # `preds` items: {image_id, category_id, bbox, score}
            if args.topk_boxes > 0:
                k_per_cat = args.topk_boxes
            else:
                k_per_cat = (args.topk_boxes_edge if feature == "edge"
                             else args.topk_boxes_face)
            if k_per_cat > 0 and len(preds) > 0:
                from collections import defaultdict
                grouped = defaultdict(list)
                for p in preds:
                    grouped[(p["image_id"], p["category_id"])].append(p)
                kept = []
                for _, group in grouped.items():
                    group.sort(key=lambda x: x["score"], reverse=True)
                    kept.extend(group[:k_per_cat])
                preds = kept

            # ---- OPTION 3: top-X% of boxes by score, PER (view, category) ----
            # Applied per (view, category) — NOT globally across all views.
            # A global top-X% would keep ~1 box for a 10-view sample with
            # X=2-10%, which is far too sparse for PartSLIP's bbox2seg
            # (each superpoint needs >=50% average visible-point coverage
            # across views to receive a label). Per-view percentage keeps
            # the voting density needed by the algorithm.
            if args.topk_pct > 0:
                pct = args.topk_pct
            else:
                pct = (args.topk_pct_edge if feature == "edge"
                       else args.topk_pct_face)
            if pct > 0 and len(preds) > 0:
                from collections import defaultdict
                grouped = defaultdict(list)
                for p in preds:
                    grouped[(p["image_id"], p["category_id"])].append(p)
                kept = []
                for _, group in grouped.items():
                    group.sort(key=lambda x: x["score"], reverse=True)
                    n_keep = max(1, int(round(len(group) * pct / 100.0)))
                    kept.extend(group[:n_keep])
                preds = kept

            print(f"[{cad_file_name}] GLIP detections={len(preds)}/{n_preds_after_whole_filter}/{n_preds_raw} "
                  f"(raw->no-whole-obj->topk; views={args.num_views}, prompt_mode={args.prompt_mode}, "
                  f"conf={args.glip_conf}, whole_obj_pc_thr={args.whole_obj_pc_thr}, "
                  f"topk_per_view_cat={k_per_cat}, topk_pct={pct})")
            superpoint = gen_superpoint(xyz, rgb, visualize=False, save_dir=sample_save_dir)
            sem_seg, _ = bbox2seg(
                xyz, superpoint, preds, screen_coords, pc_idx,
                part_names_glip, sample_save_dir,
                num_view=args.num_views, solve_instance_seg=False, visualize=False,
            )
        except Exception as e:
            print(f"[partslip error] {cad_file_name}: {e}")
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Project labelled points back to mesh faces.
        # In feature_question mode the question is the *second* category (id=2).
        if args.prompt_mode == "feature_question":
            target_label = 1
        else:
            target_label = 0
        pred_point_mask = (sem_seg == target_label)
        n_pred_points = int(pred_point_mask.sum())
        print(f"[{cad_file_name}] predicted points={n_pred_points}/{len(sem_seg)}")
        pred_face_set = np.unique(face_idx_per_point[pred_point_mask]) if pred_point_mask.any() else np.array([], dtype=np.int64)

        if feature == "face":
            pred_entities = pred_face_set.tolist()
        else:  # edge
            pred_entities = faces_to_edges(mesh.faces, pred_face_set)
        inference_time = time.time() - t0
        # ----- end timed -----

        metrics = return_localization_metrics(
            mesh_path, mesh.vertices, mesh.faces,
            pred_entities, gt_features, feature, feature_idx,
        )
        metrics["inference_time"] = inference_time
        metrics["cad_file"] = cad_file_name
        all_metrics.append(metrics)

        with open(log_filepath, "a") as f:
            f.write(f"{metrics}\n")

        # Render mesh with predicted vs GT entities (yellow=correct, red=pred-only, green=GT-only)
        if args.save_visualization:
            view_inspect_paths = visualize_entity_predictions(
                mesh.vertices, mesh.faces,
                pred_entities, gt_features, feature,
                save_prefix=os.path.join(sample_save_dir, "vis"),
            )
            print(f"Saved visualization for {cad_file_name} to {view_inspect_paths}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    if all_metrics:
        avg = {"edge": {"iou": 0, "precision": 0, "recall": 0, "F1": 0},
               "face": {"iou": 0, "precision": 0, "recall": 0, "F1": 0}}
        counts = {"edge": 0, "face": 0}
        avg_inf = 0.0
        for m in all_metrics:
            ft = m["feature"]
            counts[ft] += 1
            for k in avg[ft]:
                avg[ft][k] += m[k]
            avg_inf += m.get("inference_time", 0.0)
        for ft in avg:
            if counts[ft]:
                for k in avg[ft]:
                    avg[ft][k] /= counts[ft]
        avg_inf /= len(all_metrics)

        with open(log_filepath, "a") as f:
            f.write("\n====================================\n")
            f.write(f"Avg Metrics (samples={len(all_metrics)}, "
                    f"edge={counts['edge']}, face={counts['face']}): {avg}\n")
            f.write(f"Avg Inference Time: {avg_inf:.2f} seconds\n")
            f.write("====================================\n")
        print(avg)
        print(f"Avg inference time: {avg_inf:.2f}s")
