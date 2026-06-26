"""
Run Find3D zero-shot localization on the same CAD validation dataset used by
GeoLocLM.py and evaluate with the same `return_localization_metrics`.

Pipeline per CAD sample:
    .obj mesh  ──▶  surface-sample point cloud (xyz, rgb, normals; face indices kept)
                ──▶  Find3D preprocess (normalize / axis-swap / grid sample)
                ──▶  Find3D model forward → per-(subsampled-)point features
                ──▶  NN upsample to full point cloud, argmax against text queries
                ──▶  project labelled points back to mesh face indices
                ──▶  for `face`: predicted faces = those face indices
                    for `edge`: predicted edges = unique edges of predicted faces
                ──▶  return_localization_metrics(...)

This must be run from the Find3D root with the env that has Find3D's deps
(Pointcept / FlashAttention) installed (e.g. `find3d`). FreeCAD-based GT
extraction is shelled out to LISA's env via `_gt_features_helper.py`, the same
mechanism the PartSLIP comparison script uses.
"""

import os
import re
import sys
import ast
import time
import argparse

# Run from this directory so Find3D's relative imports / configs resolve
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)

# Find3D root must be on PYTHONPATH so `model.*` and `common.*` import
sys.path.insert(0, THIS_DIR)

# LISA root for dataset and eval utilities (two levels up: baselines/find3d/)
LISA_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, LISA_ROOT)

# Stub out LISA's `model.llava.model` submodule before importing
# `utils.dataset_`. Importing dataset_ triggers `from model.llava import
# conversation`, which runs `model/llava/__init__.py`, which eagerly imports
# LlavaLlamaForCausalLM and pulls in MPT/Bloom code that breaks against newer
# `transformers` (`_expand_mask` was removed). Find3D doesn't need any LLaVA
# code; pre-registering a stub at the right path short-circuits the chain.
import types as _types
_llava_model_stub = _types.ModuleType("model.llava.model")
_llava_model_stub.LlavaLlamaForCausalLM = None
sys.modules.setdefault("model.llava.model", _llava_model_stub)

import numpy as np
import torch
import trimesh
from tqdm import tqdm
from transformers import CLIPImageProcessor

# LISA imports
from utils.dataset_ import CAD_ViewRank_Dataset, views_collate_fn  # noqa: E402
from eval_utils import (  # noqa: E402
    stitch_mesh_topology,
    return_localization_metrics,
    visualize_entity_predictions,
)

# Find3D imports
from model.evaluation.utils import (  # noqa: E402
    load_model,
    preprocess_pcd,
    encode_text,
    set_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sample_pointcloud_from_mesh(mesh, n_points=10000):
    """Surface-sample a mesh and return (xyz, rgb, normal, face_idx_per_point)
    as float32 / int64 numpy arrays. Coordinates are returned in the mesh's
    original frame; Find3D's `preprocess_pcd` will normalize them.
    """
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    xyz = np.asarray(points, dtype=np.float32)
    face_idx = np.asarray(face_idx, dtype=np.int64)

    # Per-point normal = normal of the source face
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)
    normal = face_normals[face_idx]
    n = np.linalg.norm(normal, axis=1, keepdims=True)
    n[n == 0] = 1.0
    normal = normal / n

    # Neutral gray in [0,1]; Find3D's preprocess expects rgb.max()<=1
    rgb = np.full_like(xyz, 0.5, dtype=np.float32)
    return xyz, rgb, normal, face_idx


def faces_to_edges(faces, face_indices):
    """Return unique sorted (v0, v1) edge tuples for a set of face indices."""
    edges = set()
    for fi in face_indices:
        f = faces[int(fi)]
        edges.add(tuple(sorted((int(f[0]), int(f[1])))))
        edges.add(tuple(sorted((int(f[1]), int(f[2])))))
        edges.add(tuple(sorted((int(f[2]), int(f[0])))))
    return list(edges)


# ---------------------------------------------------------------------------
# L/R caption fix (mirror the GeoLocLM real-world spatial convention fix)
# ---------------------------------------------------------------------------
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
    if not text:
        return text
    return _LR_PATTERN.sub(lambda m: _LR_SWAP[m.group()], text)


# ---------------------------------------------------------------------------
# Cross-env GT extraction (FreeCAD lives in LISA_multi_view env)
# ---------------------------------------------------------------------------
GT_HELPER_ENV = "LISA_multi_view"
GT_HELPER_SCRIPT = os.path.abspath(os.path.join(LISA_ROOT, "_gt_features_helper.py"))


def _ensure_gt_helper_script():
    helper_src = '''import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import cad_entity_to_mesh_faces

def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

data = json.loads(sys.stdin.read())
vertices = np.array(data["vertices"], dtype=np.float64)
faces = np.array(data["faces"], dtype=np.int64)
try:
    out = cad_entity_to_mesh_faces(
        data["cad_file_path"], vertices, faces,
        entity_type=data["feature"], entity_index=data["feature_idx"],
    )
    if isinstance(out, np.ndarray):
        out = out.tolist()
    elif isinstance(out, list):
        out = [list(e) if hasattr(e, "__iter__") else int(e) for e in out]
    sys.stdout.write("@@RESULT@@" + json.dumps({"ok": True, "result": out}, default=_default))
except Exception as e:
    sys.stdout.write("@@RESULT@@" + json.dumps({"ok": False, "err": repr(e)}))
'''
    # Always (re)write so updates to the template propagate.
    with open(GT_HELPER_SCRIPT, "w") as f:
        f.write(helper_src)


def compute_gt_features(cad_file_path, vertices, faces, feature, feature_idx, timeout_s=20):
    import json, subprocess
    _ensure_gt_helper_script()

    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    payload = json.dumps({
        "cad_file_path": str(cad_file_path),
        "vertices": np.asarray(vertices, dtype=np.float64).tolist(),
        "faces": np.asarray(faces, dtype=np.int64).tolist(),
        "feature": str(feature),
        "feature_idx": int(feature_idx),
    }, default=_json_default)
    cmd = ["conda", "run", "--no-capture-output", "-n", GT_HELPER_ENV,
           "python", GT_HELPER_SCRIPT]
    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(f"[GT timeout] {cad_file_path}")
        return None
    out = proc.stdout
    if "@@RESULT@@" not in out:
        print(f"[GT error] {cad_file_path}: helper produced no result. stderr={proc.stderr[-500:]}")
        return None
    try:
        result = json.loads(out.split("@@RESULT@@", 1)[1])
    except json.JSONDecodeError as e:
        print(f"[GT error] {cad_file_path}: bad json ({e})")
        return None
    if not result.get("ok"):
        print(f"[GT error] {cad_file_path}: {result.get('err')}")
        return None
    return result["result"]


# ---------------------------------------------------------------------------
# Find3D forward + upsample (mirrors compute_3d_iou_upsample, sans GT)
# ---------------------------------------------------------------------------
@torch.no_grad()
def find3d_predict_per_point(model, data_dict, text_embeds, n_chunks=5):
    """Return (pred_label_per_full_point [N], target_sim_per_full_point [N]).

    `pred_label` is the standard Find3D argmax label (0 = unlabeled,
    1..n_queries = the queries). `target_sim` is the raw cosine-like similarity
    of every full-resolution point against the FIRST query, so we can apply
    threshold / top-k selection downstream.
    """
    # Move tensors to GPU
    for key in list(data_dict.keys()):
        if isinstance(data_dict[key], torch.Tensor) and "full" not in key:
            data_dict[key] = data_dict[key].cuda(non_blocking=True)

    temperature = float(np.exp(model.ln_logit_scale.item()))
    net_out = model(x=data_dict)  # [n_sub, feat_dim]

    xyz_sub = data_dict["coord"]               # [n_sub, 3] (cuda)
    xyz_full = data_dict["xyz_full"].squeeze() # [N, 3] (cpu in preprocess_pcd output)

    # L2-normalize point features so logits are cosine similarities in [-1, 1]
    net_norm = net_out / (net_out.norm(dim=-1, keepdim=True) + 1e-12)
    sims = net_norm @ text_embeds.T  # [n_sub, n_queries], cosine-like

    # Standard Find3D argmax label (with prepended 0 for "unlabeled")
    logits_p0 = torch.cat([torch.zeros(sims.shape[0], 1, device=sims.device), sims], dim=1)
    pred_softmax = torch.softmax(logits_p0 * temperature, dim=1)  # [n_sub, n_queries+1]

    # NN-upsample full points → nearest subsampled point (chunked)
    chunk_len = xyz_full.shape[0] // n_chunks + 1
    nn_idxs = []
    for i in range(n_chunks):
        chunk = xyz_full[chunk_len * i: chunk_len * (i + 1)]
        if chunk.shape[0] == 0:
            continue
        d = ((xyz_sub.unsqueeze(0) - chunk.cuda().unsqueeze(1)) ** 2).sum(-1).sqrt()  # [chunk, n_sub]
        nn_idxs.append(torch.min(d, dim=1)[1])
        del d
    nn_idxs = torch.cat(nn_idxs, dim=0)

    all_probs = pred_softmax[nn_idxs]                       # [N, n_queries+1]
    pred_label = all_probs.argmax(dim=1).cpu().numpy()      # [N]
    target_sim = sims[nn_idxs, 0].cpu().numpy()             # [N], similarity vs FIRST query
    return pred_label, target_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find3D localization on CAD val set")
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument("--gt_timeout", type=int, default=20)
    parser.add_argument("--prompt_mode", type=str, default="question",
                        choices=["question", "feature", "feature_question"],
                        help="What to feed Find3D as the text query: full question, "
                             "the entity word ('edge'/'face'), or both as separate queries.")
    parser.add_argument("--experiment_name", type=str, default="GeLoM_Find3D_zeroshot")
    parser.add_argument("--n_chunks", type=int, default=5,
                        help="NN-upsample chunking (lower = more memory).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection_mode", type=str, default="topk_pct",
                        choices=["argmax", "threshold", "topk_pct"],
                        help="How to pick target points. "
                             "argmax: standard Find3D label (loose); "
                             "threshold: cosine sim vs first query > --sim_threshold; "
                             "topk_pct: keep top --topk_pct%% by similarity.")
    parser.add_argument("--sim_threshold", type=float, default=0.15,
                        help="Cosine-sim cutoff used when --selection_mode=threshold.")
    parser.add_argument("--topk_pct", type=float, default=None,
                        help="Override percentage of points to keep when "
                             "--selection_mode=topk_pct. If unset, uses "
                             "--topk_pct_face for face prompts and "
                             "--topk_pct_edge for edge prompts.")
    parser.add_argument("--topk_pct_face", type=float, default=10.0,
                        help="topk%% used when the prompt targets a face.")
    parser.add_argument("--topk_pct_edge", type=float, default=2.0,
                        help="topk%% used when the prompt targets an edge.")
    parser.add_argument("--negative_queries", type=str, nargs="*", default=None,
                        help="Extra negative queries appended to the query list "
                             "(only relevant for --selection_mode=argmax).")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["default", "topk"],
                        help="Convenience switch overriding --selection_mode. "
                             "'default' = paper-faithful argmax (no negatives, no topk). "
                             "'topk'    = topk_pct with face=10, edge=2. "
                             "If --experiment_name is left at its default it gets "
                             "suffixed with the preset name so parallel runs land "
                             "in separate folders/log files.")
    parser.add_argument("--val_dataset_log", type=str, default=None,
                        help="Optional dataset log restricting the val set (e.g. the 1535 set).")
    parser.add_argument("--entity_allowlist", type=str, default=None,
                        help="Optional file of 'cad,feature,idx' keys to restrict evaluation.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Process dataset indices in [start_idx, end_idx) for sharding.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Exclusive upper bound; -1 = use len(val_dataset).")
    args = parser.parse_args()

    # Apply preset (overrides selection_mode and tags experiment_name)
    if args.preset == "default":
        args.selection_mode = "argmax"
        args.negative_queries = None
        if args.experiment_name == "GeLoM_Find3D_zeroshot":
            args.experiment_name = "GeLoM_Find3D_zeroshot_default"
    elif args.preset == "topk":
        args.selection_mode = "topk_pct"
        args.topk_pct = None  # use feature-dependent face=10 / edge=2
        args.topk_pct_face = 10.0
        args.topk_pct_edge = 2.0
        if args.experiment_name == "GeLoM_Find3D_zeroshot":
            args.experiment_name = "GeLoM_Find3D_zeroshot_topk_face10_edge2"
    print(f"[config] selection_mode={args.selection_mode} "
          f"experiment_name={args.experiment_name}")

    set_seed(args.seed)
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
    # Load Find3D once (downloads from HF on first call)
    # ------------------------------------------------------------------
    print("[loading Find3D model...]")
    model = load_model()  # already .eval().cuda()

    experiment_name = args.experiment_name
    os.makedirs(experiment_name, exist_ok=True)
    log_filepath = f"{experiment_name}/localization_metrics_find3d.log"
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

        # Resolve question, feature, feature_idx by EXACT question match across
        # both corrected caption logs (feature can't be inferred from the text).
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

        # Captions in *_augmented_corrected.log are already L/R-corrected; feed verbatim.
        chosen_ques_for_inference = chosen_ques

        # Build the text query list. Find3D scores each point against each query;
        # the first query is treated as the target for back-projection.
        if args.prompt_mode == "question":
            queries = [chosen_ques_for_inference]
        elif args.prompt_mode == "feature":
            queries = [feature]
        else:  # "feature_question"
            queries = [chosen_ques_for_inference, feature]

        # Optional negative / distractor queries make argmax more selective.
        if args.negative_queries:
            queries = queries + list(args.negative_queries)

        # Load and stitch mesh
        mesh = trimesh.load(mesh_path, process=True)
        mesh.vertices, mesh.faces = stitch_mesh_topology(mesh.vertices, mesh.faces)

        # GT features
        gt_features = compute_gt_features(
            cad_file_path, mesh.vertices, mesh.faces, feature, feature_idx,
            timeout_s=args.gt_timeout,
        )
        if gt_features is None:
            continue

        # Sample point cloud, keeping per-point face index for back-projection
        xyz_np, rgb_np, normal_np, face_idx_per_point = sample_pointcloud_from_mesh(
            mesh, n_points=args.n_points,
        )

        sample_save_dir = os.path.join(experiment_name, f"{cad_file_name}_{feature}_{feature_idx}")
        os.makedirs(sample_save_dir, exist_ok=True)

        # ----- timed Find3D inference -----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        #try:
        xyz_t = torch.from_numpy(xyz_np).cuda()
        rgb_t = torch.from_numpy(rgb_np).cuda()
        normal_t = torch.from_numpy(normal_np).cuda()
        data_dict = preprocess_pcd(xyz_t, rgb_t, normal_t)
        text_embeds = encode_text(queries)  # [n_queries, dim]
        pred_label, target_sim = find3d_predict_per_point(
            model, data_dict, text_embeds, n_chunks=args.n_chunks,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # except Exception as e:
        #     print(f"[find3d error] {cad_file_name}: {e}")
        #     continue
        network_time = time.time() - t0

        # NOTE: preprocess_pcd subsamples to 5000 points if N > 5000 BEFORE
        # producing xyz_full; in that case xyz_full and face_idx_per_point would
        # no longer align. We only call back-projection when n_points <= 5000
        # (i.e. xyz_full has the same N points in the same order). For larger
        # n_points, fall back to the random-subsample indices we'd need to
        # recreate; here we keep n_points<=5000 by default to avoid the issue.
        if args.n_points > 5000:
            print(f"[warn] n_points={args.n_points} > 5000; preprocess_pcd "
                  f"subsamples internally and breaks face-index alignment. "
                  f"Use --n_points <= 5000.")
            continue

        loc_t0 = time.time()
        if args.selection_mode == "argmax":
            # First query (post L/R-swapped question) -> label 1 after prepended 0
            pred_point_mask = (pred_label == 1)
        elif args.selection_mode == "threshold":
            pred_point_mask = (target_sim > args.sim_threshold)
        else:  # topk_pct
            if args.topk_pct is not None:
                topk_pct = args.topk_pct
            else:
                topk_pct = args.topk_pct_edge if feature == "edge" else args.topk_pct_face
            n_keep = max(1, int(round(len(target_sim) * topk_pct / 100.0)))
            top_idx = np.argpartition(-target_sim, n_keep - 1)[:n_keep]
            pred_point_mask = np.zeros_like(target_sim, dtype=bool)
            pred_point_mask[top_idx] = True

        n_pred_points = int(pred_point_mask.sum())
        print(f"[{cad_file_name}] mode={args.selection_mode} "
              f"predicted points={n_pred_points}/{len(pred_label)} "
              f"(sim min/median/max = {target_sim.min():.3f}/"
              f"{np.median(target_sim):.3f}/{target_sim.max():.3f})")
        if pred_point_mask.any():
            pred_face_set = np.unique(face_idx_per_point[pred_point_mask])
        else:
            pred_face_set = np.array([], dtype=np.int64)

        if feature == "face":
            pred_entities = pred_face_set.tolist()
        else:  # edge
            pred_entities = faces_to_edges(mesh.faces, pred_face_set)
        localization_time = time.time() - loc_t0
        # ----- end timed -----

        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) \
            if torch.cuda.is_available() else 0.0

        metrics = return_localization_metrics(
            mesh_path, mesh.vertices, mesh.faces,
            pred_entities, gt_features, feature, feature_idx,
        )
        # network_time = preprocess + text encode + Find3D forward + NN upsample
        # localization_time = back-projection only (logged separately, not folded
        # into inference_time, mirroring GeoLocLM.py).
        metrics["lisa_forward_time"] = network_time  # field name kept for log compatibility
        metrics["network_time"] = network_time
        metrics["localization_time"] = localization_time
        metrics["inference_time"] = network_time
        metrics["peak_vram_mb"] = peak_vram_mb
        metrics["cad_file"] = cad_file_name
        all_metrics.append(metrics)

        with open(log_filepath, "a") as f:
            f.write(f"{metrics}\n")

        # Render mesh with predicted vs GT entities
        try:
            view_inspect_paths = visualize_entity_predictions(
                mesh.vertices, mesh.faces,
                pred_entities, gt_features, feature,
                save_prefix=os.path.join(sample_save_dir, "vis"),
            )
            print(f"Saved visualization for {cad_file_name} to {view_inspect_paths}")
        except Exception as e:
            print(f"[vis error] {cad_file_name}: {e}")

        print(f"[{cad_file_name}] Metrics: {metrics}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    if all_metrics:
        avg = {"edge": {"iou": 0, "precision": 0, "recall": 0, "F1": 0},
               "face": {"iou": 0, "precision": 0, "recall": 0, "F1": 0}}
        counts = {"edge": 0, "face": 0}
        avg_inf = 0.0
        avg_loc = 0.0
        avg_vram = 0.0
        for m in all_metrics:
            ft = m["feature"]
            counts[ft] += 1
            for k in avg[ft]:
                avg[ft][k] += m[k]
            avg_inf += m.get("inference_time", 0.0)
            avg_loc += m.get("localization_time", 0.0)
            avg_vram += m.get("peak_vram_mb", 0.0)
        for ft in avg:
            if counts[ft]:
                for k in avg[ft]:
                    avg[ft][k] /= counts[ft]
        avg_inf /= len(all_metrics)
        avg_loc /= len(all_metrics)
        avg_vram /= len(all_metrics)

        with open(log_filepath, "a") as f:
            f.write("\n====================================\n")
            f.write(f"Avg Metrics (samples={len(all_metrics)}, "
                    f"edge={counts['edge']}, face={counts['face']}): {avg}\n")
            f.write(f"Avg Inference Time (network only): {avg_inf:.2f} seconds\n")
            f.write(f"Avg Localization Time (back-projection): {avg_loc:.2f} seconds\n")
            f.write(f"Avg Peak VRAM: {avg_vram:.2f} MB\n")
            f.write("====================================\n")
        print(avg)
        print(f"Avg inference time: {avg_inf:.2f}s")
        print(f"Avg localization time: {avg_loc:.2f}s")
        print(f"Avg peak VRAM: {avg_vram:.2f} MB")
