"""
Run PatchAlign3D zero-shot localization on the same CAD validation dataset
used by GeoLocLM.py. Mirrors find3d_geolocalization.py / partslip_geolocalization.py
so all three baselines are directly comparable.

Pipeline per CAD sample:
    .obj mesh  ──▶  surface-sample point cloud (xyz, face indices kept)
                ──▶  PatchAlign3D point transformer → patch features (B,D,G)
                ──▶  PatchToTextProj → text-aligned patch features
                ──▶  encode the (L/R-swapped) question with OpenCLIP text encoder
                ──▶  patch-vs-text similarity → assign each point to its
                     nearest patch (or membership-weighted)
                ──▶  select target points (argmax / threshold / topk_pct)
                ──▶  back-project to mesh face indices
                ──▶  return_localization_metrics(...)

Run inside the `patchalign3d` conda env (has pointnet2_ops + KNN_CUDA + open_clip).
FreeCAD-based GT extraction is shelled out to LISA_multi_view, identical to the
PartSLIP / Find3D scripts.
"""

import os
import re
import sys
import ast
import time
import argparse
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)

# Make `patchalign3d.*` importable
sys.path.insert(0, THIS_DIR)

# LISA root for dataset and eval utilities (two levels up: baselines/patchalign3d/)
LISA_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, LISA_ROOT)

# Stub out LISA's `model.llava.model` submodule before importing
# `utils.dataset_` (importing it pulls in MPT/Bloom code that breaks against
# newer transformers; PatchAlign3D doesn't need any LLaVA code).
import types as _types
_llava_model_stub = _types.ModuleType("model.llava.model")
_llava_model_stub.LlavaLlamaForCausalLM = None
sys.modules.setdefault("model.llava.model", _llava_model_stub)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from easydict import EasyDict
from tqdm import tqdm
from transformers import CLIPImageProcessor

# LISA imports
from utils.dataset_ import CAD_ViewRank_Dataset, views_collate_fn  # noqa: E402
from eval_utils import (  # noqa: E402
    stitch_mesh_topology,
    return_localization_metrics,
    visualize_entity_predictions,
)

# PatchAlign3D imports (top-level package "patchalign3d")
import open_clip  # noqa: E402
from patchalign3d.models import point_transformer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (re-used patterns from infer.py and the other geoloc scripts)
# ---------------------------------------------------------------------------
PART_ONLY_TEMPLATES = ["{}", "a {}", "{} part"]
PART_PLUS_CAT_TEMPLATES = ["a {} of a {}", "the {} of a {}", "{} of {}", "a {} part of a {}"]


def _clean_text(s):
    s = s.strip().lower().replace("_", " ")
    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


class PatchToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb):
        x = patch_emb.transpose(1, 2)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


def prepare_points(points):
    """(B, N, 3) -> (B, 3, N) with Y/Z swap to match PatchAlign3D training."""
    if points.ndim != 3:
        raise ValueError(f"Expected (B,N,C), got {points.shape}")
    pts = points.transpose(2, 1).contiguous()
    pts[:, [1, 2], :] = pts[:, [2, 1], :]
    return pts


def assign_patch_logits_to_points(points_xyz, patch_centers, patch_logits, patch_idx, mode="nearest"):
    """Distribute (B, G, K) patch logits to (B, N, K) point logits.

    points_xyz: (B, 3, N)
    patch_centers: (B, 3, G)
    """
    B, _, N = points_xyz.shape
    K = patch_logits.shape[-1]
    if mode == "membership":
        point_logits = torch.zeros(B, N, K, device=points_xyz.device)
        counts = torch.zeros(B, N, 1, device=points_xyz.device)
        for b in range(B):
            idx = patch_idx[b].reshape(-1)
            src = patch_logits[b].unsqueeze(1).expand_as(patch_idx[b].unsqueeze(-1)).reshape(-1, K)
            point_logits[b].index_add_(0, idx, src)
            ones = torch.ones(idx.shape[0], 1, device=points_xyz.device)
            counts[b].index_add_(0, idx, ones)
        return point_logits / counts.clamp_min(1.0)
    # nearest patch via KNN
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)
    _, nearest = knn(patch_centers.transpose(1, 2).contiguous(),
                     points_xyz.transpose(1, 2).contiguous())
    nearest = nearest.squeeze(-1)
    return patch_logits.gather(1, nearest.unsqueeze(-1).expand(-1, -1, K))


def encode_clip_text(names, setting, clip_model, tokenizer, device):
    """Mirror infer.py::encode_texts. Average prompt-template embeddings per query."""
    texts = []
    for nm in names:
        nm = _clean_text(nm)
        if setting == "part_plus_cat":
            for tpl in PART_PLUS_CAT_TEMPLATES:
                texts.append(tpl.format(nm, "object"))
        else:
            for tpl in PART_ONLY_TEMPLATES:
                texts.append(tpl.format(nm) if tpl.count("{}") == 1 else nm)
    toks = tokenizer(texts).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(toks)
    feat = F.normalize(feat, dim=-1)
    if setting == "part_plus_cat":
        prompts_per_label = len(PART_PLUS_CAT_TEMPLATES)
    else:
        prompts_per_label = len(PART_ONLY_TEMPLATES)
    per_label = []
    for i in range(len(names)):
        chunk = feat[i * prompts_per_label : (i + 1) * prompts_per_label]
        per_label.append(F.normalize(chunk.mean(dim=0, keepdim=True), dim=-1))
    return torch.cat(per_label, dim=0)


def sample_pointcloud_from_mesh(mesh, n_points=10000):
    """Surface-sample a mesh keeping per-point face indices."""
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    xyz = np.asarray(points, dtype=np.float32)
    face_idx = np.asarray(face_idx, dtype=np.int64)
    # Center + scale to unit ball (consistent with the PartSLIP / Find3D scripts)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz = xyz / scale
    return xyz, face_idx


def faces_to_edges(faces, face_indices):
    edges = set()
    for fi in face_indices:
        f = faces[int(fi)]
        edges.add(tuple(sorted((int(f[0]), int(f[1])))))
        edges.add(tuple(sorted((int(f[1]), int(f[2])))))
        edges.add(tuple(sorted((int(f[2]), int(f[0])))))
    return list(edges)


# ---------------------------------------------------------------------------
# L/R caption fix
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
# Cross-env GT extraction (FreeCAD is in LISA_multi_view env)
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
        proc = subprocess.run(cmd, input=payload, capture_output=True,
                              text=True, timeout=timeout_s)
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
# Inference: forward + per-point similarity vs queries
# ---------------------------------------------------------------------------
@torch.no_grad()
def patchalign3d_predict_per_point(model, proj, clip_model, clip_tokenizer,
                                   xyz_np, queries, args, device):
    """Return (pred_label [N], target_sim [N]) for the full point cloud.

    `pred_label`: standard argmax label, 0 = "unlabeled" (prepended-0 class), 1..K = queries.
    `target_sim`: cosine similarity vs the FIRST query for every full-resolution point.
    """
    pts_t = torch.as_tensor(xyz_np, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    pts = prepare_points(pts_t).to(device)                             # (1, 3, N)

    patch_emb, patch_centers, patch_idx = model.forward_patches(pts)
    patch_feat = proj(patch_emb)  # (1, G, text_dim), L2-normalized

    text_feats = encode_clip_text(queries, args.text_setting, clip_model,
                                  clip_tokenizer, device)  # (K, text_dim)

    # Cosine similarities (text feats are L2-normalized too)
    sims = (patch_feat @ text_feats.t())  # (1, G, K) in [-1, 1]

    # Standard argmax label with prepended 0 (unlabeled) class
    logits = sims / max(float(args.clip_tau), 1e-6)
    logits_p0 = torch.cat([torch.zeros(1, logits.shape[1], 1, device=device), logits], dim=-1)  # (1, G, K+1)

    # Distribute patch-level results to all points
    point_sims_all = assign_patch_logits_to_points(
        pts[:, :3, :], patch_centers, sims, patch_idx, mode=args.assign,
    )  # (1, N, K)
    point_logits_p0 = assign_patch_logits_to_points(
        pts[:, :3, :], patch_centers, logits_p0, patch_idx, mode=args.assign,
    )  # (1, N, K+1)

    pred_label = point_logits_p0.argmax(dim=-1).squeeze(0).cpu().numpy()   # (N,)
    target_sim = point_sims_all[0, :, 0].cpu().numpy()                     # (N,) vs first query
    return pred_label, target_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchAlign3D localization on CAD val set")
    parser.add_argument("--ckpt", type=str,
                        default=os.path.join(THIS_DIR, "ckpts", "patchalign3d.pt"))
    parser.add_argument("--n_points", type=int, default=10000,
                        help="Number of surface samples per mesh.")
    parser.add_argument("--gt_timeout", type=int, default=20)
    parser.add_argument("--prompt_mode", type=str, default="question",
                        choices=["question", "feature", "feature_question"])
    parser.add_argument("--experiment_name", type=str, default="GeLoM_PatchAlign3D_zeroshot")
    parser.add_argument("--num_group", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--clip_model", type=str, default="ViT-bigG-14")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s39b_b160k")
    parser.add_argument("--clip_tau", type=float, default=0.07)
    parser.add_argument("--text_setting", type=str, default="part_only",
                        choices=["part_only", "part_plus_cat"])
    parser.add_argument("--assign", type=str, default="nearest",
                        choices=["nearest", "membership"],
                        help="How to spread (G,K) patch logits over (N,K) points.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection_mode", type=str, default="topk_pct",
                        choices=["argmax", "threshold", "topk_pct"],
                        help="argmax: paper-faithful (loose); "
                             "threshold: cosine sim vs first query > --sim_threshold; "
                             "topk_pct: keep top --topk_pct%% by similarity.")
    parser.add_argument("--sim_threshold", type=float, default=0.15)
    parser.add_argument("--topk_pct", type=float, default=None,
                        help="Override --topk_pct_face / --topk_pct_edge if set.")
    parser.add_argument("--topk_pct_face", type=float, default=10.0)
    parser.add_argument("--topk_pct_edge", type=float, default=2.0)
    parser.add_argument("--negative_queries", type=str, nargs="*", default=None)
    parser.add_argument("--preset", type=str, default=None,
                        choices=["default", "topk"],
                        help="'default' = paper-faithful argmax. "
                             "'topk' = topk_pct with face=10, edge=2.")
    parser.add_argument("--val_dataset_log", type=str, default=None,
                        help="Optional dataset log restricting the val set (e.g. the 1535 set).")
    parser.add_argument("--entity_allowlist", type=str, default=None,
                        help="Optional file of 'cad,feature,idx' keys to restrict evaluation.")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Process dataset indices in [start_idx, end_idx) for sharding.")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="Exclusive upper bound; -1 = use len(val_dataset).")
    args = parser.parse_args()

    if args.preset == "default":
        args.selection_mode = "argmax"
        args.negative_queries = None
        if args.experiment_name == "GeLoM_PatchAlign3D_zeroshot":
            args.experiment_name = "GeLoM_PatchAlign3D_zeroshot_default"
    elif args.preset == "topk":
        args.selection_mode = "topk_pct"
        args.topk_pct = None
        args.topk_pct_face = 10.0
        args.topk_pct_edge = 2.0
        if args.experiment_name == "GeLoM_PatchAlign3D_zeroshot":
            args.experiment_name = "GeLoM_PatchAlign3D_zeroshot_topk_face10_edge2"
    print(f"[config] selection_mode={args.selection_mode} "
          f"experiment_name={args.experiment_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
        shard_dataset, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=views_collate_fn, pin_memory=True, persistent_workers=False,
    )

    # ------------------------------------------------------------------
    # Build PatchAlign3D model + projection + CLIP text tower
    # ------------------------------------------------------------------
    print(f"[loading PatchAlign3D model from {args.ckpt}...]")
    cfg = EasyDict(
        trans_dim=384, depth=12, drop_path_rate=0.1, cls_dim=50, num_heads=6,
        group_size=args.group_size, num_group=args.num_group,
        encoder_dims=256, color=False, num_classes=16,
    )
    model = point_transformer.get_model(cfg).to(device)

    print(f"[loading OpenCLIP {args.clip_model} / {args.clip_pretrained}...]")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_dim = int(clip_model.text_projection.shape[1]) \
        if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None \
        else 1280  # ViT-bigG default

    proj = PatchToTextProj(in_dim=cfg.trans_dim, out_dim=text_dim).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"], strict=False)

    model.eval()
    proj.eval()
    clip_model.eval()

    experiment_name = args.experiment_name
    os.makedirs(experiment_name, exist_ok=True)
    log_filepath = f"{experiment_name}/localization_metrics_patchalign3d.log"
    if not os.path.exists(log_filepath):
        open(log_filepath, "w").close()

    all_metrics = []

    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        cad_folder_path = batch["image_paths"][0][0].split("/mesh_views_corrected/")[0]

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
        if args.prompt_mode == "question":
            queries = [chosen_ques_for_inference]
        elif args.prompt_mode == "feature":
            queries = [feature]
        else:  # feature_question
            queries = [chosen_ques_for_inference, feature]
        if args.negative_queries:
            queries = queries + list(args.negative_queries)

        mesh = trimesh.load(mesh_path, process=True)
        mesh.vertices, mesh.faces = stitch_mesh_topology(mesh.vertices, mesh.faces)

        gt_features = compute_gt_features(
            cad_file_path, mesh.vertices, mesh.faces, feature, feature_idx,
            timeout_s=args.gt_timeout,
        )
        if gt_features is None:
            continue

        xyz_np, face_idx_per_point = sample_pointcloud_from_mesh(
            mesh, n_points=args.n_points,
        )
        sample_save_dir = os.path.join(
            experiment_name, f"{cad_file_name}_{feature}_{feature_idx}",
        )
        os.makedirs(sample_save_dir, exist_ok=True)

        # ----- timed PatchAlign3D inference -----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        try:
            pred_label, target_sim = patchalign3d_predict_per_point(
                model, proj, clip_model, tokenizer,
                xyz_np, queries, args, device,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[patchalign3d error] {cad_file_name}: {e}")
            continue
        network_time = time.time() - t0

        # Selection
        loc_t0 = time.time()
        if args.selection_mode == "argmax":
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
        else:
            pred_entities = faces_to_edges(mesh.faces, pred_face_set)
        localization_time = time.time() - loc_t0
        # ----- end timed -----

        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) \
            if torch.cuda.is_available() else 0.0

        metrics = return_localization_metrics(
            mesh_path, mesh.vertices, mesh.faces,
            pred_entities, gt_features, feature, feature_idx,
        )
        metrics["network_time"] = network_time
        metrics["localization_time"] = localization_time
        metrics["inference_time"] = network_time
        metrics["peak_vram_mb"] = peak_vram_mb
        metrics["cad_file"] = cad_file_name
        all_metrics.append(metrics)

        with open(log_filepath, "a") as f:
            f.write(f"{metrics}\n")

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
        print('Lets see!!')

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    if all_metrics:
        avg = {"edge": {"iou": 0, "precision": 0, "recall": 0, "F1": 0},
               "face": {"iou": 0, "precision": 0, "recall": 0, "F1": 0}}
        counts = {"edge": 0, "face": 0}
        avg_inf = avg_loc = avg_vram = 0.0
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
