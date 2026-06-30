import os
# OpenRouter key is read from the environment (export OPENROUTER_API_KEY=...).
# Localization evaluation does not require it; only optional LLM-captioning paths do.
apikey = os.environ.get('OPENROUTER_API_KEY', '')
import ast
import trimesh
import torch
import numpy as np
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPImageProcessor, AutoTokenizer
from PIL import Image
from torchvision import transforms
from model.part_views import *
from model.LISA import LISAForCausalLM
from cad_utils import *
from eval_utils import *
from utils.dataset_ import *
import argparse
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from tqdm import tqdm
import time
import gc
import multiprocessing as mp

# ---------------------------------------------------------------------------
# Path resolution (repo-relative, override-able).
#   ROOT      -> the repo checkout (this file's directory): code + configs live here.
#   DATA_ROOT -> where large, git-ignored artefacts live (base LISA-7B / SAM / CLIP
#                weights, the runs/CAD_LISA checkpoint, view-selector .pt files,
#                and the CAD dataset). Defaults to ROOT; set MVGEL_ROOT to point at
#                a separate weights/data location without editing the code.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)


def _gt_worker(q, cad_file_path, vertices, faces, feature, feature_idx):
    """Worker process: compute GT features and put result on the queue."""
    try:
        res = cad_entity_to_mesh_faces(
            cad_file_path, vertices, faces,
            entity_type=feature, entity_index=feature_idx,
        )
        q.put(("ok", res))
    except Exception as e:
        q.put(("err", repr(e)))


def compute_gt_features_with_timeout(cad_file_path, vertices, faces, feature, feature_idx, timeout=20):
    """Run cad_entity_to_mesh_faces in a separate process and hard-kill it on timeout.

    Returns (status, payload):
        ('ok', gt_features) on success
        ('timeout', None)   if the worker exceeded `timeout` seconds (it is killed)
        ('error', repr)     if the worker raised
        ('empty', None)     if the worker died without producing a result
    """
    ctx = mp.get_context("spawn")  # avoid forking CUDA/OCCT state
    q = ctx.Queue()
    p = ctx.Process(
        target=_gt_worker,
        args=(q, cad_file_path, vertices, faces, feature, feature_idx),
    )
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
            p.join()
        return "timeout", None

    if q.empty():
        return "empty", None

    status, payload = q.get()
    if status != "ok":
        return "error", payload
    return "ok", payload


def merge_lora_into_state_dict(state_dict, alpha=16, r=8):
    new_state_dict = {}
    lora_groups = {}

    for key, val in state_dict.items():
        if "vision_tower" in key:
            continue
        
        key_ = key.replace("base_model.model.", "")
        if "lora_A.default.weight" in key:
            prefix = key_.replace(".lora_A.default.weight", ".weight")
            lora_groups.setdefault(prefix, {})["A"] = val

        elif "lora_B.default.weight" in key:
            prefix = key_.replace(".lora_B.default.weight", ".weight")
            lora_groups.setdefault(prefix, {})["B"] = val
        else:
            new_state_dict[key_] = val

    scaling = alpha / r
    for prefix, group in lora_groups.items():
        base_key_clean = prefix.replace("base_model.model.", "")

        if base_key_clean not in new_state_dict.keys():
            continue

        W = new_state_dict[base_key_clean]
        A = group["A"]
        B = group["B"]

        delta = (torch.matmul(B.to(W.dtype), A.to(W.dtype))) * scaling

        new_state_dict[base_key_clean] = W + delta

    return new_state_dict


if __name__ == '__main__':
    
    device = "cuda:0"

    parser = argparse.ArgumentParser(description="MVGEL training")
    parser.add_argument("--view_selector_model_path", default=os.path.join(DATA_ROOT, "best_model_view_ranker_cliplora_film.pt"), type=str)
    parser.add_argument("--LISA_model_path", default=os.path.join(DATA_ROOT, "runs/CAD_LISA/global_step5076"), type=str)
    parser.add_argument('--num_top_views', default=1, type=int)
    parser.add_argument('--view_nms', default=0, type=int)
    parser.add_argument('--val_dataset_log', default=None, type=str,
                        help='Path to a custom validation dataset log to evaluate on '
                             '(e.g. val_dataset.log). May be a '
                             'bare filename (resolved against the LISA repo root) or an '
                             'absolute path. If omitted, the default val split '
                             '(val_dataset.log) is used.')
    parser.add_argument('--start_idx', default=0, type=int,
                        help='Skip the first N samples of the val dataset (resume support).')
    parser.add_argument('--entity_allowlist', default=None, type=str,
                        help='Optional path to a text file with one '
                             '"cad_name,feature,feature_idx" key per line (feature is '
                             "'edge'/'face'). When set, evaluation is restricted to "
                             'exactly those entities (single-pass clean re-run of a '
                             'specific entity subset). Bare filename is resolved against '
                             'the LISA repo root.')
    parser.add_argument('--caption_variant', default='corrected', type=str,
                        choices=['corrected', 'original'],
                        help="Which per-CAD caption logs to evaluate on: 'corrected' -> "
                             "views_and_ques_{edge,face}_augmented_corrected.log (viewpoint-"
                             "fixed, default); 'original' -> views_and_ques_{edge,face}_"
                             "augmented_.log (old captions). For 'original' every view "
                             "azimuth is converted to (360 - az) %% 360 so the views that "
                             "are actually loaded/selected match the images on disk.")
    parser.add_argument('--save_vis', type=int, default=0,
                        help='If 1, render and save the PyVista 3D mesh figures of the '
                             'final combined Top-K entity predictions (via '
                             'visualize_entity_predictions) once per CAD model. This is '
                             'slow and not needed for metric runs, so it is OFF by '
                             'default; the persisted localization metrics do not depend '
                             'on these figures.')
    parser.add_argument('--render_target_masks', type=int, default=0,
                        help='If 1, render the GT target masks (slow PyVista render, '
                             '~100x the per-sample cost) to produce the diagnostic 2D '
                             'IoU printouts and inspect_masks overlays. The persisted '
                             'localization metrics DO NOT depend on these renders (GT '
                             'is CAD-derived, predictions are back-projected), so leave '
                             'this 0 for metric runs.')
    parser.add_argument('--gt_timeout', type=int, default=20,
                        help='Hard wall-clock timeout (seconds) for the gt_features '
                             'subprocess. On timeout the entity is skipped and NOT '
                             'logged. A few geometrically heavy meshes exceed the '
                             'default 20s and get dropped; raise this (e.g. 120) when '
                             'recovering those specific entities.')
    parser.add_argument('--gt_view_offset', type=int, default=0,
                        help='ORACLE (GT) mode only: rank offset into the ground-truth '
                             'ordered top-views list. 0 (default) starts at the best '
                             'view (standard GTviews oracle). 1 starts at the 2nd-best '
                             'view (skips the single best), 2 at the 3rd-best, etc. The '
                             'per-K selection becomes top_views_desc[offset:offset+K]. '
                             'Runs with offset>0 are written to a separate '
                             '_gtoffset{offset} experiment dir so they never overwrite '
                             'the standard oracle results. No effect for non-GT '
                             'selectors.')
    parser.add_argument('--gt_target_view', type=int, default=0,
                        help='ORACLE (GT) mode only: if 1, the oracle view is the '
                             'caption-marked TARGET view (part_dict["marked_image"], the '
                             'single view the entity was highlighted on and the caption '
                             'was authored against) instead of the geometric-visibility '
                             'ranking top_views_desc. This is the TRUE view oracle the '
                             'selectors aim to recover (the marked view is in their '
                             'candidate pool). Only one such view exists per entity, so '
                             'this is a Top-1 oracle and --gt_view_offset is ignored. '
                             'Written to a separate _gttarget experiment dir so it never '
                             'collides with the standard oracle. No effect for non-GT '
                             'selectors.')

    args = parser.parse_args()
    #args.view_selector_model_path = 'random'
    modality_fusion_variants = ['cross_attention', 'film', 'no_fusion', 'only_clip'] #'add', 'gated_add',

    # Resolve relative checkpoint paths so bare filenames (e.g.
    # "best_model_view_ranker_cliplora_film.pt") work regardless of the current
    # working directory. Sentinel values ('random'/'GT'/'vanilla') and absolute or
    # already-existing paths are left untouched.
    def _resolve_ckpt(p, sentinels=()):
        if p in sentinels or os.path.isabs(p) or os.path.exists(p):
            return p
        for base in (DATA_ROOT, ROOT):
            cand = os.path.join(base, p)
            if os.path.exists(cand):
                return cand
        return p

    args.view_selector_model_path = _resolve_ckpt(args.view_selector_model_path, sentinels=('random', 'GT'))
    args.LISA_model_path = _resolve_ckpt(args.LISA_model_path, sentinels=('vanilla',))

    # Optional entity-level allowlist: a text file with one
    # "cad_name,feature,feature_idx" key per line (feature is 'edge'/'face').
    # When provided, evaluation is restricted to exactly those entities, so a clean
    # single-pass re-run can target a handful of entities instead of every entity in
    # their CAD folders. Resolved against the repo root if not absolute.
    entity_allowlist = None
    if args.entity_allowlist:
        allow_path = args.entity_allowlist
        if not os.path.isabs(allow_path):
            allow_path = os.path.join(ROOT, allow_path)
        entity_allowlist = set()
        with open(allow_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                cad, feat, fidx = line.split(',')
                entity_allowlist.add((str(cad).strip(), feat.strip(), int(fidx)))
        print(f"[entity_allowlist] loaded {len(entity_allowlist)} keys from {allow_path}")

    val_dataset_view_selector = CAD_ViewRank_Dataset(
        clip_image_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16"),
        split="val",
        dataset_log_path=args.val_dataset_log,
        caption_variant=args.caption_variant,
        entity_allowlist=entity_allowlist)

    # NOTE: the resume Subset + DataLoader are built later, AFTER experiment_name is
    # known, so we can auto-resume from the per-experiment progress file on restart.

    # 'GT' anywhere in --view_selector_model_path switches to ORACLE mode: instead of a
    # learned selector or random views, use the ground-truth top views stored in each
    # sample's annotation (part_dict["top_views_desc"]). This produces the oracle
    # upper-bound that every view-selector variant is compared against.
    use_gt_views = 'GT' in args.view_selector_model_path

    if use_gt_views:
        view_selector_name = 'GT'
        fusion_type = 'GT'
    else:
        try:
            view_selector_name = args.view_selector_model_path.split('best_model_view_ranker_')[1].split('.')[0]
        except: 
            view_selector_name = args.view_selector_model_path
            fusion_type = 'random'

        if 'cliplora' in args.view_selector_model_path:
            view_selector_type = os.path.basename(args.view_selector_model_path).split('.')[0].split('cliplora_')[-1]
            if view_selector_type in modality_fusion_variants:
                fusion_type = view_selector_type
                fusion_type = None if fusion_type=='None' else fusion_type
                view_selector_model = LoraCLIPViewSelector_Ablation(fusion_type=fusion_type).to(device)
            
                checkpoint = torch.load(args.view_selector_model_path, map_location=device)
        
            view_selector_model.load_state_dict(checkpoint['model_state_dict'])
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        
    topk = args.num_top_views

    # views_dict = {}
    # if view_selector_type in ['random', 'no_fusion', 'film', 'cross_attention', 'only_clip', 'film_no_clip', 'cross_attention_no_clip', 'add', 'gated_add']:
    #     with open(f'GeLoM_topviews_{view_selector_type}.log', 'r') as fread:
    #         print(f'Successfuly reading GeLoM_topviews_{view_selector_type}.log')
    #         for line in fread:
    #             dict_ = ast.literal_eval(line.strip())
    #             geometry_name = dict_['top_pred_views_nms45'][0].split('/')[-3]
    #             feature, feature_idx = dict_['feature'], dict_['feature_idx']
    #             views_dict[f'{geometry_name}_{feature}_{feature_idx}'] = dict_['top_pred_views_nms45']
        
    # ------------------------
    # Load model
    # ------------------------
    lisa_model_path = os.path.join(DATA_ROOT, 'model/load_files&weights/LISA-7B-v1-explanatory')
    segllm_tokenizer = AutoTokenizer.from_pretrained(
        lisa_model_path,
        use_fast=False
    )
    seg_token_idx = segllm_tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    model_args = {
        "out_dim": 256,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": os.path.join(DATA_ROOT, "model/segment_anything/load_files&weights/sam_vit_h_4b8939.pth"),
        "vision_tower": "openai/clip-vit-large-patch14",
        "use_mm_start_end": True,
    }

    model = LISAForCausalLM.from_pretrained(
        lisa_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **model_args
    )
    
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=torch.bfloat16)

    # -------------------------------------------------------------------------
    # Load the trained LISA weights from a DeepSpeed ZeRO checkpoint.
    #
    # The path given in --LISA_model_path decides everything:
    #   * If it points directly at a checkpoint tag dir (e.g.
    #     .../ckpt_model/global_step4653), that EXACT checkpoint is loaded.
    #   * If it points at a ckpt_model parent dir containing a `latest` file,
    #     the tag named in `latest` is loaded.
    #   * Otherwise (a non-existent path / random string like "Vanilla"), no
    #     checkpoint is loaded and the base LISA-7B weights are kept.
    #
    # `get_fp32_state_dict_from_zero_checkpoint` consolidates ALL per-rank shards
    # (here 4 GPUs) into a single fp32 state dict, so the number of GPUs that
    # produced the checkpoint is irrelevant.
    # -------------------------------------------------------------------------
    LLM_name = 'Vanilla_LISA'
    ds_ckpt_dir = args.LISA_model_path.rstrip('/')
    ckpt_tag = None

    # --LISA_model_path pointed directly at a global_stepXXXX directory.
    if os.path.basename(ds_ckpt_dir).startswith('global_step') and os.path.isdir(ds_ckpt_dir):
        ckpt_tag = os.path.basename(ds_ckpt_dir)
        ds_ckpt_dir = os.path.dirname(ds_ckpt_dir)
        is_zero_ckpt = True
    else:
        # A ckpt_model parent dir with a `latest` pointer.
        is_zero_ckpt = os.path.exists(os.path.join(ds_ckpt_dir, 'latest'))

    if is_zero_ckpt:
        print(f"[load] Loading DeepSpeed checkpoint dir={ds_ckpt_dir} "
              f"tag={ckpt_tag or 'latest'}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(ds_ckpt_dir, tag=ckpt_tag)
        compatible_state_dict = merge_lora_into_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(compatible_state_dict, strict=False)
        print(f"[load] loaded tag={ckpt_tag or 'latest'} | "
              f"missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        LLM_name = 'CAD_LISA' if 'CAD_LISA' in args.LISA_model_path else 'Repro20_LISA'
    else:
        print(f"[load] No DeepSpeed checkpoint found at {args.LISA_model_path}; "
              f"using base Vanilla_LISA weights.")
    
    experiment_name = f'MVGEL_{LLM_name}_{view_selector_name}_vnms{args.view_nms}'
    # When evaluating on a non-default validation log, suffix the experiment dir with
    # the log name so its progress/metrics don't collide with the default val run.
    if args.val_dataset_log is not None:
        val_log_tag = os.path.splitext(os.path.basename(args.val_dataset_log))[0]
        experiment_name = f'{experiment_name}_{val_log_tag}'
    # Keep 'original'-caption runs in a separate dir so they don't overwrite the
    # default corrected-caption metrics/progress.
    if args.caption_variant == 'original':
        experiment_name = f'{experiment_name}_origcaps'
    # Oracle rank-offset ablation (2nd-best view, etc.): isolate in its own dir so it
    # never collides with the standard GTviews (best-view) oracle results.
    if use_gt_views and args.gt_view_offset > 0:
        experiment_name = f'{experiment_name}_gtoffset{args.gt_view_offset}'
    # Marked-target-view oracle (the caption's true target view): isolate in its own
    # dir so it never collides with the standard / offset oracle results.
    if use_gt_views and args.gt_target_view:
        experiment_name = f'{experiment_name}_gttarget'
    os.makedirs(experiment_name, exist_ok=True)
    
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Resume support: a small progress file records the highest dataset index
    # that has been fully processed. On (re)start we skip everything up to and
    # including it, so an auto-restart after an OOM/crash continues cleanly
    # instead of re-doing (and duplicating) already-logged samples.
    # -------------------------------------------------------------------------
    progress_file = os.path.join(experiment_name, 'progress.txt')
    resume_idx = args.start_idx
    if resume_idx == 0 and os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as pf:
                last_done = int(pf.read().strip())
            resume_idx = last_done + 1
            print(f"[resume] Found progress file -> resuming from batch_idx={resume_idx}.")
        except Exception as e:
            print(f"[resume] Could not parse progress file ({e}); starting from 0.")
            resume_idx = 0
    args.start_idx = resume_idx

    if resume_idx >= len(val_dataset_view_selector):
        print(f"[resume] start_idx={resume_idx} >= dataset size "
              f"{len(val_dataset_view_selector)}; nothing left to do.")

    if args.start_idx > 0:
        from torch.utils.data import Subset
        print(f"[resume] Skipping first {args.start_idx} samples (starting from batch_idx={args.start_idx}).")
        val_dataset_iter = Subset(
            val_dataset_view_selector,
            range(args.start_idx, len(val_dataset_view_selector)),
        )
    else:
        val_dataset_iter = val_dataset_view_selector

    val_loader = torch.utils.data.DataLoader(
        val_dataset_iter,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=views_collate_fn,
        pin_memory=True,
        persistent_workers=False
    )

    # Built ONCE (previously rebuilt every iteration, which re-hit the HF hub and
    # churned host memory on every sample).
    clip_img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    # -------------------------------------------------------------------------
    # Initialize dictionary to hold metrics for EACH top-K step
    # lc_metrics_list_per_k[k] will hold a list of results for k views
    # -------------------------------------------------------------------------
    lc_metrics_list_per_k = {k:[] for k in range(1, topk + 1)}
    lc_metrics_net = []  # one entry per sample (net top-K result)

    def mark_done(idx):
        """Persist the highest dataset index that has been fully handled so a
        restart can resume right after it. Called for completed AND skipped
        samples (skipped ones would otherwise be re-attempted forever)."""
        try:
            with open(progress_file, 'w') as pf:
                pf.write(str(idx))
        except Exception as e:
            print(f"[resume] WARNING: could not write progress file: {e}")

    def free_sample_memory():
        """Drop per-sample objects and run a GC + CUDA cache flush so host and
        device memory stay flat across the full run."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for batch_idx, batch in tqdm(
        enumerate(val_loader, start=args.start_idx),
        total=len(val_dataset_view_selector),
        initial=args.start_idx,
    ):
        #if batch_idx < 625: continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        cad_folder_path = batch["image_paths"][0][0].split("/mesh_views_corrected/")[0] 
        for file in os.listdir(cad_folder_path):
            if file.endswith('.step'):
                cad_file_path = os.path.join(cad_folder_path, file)
                break
        cad_file_name = cad_file_path.split('/')[-2]
        img_paths = batch["image_paths"][0]
        images_tensors = batch["images"][0]

        chosen_ques = batch["question"][0]
        #if feature == 'face': continue
        caption_suffix = '_augmented_corrected' if args.caption_variant == 'corrected' else '_augmented_'
        # Resolve the entity by EXACT question match. The feature cannot be inferred from
        # the question text ('edge' in chosen_ques) because face captions frequently
        # describe bounding edges and vice-versa; that misclassifies the entity and
        # silently falls through to the last line of the wrong caption log. Search both
        # logs and take the authoritative feature from the matched entry's marked_image.
        feature, part_dict = None, None
        for feat_try in ('edge', 'face'):
            log_path = f"{cad_folder_path}/views_and_ques_{feat_try}{caption_suffix}.log"
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r', encoding="utf-8") as f:
                for line in f:
                    cand_dict = ast.literal_eval(line.strip())
                    if cand_dict['question'] == chosen_ques:
                        part_dict = cand_dict
                        feature = feat_try
                        break
            if part_dict is not None:
                break
        assert part_dict is not None, (
            f"Question not found in either caption log for {cad_file_name}: {chosen_ques!r}")

        chosen_ans = part_dict['answer']
        chosen_marked_view_path = part_dict['marked_image']
        feature_ = os.path.basename(chosen_marked_view_path).split('_marked_')[1].split('[')[0]
        assert feature == feature_, f"Feature in question {feature} does not match feature in marked view {feature_}"
        
        feature_idx = int(os.path.basename(chosen_marked_view_path).split('[')[1].split(']')[0])
        print(f"Randomly selected question for {cad_file_name}: {chosen_ques}")

        modality_fusion_variants = ['cross_attention', 'film', 'no_fusion', 'only_clip']
        
        view_selector_time = 0.0
        vs_start = time.time()
        if use_gt_views:
            # ORACLE: use the ground-truth top views from the annotation. Map each GT
            # marked-view description to its plain rendered view in the same corrected
            # views directory used by the other selectors, so the input domain matches.
            views_dir = os.path.dirname(img_paths[0])
            selected_views = []
            if args.gt_target_view:
                # TRUE view oracle: the single caption-marked TARGET view the entity was
                # highlighted on (and the caption was authored against). Only one such
                # view exists, so this is inherently Top-1 (offset is ignored).
                gt_views_ranked = [part_dict["marked_image"]]
            else:
                # Rank offset: 0 -> best view (standard oracle); 1 -> 2nd-best (skip best); etc.
                gt_views_ranked = part_dict["top_views_desc"][args.gt_view_offset:]
            for gt_view in gt_views_ranked[:topk]:
                el_g, az_g = extract_el_az_from_view_desc(gt_view)
                # 'original' captions use the flipped azimuth convention; convert to the
                # on-disk (corrected) azimuth so the loaded view matches the image and the
                # azimuth used for back-projection downstream.
                if args.caption_variant == 'original':
                    az_g = (360 - az_g) % 360
                selected_views.append(f"{views_dir}/view_e{el_g}_a{az_g}.png")
        elif fusion_type in modality_fusion_variants:

            tokenized = clip_tokenizer(
                chosen_ques,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            view_selector_model.eval()
            
            with torch.no_grad():
                scores = view_selector_model(images_tensors.unsqueeze(0).to(device), tokenized)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            #idxs = torch.argsort(scores.squeeze(0), descending=True).cpu().numpy().tolist()
            if args.view_nms>0:
                #selected_views_no_view_nms = [img for img, _ in selected_no_view_nms][:topk]
                selected_views = viewNMS(
                    img_paths,
                    scores.squeeze(0).cpu().numpy(), angle_threshold=45)
                selected_views = selected_views[:topk]
            else:
                selected_no_view_nms = sorted(zip(img_paths, scores.squeeze(0).cpu().numpy()), key=lambda x: x[1], reverse=True)
                selected_views = [img for img, _ in selected_no_view_nms][:topk]
            
        else:
            selected_views = np.random.choice(img_paths, size=topk, replace=False).tolist()
        view_selector_time = time.time() - vs_start
        ### Use saved views
        # if len(views_dict.keys())>0:
        #     print(f'Selecting views {topk} from {cad_file_name}_{feature}_{feature_idx}')
        #     selected_views = views_dict[f'{cad_file_name}_{feature}_{feature_idx}'][:topk]
        # ### Ground truth views
        # else:
        #     print('Selecting ground truth views')
        #     selected_views_basenames = [f"{os.path.basename(selected_view_path).split('_marked')[0]}.png" for selected_view_path in part_dict["top_views_desc"]][:topk]
        #     selected_views_dirname = os.path.dirname(part_dict["top_views_desc"][0])
        #     selected_views = [f'{selected_views_dirname}/mesh_views/{selected_views_basename}' for selected_views_basename in selected_views_basenames]
        # print(f"Selected {topk} views: {selected_views}")
        
        images_clip, images_sam, input_ids_list, resize_list, original_size_list, target_image_paths = [], [],[], [], [], []
        target_masks = []
        dict_batch =[]
        target_render_time = 0.0  # diagnostic only; NOT included in localization metrics
        for selected_view in selected_views:
            el,az = extract_el_az_from_view_desc(selected_view)
            image = Image.open(selected_view).convert("RGB")
            image_clip = clip_img_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  
            
            image = np.array(image)  
            image_sam = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            image_sam = (image_sam - pixel_mean) / pixel_std

            target_img_path = f"{cad_folder_path}/target_CDviews/view_e{el}_a{az}_marked_{feature}[{feature_idx}].png"

            _t_render = time.time()
            if args.render_target_masks:
                rendered_imgs = render_cad_views(
                    cad_file_path,
                    output_dir=f"{cad_folder_path}/target_CDviews",
                    n_azimuth=[az],
                    n_elevation=[el],
                    verbose=True,
                    use_cad=True,
                    highlight_edge_idxs=[feature_idx] if feature=='edge' else None,
                    highlight_face_idxs=[feature_idx] if feature=='face' else None,
                    save_img=False,
                )
                target_mask = extract_mask(rendered_imgs[target_img_path], f"{feature}s")
            else:
                # Skip the (slow) GT render. The localization metrics never use this
                # mask -- it only drives the diagnostic 2D IoU/overlays. We still need
                # a correctly-shaped placeholder because its shape sets the predicted
                # mask resolution (label/resize). The input view is already 1024x1024.
                target_mask = np.zeros(image.shape[:2], dtype=np.float32)
            target_render_time += time.time() - _t_render
                
            target_masks.append(torch.from_numpy(target_mask))
            
            for filename in os.listdir(cad_folder_path):
                if filename.endswith('.obj'):
                    mesh_path = os.path.join(cad_folder_path, filename)
                    break
            mesh = trimesh.load(mesh_path, process=True)
            mesh.vertices, mesh.faces = stitch_mesh_topology(mesh.vertices, mesh.faces)
            assert len(mesh.vertices) == len(np.unique(mesh.vertices, axis=0)), "Vertices not unique, mesh not stitched"
            
            dict_batch.append({
                "input_img_path": selected_view,
                "input_img": image_sam,
                "input_img_clip": image_clip,
                "question": chosen_ques,
                "answer": chosen_ans,
                "target_img_path": target_img_path,
                "target_mask": torch.from_numpy(target_mask),
                "label": torch.ones(target_mask.shape) * 255, 
                "resize": target_mask.shape,
                "feature": feature,
                "top_views": part_dict['top_views_desc'],
                "inference": True,
            })

        collated_batches =[]
        for dict_batch_ in dict_batch:
            collated_batch = cad_collate_fn([dict_batch_], tokenizer=segllm_tokenizer, use_mm_start_end=True, local_rank=0)
            collated_batches.append(collated_batch)

        pred_masks, gt_masks = [],[]
        inference_time = 0.0
        for i, sample in enumerate(collated_batches):
            sample = dict_to_cuda(sample)
            sample["images"] = sample["images"].to(dtype=torch.bfloat16)
            sample["images_clip"] = sample["images_clip"].to(dtype=torch.bfloat16)

            forward_start = time.time()
            with torch.no_grad():
                output_dict = model(**sample)
            pred_i = (output_dict["pred_masks"][0] > 0).int()
            mask_i = output_dict["gt_masks"][0].int()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time += time.time() - forward_start

            pred_masks.append(pred_i)
            gt_masks.append(mask_i)
            if args.render_target_masks:
                inspect_masks(pred_i.squeeze(0), mask_i, selected_views[i], experiment_name, part_dict["marked_image"], sample['conversation_list'][0], append_str=f"_{feature}[{feature_idx}]_view_{i}_")

        # Peak VRAM (bytes -> MB) consumed during view-selector + LISA inference for this sample
        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        
        # -----------------------------------------------------------------------------------------
        # 1. Compute GT Features ONCE per mesh (with HARD Timeout via terminable subprocess)
        # -----------------------------------------------------------------------------------------
        start_time = time.time()
        status, payload = compute_gt_features_with_timeout(
            cad_file_path, mesh.vertices, mesh.faces, feature, feature_idx, timeout=args.gt_timeout,
        )
        elapsed_time = time.time() - start_time

        if status == "timeout":
            print(f"Timeout: gt_features took > {args.gt_timeout} seconds for {cad_file_name} (killed after {elapsed_time:.2f}s). Skipping to next.")
            mark_done(batch_idx)
            free_sample_memory()
            continue
        if status == "error":
            print(f"Error computing gt_features for {cad_file_name}: {payload}. Skipping.")
            mark_done(batch_idx)
            free_sample_memory()
            continue
        if status == "empty":
            print(f"GT worker died without result for {cad_file_name}. Skipping.")
            mark_done(batch_idx)
            free_sample_memory()
            continue

        gt_features = payload
        print(f"Successfully loaded gt_features in {elapsed_time:.2f} seconds.")

        if gt_features is None:
            print(f"GT features for {cad_file_name} returned None. Skipping.")
            mark_done(batch_idx)
            free_sample_memory()
            continue

        # # -----------------------------------------------------------------------------------------
        # # 2. Simulate Top-K Study Incrementally (K = 1 to actual_max_views)
        # # -----------------------------------------------------------------------------------------
        # os.makedirs(experiment_name, exist_ok=True)
        # max_views_for_sample = len(selected_views)
        
        # cumulative_pred_features =[]
        # unique_pred_features =[]
        # localization_time_cum = 0.0
        
        # for k in range(1, max_views_for_sample + 1):
        #     view_idx = k - 1
        #     pred_mask = pred_masks[view_idx]
        #     selected_view = selected_views[view_idx]
            
        #     # Extract 3D features for THIS specific view (timed)
        #     loc_start = time.time()
        #     el, az = extract_el_az_from_view_desc(selected_view)
        #     new_features = return_mesh_entities_from_2D_masks(
        #         mesh.vertices, mesh.faces, pred_mask, el, az, feature
        #     )
            
        #     # Accumulate and Deduplicate
        #     cumulative_pred_features.extend(new_features)
            
        #     if feature == 'edge':    
        #         unique_pred_features = list({tuple(sorted(e)) for e in cumulative_pred_features})
        #     else:
        #         unique_pred_features = np.unique(np.array(cumulative_pred_features)).tolist()
        #     localization_time_cum += time.time() - loc_start

        #     # Compute localization metrics incrementally for this K
        #     localization_metrics = return_localization_metrics(
        #         mesh_path, mesh.vertices, mesh.faces, unique_pred_features, gt_features, feature, feature_idx
        #     )
        #     # Per-view inference time (forward pass for this k-th view) + cumulative localization time + view selector time
        #     per_view_inference_time = inference_time / max_views_for_sample
        #     localization_metrics['inference_time'] = view_selector_time + per_view_inference_time * k + localization_time_cum
        #     localization_metrics['peak_vram_mb'] = peak_vram_mb
            
        #     lc_metrics_list_per_k[k].append(localization_metrics) 

        #     # Write this specific sample's K-result to the respective log file
        #     log_filepath = f'{experiment_name}/localization_metrics_view_selector_{view_selector_name}_views_{k}.log'
        #     if not os.path.exists(log_filepath):
        #         with open(log_filepath, 'w') as f:
        #             f.write('')    
        #     with open(log_filepath, 'a') as f:
        #         f.write(f'{localization_metrics}\n')

        os.makedirs(experiment_name, exist_ok=True)
        max_views_for_sample = len(selected_views)
        
        cumulative_pred_features =[]
        unique_pred_features =[]
        localization_time_cum = 0.0

        # Per-view share of the already-measured total LISA forward time, so the reported
        # inference_time grows with the number of views actually used at each K.
        per_view_inference_time = inference_time / max_views_for_sample if max_views_for_sample else 0.0
        localization_metrics = None

        # =========================================================================
        # INCREMENTAL PER-K MODE: accumulate predictions one view at a time and score
        # the running union at EVERY K (1, 2, ..., max_views_for_sample). This persists
        # localization metrics for every intermediate top-K, not just the final one.
        # The final K (== max_views_for_sample) is also the "net top-K" result.
        # =========================================================================
        for k in range(1, max_views_for_sample + 1):
            view_idx = k - 1
            pred_mask = pred_masks[view_idx]
            selected_view = selected_views[view_idx]

            # Extract 3D features for THIS specific view (timed)
            loc_start = time.time()
            el, az = extract_el_az_from_view_desc(selected_view)
            # IMPORTANT: selected_view lives in mesh_views_corrected/, whose azimuth in the
            # FILENAME is a relabeled flip of the true render azimuth: the image stored under
            # label `aA` was physically rendered at (360 - A) % 360. The back-projection camera
            # model (return_mesh_entities_from_2D_masks) is identical to the render camera model
            # (cad_utils.render_cad_views), so it must be fed the TRUE render azimuth, otherwise
            # the camera is placed on the mirrored side (sin(A) vs sin(360-A)) and edges miss
            # entirely while faces only partially overlap. Un-flip here to restore the original
            # (mesh_views) convention used by the rebuttal pipeline.
            az = (360 - az) % 360
            new_features = return_mesh_entities_from_2D_masks(
                mesh.vertices, mesh.faces, pred_mask, el, az, feature
            )

            # Accumulate and deduplicate the running union of predicted entities
            cumulative_pred_features.extend(new_features)
            if feature == 'edge':
                unique_pred_features = list({tuple(sorted(e)) for e in cumulative_pred_features})
            else:
                unique_pred_features = np.unique(np.array(cumulative_pred_features)).tolist()
            localization_time_cum += time.time() - loc_start

            # Score the running union for this K
            localization_metrics = return_localization_metrics(
                mesh_path, mesh.vertices, mesh.faces, unique_pred_features, gt_features, feature, feature_idx
            )
            # NOTE: `inference_time` reports only network cost (view selector + LISA forwards).
            # Back-projection time is reported separately as `localization_time` because its
            # cost scales with predicted-mask area, which would unfairly make failed/empty
            # predictions appear faster than successful ones.
            localization_metrics['view_selector_time'] = view_selector_time
            localization_metrics['lisa_forward_time'] = per_view_inference_time * k
            localization_metrics['localization_time'] = localization_time_cum
            localization_metrics['inference_time'] = view_selector_time + per_view_inference_time * k
            localization_metrics['peak_vram_mb'] = peak_vram_mb
            localization_metrics['num_views'] = k
            lc_metrics_list_per_k[k].append(localization_metrics)

            # One log file per (view_selector, K) so every intermediate top-K is persisted.
            log_filepath_k = f'{experiment_name}/localization_metrics_view_selector_{view_selector_name}_views_{k}.log'
            if not os.path.exists(log_filepath_k):
                with open(log_filepath_k, 'w') as f:
                    f.write('')
            with open(log_filepath_k, 'a') as f:
                f.write(f'{localization_metrics}\n')

        print(
            f"[timing] sample={cad_file_name} "
            f"num_views={len(selected_views)} "
            f"view_selector={view_selector_time:.2f}s "
            f"lisa_forward={inference_time:.2f}s "
            f"target_render={target_render_time:.2f}s "
            f"localization={localization_time_cum:.2f}s"
        )

        # The final K (full top-K union) is the net result for this sample; it was already
        # computed, tagged, and logged (to ..._views_{max_views_for_sample}.log) inside the
        # incremental loop above. Record it once more in the net list for the run summary.
        if localization_metrics is not None:
            lc_metrics_net.append(localization_metrics)

        # Visualize only the final combined Top-K prediction once per CAD model
        if args.save_vis:
            view_paths = visualize_entity_predictions(mesh.vertices, mesh.faces, unique_pred_features, gt_features, feature, save_prefix=f'{experiment_name}/{view_selector_name}_views_{max_views_for_sample}_{cad_file_name}_{feature}_{feature_idx}')
        print('Lets see')

        # Sample fully handled: record progress and release per-sample memory so the
        # run stays flat in RAM/VRAM over the (multi-day) evaluation.
        mark_done(batch_idx)
        free_sample_memory()
    # -----------------------------------------------------------------------------------------
    # 3. Aggregate and Save Average Metrics over the whole run, per top-K step.
    #    Each top-K (K = 1 .. topk) is summarized into its own ..._views_{K}.log so the
    #    full Top-K curve is available, not just the final K.
    # -----------------------------------------------------------------------------------------
    for k in range(1, topk + 1):
        num_samples_evaluated = len(lc_metrics_list_per_k[k])
        if num_samples_evaluated == 0:
            continue

        avg_metrics = {
            'edge': {'iou': 0, 'precision': 0, 'recall': 0, 'F1': 0},
            'face': {'iou': 0, 'precision': 0, 'recall': 0, 'F1': 0},
        }
        counts = {'edge': 0, 'face': 0}
        avg_inference_time = 0.0
        avg_localization_time = 0.0
        avg_peak_vram_mb = 0.0

        for lc_dict in lc_metrics_list_per_k[k]:
            f_type = lc_dict['feature']
            counts[f_type] += 1
            for metric_key in avg_metrics[f_type].keys():
                avg_metrics[f_type][metric_key] += lc_dict[metric_key]
            avg_inference_time += lc_dict.get('inference_time', 0)
            avg_localization_time += lc_dict.get('localization_time', 0)
            avg_peak_vram_mb += lc_dict.get('peak_vram_mb', 0)

        for f_type in avg_metrics:
            if counts[f_type]:
                for metric_key in avg_metrics[f_type]:
                    avg_metrics[f_type][metric_key] /= counts[f_type]
        avg_inference_time /= num_samples_evaluated
        avg_localization_time /= num_samples_evaluated
        avg_peak_vram_mb /= num_samples_evaluated

        log_filepath = f'{experiment_name}/localization_metrics_view_selector_{view_selector_name}_views_{k}.log'
        with open(log_filepath, 'a') as f:
            f.write(f'\n====================================\n')
            f.write(f'Avg Metrics for Top-{k} Views (Samples evaluated: {num_samples_evaluated}, edge={counts["edge"]}, face={counts["face"]}): {avg_metrics}\n')
            f.write(f'Avg Inference Time (view_selector + LISA forward): {avg_inference_time:.2f} seconds\n')
            f.write(f'Avg Localization Time (back-projection only): {avg_localization_time:.2f} seconds\n')
            f.write(f'Avg Peak VRAM: {avg_peak_vram_mb:.2f} MB\n')
            f.write(f'====================================\n')

        print(f'Lets See')