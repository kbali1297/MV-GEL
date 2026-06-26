# Localization-metric provenance (corrected 1535-query test set)

Test set: 1535 entities (655 faces, 880 edges), 218 meshes. Allowlist: `val_dataset_1535_entities.txt`.

Each config is the combination (deduped) of the listed log files; sharded runs and their completion/recovery re-runs are merged with the precedence noted in `aggregate_all_tables.py`.

## `baseline::PartSLIP (Default)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** PartSLIP pretrained (GLIP+SAM), conf preset
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/162-PartSLIP-Low-Shot-Part-Segmentation-for-3D-Point-Clouds-via-Pretrained-Image-Language-Models/GeLoM_PartSLIP_default_1535_shard*/localization_metrics_partslip.log`

## `baseline::PartSLIP (Top-PCD)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** PartSLIP pretrained, topk_pct preset (face10/edge2)
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/162-PartSLIP-Low-Shot-Part-Segmentation-for-3D-Point-Clouds-via-Pretrained-Image-Language-Models/GeLoM_PartSLIP_toppcd_1535_shard*/localization_metrics_partslip.log`

## `baseline::Find3D (Default)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** Find3D pretrained, argmax preset
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/Find3D/GeLoM_Find3D_default_1535_shard*/localization_metrics_find3d.log`
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/Find3D/GeLoM_Find3D_default_1535_recover/localization_metrics_find3d.log`

## `baseline::Find3D (Top-PCD)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** Find3D pretrained, topk_pct preset (face10/edge2)
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/Find3D/GeLoM_Find3D_toppcd_1535_shard*/localization_metrics_find3d.log`
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/Find3D/GeLoM_Find3D_toppcd_1535_recover/localization_metrics_find3d.log`

## `baseline::PatchAlign3D (Default)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** PatchAlign3D pretrained, argmax preset
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/PatchAlign3D/GeLoM_PatchAlign3D_default_1535_shard*/localization_metrics_patchalign3d.log`
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/PatchAlign3D/GeLoM_PatchAlign3D_default_1535_recover/localization_metrics_patchalign3d.log`

## `baseline::PatchAlign3D (Top-PCD)`  (655/655 face, 880/880 edge)
- **checkpoint(s):** PatchAlign3D pretrained, topk_pct preset (face10/edge2)
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/PatchAlign3D/GeLoM_PatchAlign3D_toppcd_1535_shard*/localization_metrics_patchalign3d.log`
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/PatchAlign3D/GeLoM_PatchAlign3D_toppcd_1535_recover/localization_metrics_patchalign3d.log`

## `mvgel::CAD_LISA::cliplora_cross_attention::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::cliplora_cross_attention::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::cliplora_cross_attention::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::CAD_LISA::cliplora_film::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::cliplora_film::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::cliplora_film::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::CAD_LISA::cliplora_no_fusion::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::cliplora_no_fusion::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::cliplora_no_fusion::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::CAD_LISA::cliplora_only_clip::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::cliplora_only_clip::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::cliplora_only_clip::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_CAD_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::CAD_LISA::random::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: random views)
- **logs:**
    - `MVGEL_CAD_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::random::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: random views)
- **logs:**
    - `MVGEL_CAD_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::random::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: random views)
- **logs:**
    - `MVGEL_CAD_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::CAD_LISA::GT::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_CAD_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::CAD_LISA::GT::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_CAD_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::CAD_LISA::GT::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_CAD_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::cliplora_cross_attention::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::cliplora_cross_attention::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::cliplora_cross_attention::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_cross_attention.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_cross_attention_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::cliplora_film::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::cliplora_film::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::cliplora_film::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_film.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_film_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::cliplora_no_fusion::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::cliplora_no_fusion::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::cliplora_no_fusion::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_no_fusion.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_no_fusion_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::cliplora_only_clip::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::cliplora_only_clip::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::cliplora_only_clip::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + best_model_view_ranker_cliplora_only_clip.pt
- **logs:**
    - `MVGEL_Vanilla_LISA_cliplora_only_clip_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::random::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: random views)
- **logs:**
    - `MVGEL_Vanilla_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::random::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: random views)
- **logs:**
    - `MVGEL_Vanilla_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::random::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: random views)
- **logs:**
    - `MVGEL_Vanilla_LISA_random_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `mvgel::Vanilla_LISA::GT::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_Vanilla_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_1.log`

## `mvgel::Vanilla_LISA::GT::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_Vanilla_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_3.log`

## `mvgel::Vanilla_LISA::GT::top5`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + (sentinel: geometric-visibility / marked-target oracle)
- **logs:**
    - `MVGEL_Vanilla_LISA_GT_vnms0_val_dataset_{total,dupes,missing24}/...views_5.log`

## `gttarget::CAD_LISA::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + marked-target oracle
- **logs:**
    - `MVGEL_CAD_LISA_GT_vnms0_val_dataset_1535_shard*_gttarget/...views_1.log`

## `gttarget::Vanilla_LISA::top1`  (655/655 face, 880/880 edge)
- **checkpoint(s):** vanilla (base LISA-7B) + marked-target oracle
- **logs:**
    - `MVGEL_Vanilla_LISA_GT_vnms0_val_dataset_1535_shard*_gttarget/...views_1.log`

## `vnms1::CAD_LISA::film::top3`  (655/655 face, 880/880 edge)
- **checkpoint(s):** runs/CAD_LISA_repro20/ckpt_model/global_step5076 + best_model_view_ranker_cliplora_film.pt (+ViewNMS)
- **logs:**
    - `/data/1bali/Other_LLM_projects/ECCV_2026/LISA/MVGEL_CAD_LISA_cliplora_film_vnms1_val_dataset_1535_shard*/localization_metrics_view_selector_cliplora_film_views_3.log`

