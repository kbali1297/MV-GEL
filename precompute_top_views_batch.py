import os
import ast
import torch
import numpy as np
from cad_utils import viewNMS
from model.part_views import LoraCLIPViewSelector_Ablation
from utils.dataset_ import CAD_ViewRank_Dataset, views_collate_fn
from transformers import CLIPTokenizer, CLIPImageProcessor
from tqdm import tqdm
import argparse

if __name__ == '__main__':

    device = "cuda:0"
    parser = argparse.ArgumentParser(description="GLOviews training")
    parser.add_argument("--view_selector_model_path", default="/data/1bali/Other_LLM_projects/ECCV_2026/LISA/best_model_view_ranker_cliplora_film.pt", type=str)
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for validation loader")
    parser.add_argument("--valset_name", default="val", type=str)
    args = parser.parse_args()

    model_ckpt = args.view_selector_model_path
    topk = 10
    fusion_type = model_ckpt.split('cliplora_')[1].split('.pt')[0]
    
    view_selector_model = LoraCLIPViewSelector_Ablation(fusion_type=fusion_type).to(device)

    val_dataset_view_selector = CAD_ViewRank_Dataset(
        clip_image_processor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16"),
        split=args.valset_name)

    val_loader = torch.utils.data.DataLoader(
        val_dataset_view_selector,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.batch_size, # Optional: scale num_workers with batch size
        collate_fn=views_collate_fn,
        pin_memory=True,
        persistent_workers=False
    )
    
    # try:
    checkpoint = torch.load(f"/data/1bali/Other_LLM_projects/ECCV_2026/LISA/model/load_files&weights/best_model_view_ranker_cliplora_{fusion_type}.pt", map_location=device)
    view_selector_model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded: {fusion_type}')
    # except:
    #     print(f'Model Could not be loaded: {fusion_type}')
        
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    fpath = f'GeLoM_topviews_{fusion_type}.log'
    with open(fpath, 'w') as fout:
        fout.write(f'')

    view_selector_model.eval()
    modality_fusion_variants = ['cross_attention', 'film', 'no_fusion', 'only_clip', 'gated_add', 'add', 'cross_attention_no_clip', 'film_no_clip']
    
    global_sample_idx = 0
    
    for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        questions = batch["question"]
        image_paths_list = batch["image_paths"]
        
        batch_size_current = len(questions)
        
        # Adjust skip logic from batches to actual samples
        # if global_sample_idx + batch_size_current <= 1800:
        #     global_sample_idx += batch_size_current
        #     continue
            
        start_i = 0
        # if global_sample_idx < 1800:
        #     start_i = 1800 - global_sample_idx
        
        global_sample_idx += batch_size_current

        # ------------------------------------------------------------------
        # Batched Forward Pass
        # ------------------------------------------------------------------
        if fusion_type in modality_fusion_variants:
            tokenized = clip_tokenizer(
                questions,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Handle whether collate_fn returns a list of tensors or a single stacked tensor
            if isinstance(batch["images"], (list, tuple)):
                images_tensors = torch.stack(batch["images"]).to(device)
            else:
                images_tensors = batch["images"].to(device)

            with torch.no_grad():
                # Expected Output shape: [Batch_Size, Num_Views]
                scores_batch = view_selector_model(images_tensors, tokenized)
            print("fusion type found and scores computed!")

        # ------------------------------------------------------------------
        # Per-Sample Logic
        # ------------------------------------------------------------------
        for i in range(start_i, batch_size_current):
            img_paths = image_paths_list[i]
            chosen_ques = questions[i]
            
            # Get the cad folder path from the image path
            cad_folder_path = img_paths[0].split("/mesh_views/")[0] 
            for file in os.listdir(cad_folder_path):
                if file.endswith('.step'):
                    cad_file_path = os.path.join(cad_folder_path, file)
                    break
            cad_file_name = os.path.basename(cad_file_path)

            feature = 'edge' if 'edge' in chosen_ques else 'face'
            
            with open(f"{cad_folder_path}/views_and_ques_{feature}_augmented_.log", 'r', encoding="utf-8") as f:
                for line in f:
                    part_dict = ast.literal_eval(line.strip())
                    if part_dict['question'] == chosen_ques:
                        break

            chosen_ans = part_dict['answer']
            chosen_marked_view_path = part_dict['marked_image']
            feature_ = os.path.basename(chosen_marked_view_path).split('_marked_')[1].split('[')[0]
            assert feature == feature_, f"Feature in question {feature} does not match feature in marked view {feature_}"
            
            feature_idx = int(os.path.basename(chosen_marked_view_path).split('[')[1].split(']')[0])
            print(f"Randomly selected question for {cad_file_name}: {chosen_ques}")

            # Non-Maximum Suppression / View Selection
            if fusion_type in modality_fusion_variants:
                # Extract the specific scores for this sample
                scores = scores_batch[i].squeeze().cpu().numpy()
                selected_views = viewNMS(img_paths, scores, angle_threshold=45)
            else:
                selected_views = np.random.choice(img_paths, size=topk, replace=False).tolist()

            # Logging
            view_path_dict = {}
            view_path_dict['file_path'] = cad_file_name
            view_path_dict['feature'], view_path_dict['feature_idx'] = feature, feature_idx
            view_path_dict['top_pred_views_nms45'] = selected_views
            
            with open(fpath, 'a') as fout:
                fout.write(f'{view_path_dict}\n')