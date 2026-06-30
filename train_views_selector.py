import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPImageProcessor
import torch.optim as optim
from model.part_views import *
from PIL import Image
from torchvision import transforms
import numpy as np
from utils.dataset_ import CAD_ViewRank_Dataset, views_collate_fn, extract_el_az_from_view_desc
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
import torch.multiprocessing as mp
import argparse

# Repo-relative paths: ROOT is this file's dir; DATA_ROOT (override via MVGEL_ROOT)
# is where the git-ignored view-selector checkpoints are written/resumed.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)

#     images, ranks, image_paths = [],[],[]
#     for sample in img_batch:
#         img_paths= [f"{os.path.dirname(img_path)}/mesh_views/{os.path.basename(img_path.split('_marked')[0])}.png" for img_path in sample]
#         ranks = list(range(len(img_paths)-1, -1, -1)) # highest rank = best view = 0, lowest rank = worst view = V-1
#         image_orient_dict = {}
#         el_set, az_set = set(), set()
#         for idx, img_path in enumerate(img_paths):
#             img_PIL = Image.open(img_path).convert("RGB")
#             img_tensor = clip_img_processor(images=img_PIL, return_tensors="pt")["pixel_values"].squeeze(0) #(3,H,W)
#             el, az = extract_el_az_from_view_desc(img_path)
#             el_set.add(el)
#             az_set.add(az)
#             image_orient_dict[f'{el}_{az}'] = (img_tensor, ranks[idx], img_path)
        
#         sample_images, sample_ranks, sample_image_paths = [], [], []
#         el_list_sorted = sorted(el_set, reverse=True) # sort by elevation first (descending)
#         az_list_sorted = sorted(az_set) # sort by azimuth next (ascending)
#         for el in el_list_sorted:
#             for az in az_list_sorted:
#                 key = f'{el}_{az}'
#                 if key in image_orient_dict:
#                     sample_images.append(image_orient_dict[key][0])
#                     sample_ranks.append(image_orient_dict[key][1])
#                     sample_image_paths.append(image_orient_dict[key][2])
#         sample_images_tensor = torch.stack(sample_images) #(V,3,H,W)
#         sample_ranks_tensor = torch.tensor(sample_ranks) #(V,)
#         images.append(sample_images_tensor)
#         ranks.append(sample_ranks_tensor)
#         image_paths.append(sample_image_paths)
        
#     images_batch = torch.stack(images)
#     ranks_batch = torch.stack(ranks)
    
#     return images_batch, ranks_batch, image_paths


if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    parser = argparse.ArgumentParser(description="GELviews training")
    parser.add_argument("--modality_fusion", default=None, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()

    fusion_method = args.modality_fusion
    batch_size = args.batch_size
    device = "cuda:0"

    print(args)
    #model = PromptConditionedViewSelector(clip_model_name="openai/clip-vit-base-patch16").to(device)
    #model = BiDirectionalPatchViewSelector(clip_model_name="openai/clip-vit-base-patch16").to(device)
    #model = Simple_ViewSelector().to(device)
    #model = GeometricViewAdapter().to(device)
    fusion_method = None if fusion_method=="No fusion" else fusion_method
    model = LoraCLIPViewSelector_Ablation(fusion_type=fusion_method).to(device)
    #model = LoraCLIPViewSelector_simple().to(device)
    #model = CLIPViewSelector().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")#openai/clip-vit-large-patch14")
    clip_img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")#openai/clip-vit-large-patch14")
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    model_path = os.path.join(DATA_ROOT, f"best_model_view_ranker_cliplora_{fusion_method}.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("No checkpoint found, starting from scratch.")

    train_dataset = CAD_ViewRank_Dataset(
        clip_image_processor=clip_img_processor,
        split='train'
    )
    val_dataset = CAD_ViewRank_Dataset(
        clip_image_processor=clip_img_processor,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=views_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=views_collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False
    )
    # ==============================
    # Setup
    # ==============================

    num_epochs = 3
    patience = 10
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    writer = SummaryWriter(log_dir="runs/view_selector")


    def cleanup():
        print("\nCleaning up CUDA + workers...")
        torch.cuda.empty_cache()
        writer.close()

    def signal_handler(sig, frame):
        print("\nInterrupted! Cleaning up before exit...")
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # ==============================
    # TRAINING LOOP
    # ==============================

    try:
        global_step = 0

        for epoch in range(num_epochs):

            print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")

            model.train()
            total_loss = 0.0
            epoch_start = time.time()

            train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

            for step, batch in enumerate(train_bar):

                global_step += 1

                images_batch, ranks_batch = batch["images"], batch["ranks"]
                images_batch = images_batch.to(device)
                ranks_batch = ranks_batch.to(device)

                questions = batch["question"]

                tokenized = tokenizer(
                    questions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                scores = model(images_batch, tokenized)
                loss = pairwise_ranking_loss(scores, ranks_batch, topk_ratio=0.09) #Number of views positive: int(V*topk_ratio)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)

                current_lr = optimizer.param_groups[0]["lr"]

                train_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}"
                })

                writer.add_scalar("Loss/train_step", loss.item(), global_step)

                ## validation every 1000 steps
                if global_step % 1000 == 0 or global_step==len(train_bar):

                    model.eval()
                    val_loss_total = 0.0
                    val_bar = tqdm(val_loader, desc="Validating", dynamic_ncols=True)
                    with torch.no_grad():
                        for val_step, val_batch in enumerate(val_bar):

                            images_batch, ranks_batch = val_batch["images"], val_batch["ranks"]
                            images_batch = images_batch.to(device)
                            ranks_batch = ranks_batch.to(device)

                            questions = val_batch["question"]

                            tokenized = tokenizer(
                                questions,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                            ).to(device)

                            scores = model(images_batch, tokenized)
                            loss_rank = pairwise_ranking_loss(scores, ranks_batch)

                            val_loss_total += loss_rank.item()

                    val_loss = val_loss_total / len(val_loader)

                    writer.add_scalar("Loss/val_step", val_loss, global_step)

                    print(f"\nStep {global_step} | Validation Loss: {val_loss:.4f}")

                    # ==============================================
                    # SAVE ONLY IF IMPROVED
                    # ==============================================
                    if val_loss < best_val_loss:
                        print("✅ Validation improved — saving model.")
                        best_val_loss = val_loss
                        epochs_without_improvement = 0

                        torch.save({
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": val_loss,
                        }, os.path.basename(model_path))

                    else:
                        epochs_without_improvement += 1
                        print(f"⚠️ No improvement ({epochs_without_improvement}/{patience})")

                        if epochs_without_improvement >= patience:
                            print("⛔ Early stopping triggered.")
                            cleanup()
                            sys.exit(0)

                    model.train()

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

        writer.close()
        print("Training complete.")

    finally:
        cleanup()
        