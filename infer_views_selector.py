import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
import torch.optim as optim
from model.part_views import PromptConditionedViewSelector, pairwise_ranking_loss
from PIL import Image
from torchvision import transforms
import numpy as np
from utils.dataset_ import CAD_VQA_dataset, views_collate_fn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def compute_img_rank_batch(img_batch):

    images, ranks = [], []
    random_img_paths = []
    for sample in img_batch:
        img_paths= [f"{os.path.dirname(img_path)}/saved_views/{os.path.basename(img_path.split('_marked')[0])}.png" for img_path in sample]
        img_tensors= []
        for img_path in img_paths:
            img_PIL = Image.open(img_path).convert("RGB")
            img_tensor = transform(img_PIL)
            img_tensors.append(img_tensor)
        random_idxs = np.random.choice(range(len(img_tensors)), size=len(img_tensors), replace=False)    
        images.append(torch.stack(img_tensors)[random_idxs])
        ranks.append(torch.linspace(len(img_tensors)-1, 0, len(img_tensors))[random_idxs])
        random_img_paths.append([img_paths[idx] for idx in random_idxs])
    
    images_batch = torch.stack(images)
    ranks_batch = torch.stack(ranks)

    return images_batch, ranks_batch, random_img_paths


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    device = "cuda:0"

    model = PromptConditionedViewSelector().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    train_dataset = CAD_VQA_dataset(
        tokenizer=tokenizer,
        split='train'
    )
    val_dataset = CAD_VQA_dataset(
        tokenizer=tokenizer,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=42,
        shuffle=True,
        collate_fn=views_collate_fn,
        num_workers=4
    )

    
    val_loader = DataLoader(
        val_dataset,
        batch_size=42,
        shuffle=False,
        collate_fn=views_collate_fn,
        num_workers=4
    )
    # ==============================
    # Setup
    # ==============================

    num_epochs = 20
    patience = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    writer = SummaryWriter(log_dir="runs/view_selector")

    # ==============================
    # TRAINING LOOP
    # ==============================

    for epoch in range(num_epochs):

        print(f"\n========== Epoch {epoch+1}/{num_epochs} ==========")

        # =========================
        # TRAIN
        # =========================
        model.train()
        total_loss = 0.0

        train_bar = tqdm(train_loader, desc="Training", dynamic_ncols=True)
        epoch_start = time.time()

        for step, batch in enumerate(train_bar):

            images_batch, ranks_batch, _ = compute_img_rank_batch(batch["top_views"])
            images_batch = images_batch.to(device)
            ranks_batch = ranks_batch.to(device)

            questions = batch["question"]

            tokenized = tokenizer(
                questions,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            scores, _ = model(images_batch, tokenized)
            loss = pairwise_ranking_loss(scores, ranks_batch)

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

        epoch_time = time.time() - epoch_start
        train_epoch_loss = total_loss / len(train_loader)

        print(f"Train Loss: {train_epoch_loss:.4f} | Time: {epoch_time:.1f}s")

        writer.add_scalar("Loss/train", train_epoch_loss, epoch)

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        val_loss_total = 0.0

        val_bar = tqdm(val_loader, desc="Validation", dynamic_ncols=True)

        with torch.no_grad():
            for step, batch in enumerate(val_bar):

                images_batch, ranks_batch, _ = compute_img_rank_batch(batch["top_views"])
                images_batch = images_batch.to(device)
                ranks_batch = ranks_batch.to(device)

                questions = batch["question"]

                tokenized = tokenizer(
                    questions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(device)

                scores, _ = model(images_batch, tokenized)
                loss_rank = pairwise_ranking_loss(scores, ranks_batch)

                val_loss_total += loss_rank.item()
                avg_val_loss = val_loss_total / (step + 1)

                val_bar.set_postfix({
                    "val_loss": f"{loss_rank.item():.4f}",
                    "avg_val_loss": f"{avg_val_loss:.4f}"
                })

        val_epoch_loss = val_loss_total / len(val_loader)
        print(f"Val Loss: {val_epoch_loss:.4f}")

        writer.add_scalar("Loss/val", val_epoch_loss, epoch)

        # =========================
        # EARLY STOPPING + SAVE BEST
        # =========================
        if val_epoch_loss < best_val_loss:
            print("✅ Validation improved — saving model.")
            best_val_loss = val_epoch_loss
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_epoch_loss,
            }, "best_model_view_ranker.pt")

        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print("⛔ Early stopping triggered.")
                break

    writer.close()
    print("Training complete.")
        