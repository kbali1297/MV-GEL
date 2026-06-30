import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def inspect_masks(pred_mask, gt_mask, image_path, prompt=None):
    save_dir = "debug_masks_cad"
    os.makedirs(save_dir, exist_ok=True)

    pred_mask = (pred_mask * 255).cpu().numpy()
    pred_uint8 = pred_mask.astype(np.uint8)

    cad_name = os.path.dirname(image_path).split('/')[-1]
    image_id = os.path.basename(image_path)
    # Image.fromarray(pred_uint8).save(
    #     os.path.join(save_dir, f"{image_id}_pred.png")
    # )

    gt_mask = (gt_mask * 255).cpu().numpy()
    gt_uint8 = gt_mask.astype(np.uint8)

    # Image.fromarray(gt_uint8).save(
    #     os.path.join(save_dir, f"{image_id}_gt.png")
    # )

    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")

    # 2️⃣ Convert to tensor (C,H,W)
    transform = transforms.ToTensor()
    image = transform(img)

    # 3️⃣ Add batch dimension → (1,C,H,W)
    image = image.unsqueeze(0)

    image_np = image[0].permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(image_np)
    plt.imshow(pred_mask, alpha=0.5)
    plt.axis("off")

    # plt.savefig(os.path.join(save_dir, f"{image_id}_pred_overlay.png"),
    #             bbox_inches="tight",
    #             pad_inches=0)
    plt.close()

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ax[0].imshow(image_np)
    ax[0].set_title("Image")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground Truth")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Prediction")

    for a in ax:
        a.axis("off")

    plt.savefig(os.path.join(save_dir, f"{cad_name}_{image_id}_comparison.png"))
    plt.close()

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()

    iou = intersection / (union + 1e-6)

    print(f"{cad_name}_{image_id} IoU: {iou:.4f}")

    print(prompt) if prompt is not None else print("")