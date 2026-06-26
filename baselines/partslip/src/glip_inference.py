import os
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import (
    GLIPDemo,
    create_positive_map,
    create_positive_map_label_to_token_from_positive_map,
)
from maskrcnn_benchmark.structures.image_list import to_image_list

def load_img(file_name):
    pil_image = Image.open(file_name).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def load_model(config_file, weight_file):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )
    return glip_demo

def draw_rectangle(img, x0, y0, x1, y1):
    color = np.random.rand(3) * 255
    img = img.astype(np.float64)
    img[y0:y1, x0-1:x0+2, :3] = color
    img[y0:y1, x1-1:x1+2, :3] = color
    img[y0-1:y0+2, x0:x1, :3] = color
    img[y1-1:y1+2, x0:x1, :3] = color
    img[y0:y1, x0:x1, :3] /= 2
    img[y0:y1, x0:x1, :3] += color * 0.5
    img = img.astype(np.uint8)
    return img

def save_individual_img(image, bbox, labels, n_cat, pred_dir, view_id):
    n = len(labels)
    result_list = [np.copy(image) for i in range(n_cat)]
    for i in range(n):
        l = labels[i] - 1
        x0, y0, x1, y1 = bbox[i]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        result_list[l] = draw_rectangle(result_list[l], x0, y0, x1, y1)
    for i in range(n_cat):
        plt.imsave("%s/%d_%d.png" % (pred_dir, view_id, i), result_list[i][:, :, [2, 1, 0]])

def glip_inference(glip_demo, save_dir, part_names, num_views=10,
                    save_pred_img=True, save_individual_img=False, save_pred_json=False,
                    confidence=0.5):
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok = True)
    predictions = []
    for i in range(num_views):
        image = load_img("%s/rendered_img/%d.png" % (save_dir, i))
        result, top_predictions = glip_demo.run_on_web_image(image, part_names, confidence) 
        if save_pred_img:   
            plt.imsave("%s/%d.png" % (pred_dir, i), result[:, :, [2, 1, 0]])
        bbox = top_predictions.bbox.cpu().numpy()
        score = top_predictions.get_field("scores").cpu().numpy()
        labels = top_predictions.get_field("labels").cpu().numpy()
        if save_individual_img:
            save_individual_img(image, bbox, labels, len(part_names), pred_dir, i)
        for j in range(len(bbox)):
            x1, y1, x2, y2 = bbox[j].tolist()
            predictions.append({"image_id" : i,
                                "category_id" : labels[j].item(),
                                "bbox" : [x1,y1, x2-x1, y2-y1],
                                "score" : score[j].item()})
    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions, outfile)
    return predictions


def glip_inference_batched(glip_demo, save_dir, part_names, num_views=10,
                           save_pred_img=True, save_pred_json=False,
                           confidence=0.5, batch_size=None):
    """Same output format as ``glip_inference`` but runs multiple views in
    one forward pass. ``part_names`` must be a list of category strings
    (the standard PartSLIP usage). The same caption / positive_map is
    reused for every image in the batch, which is correct because every
    view of a sample is grounded against the same set of category names.

    Set ``batch_size`` to control GPU memory; default is all views in
    a single batch.
    """
    pred_dir = os.path.join(save_dir, "glip_pred")
    os.makedirs(pred_dir, exist_ok=True)

    assert isinstance(part_names, list), \
        "glip_inference_batched expects part_names to be a list of category strings"

    # Build caption + positive map ONCE (shared across all views).
    caption_string = ""
    tokens_positive = []
    sep = " . "
    glip_demo.entities = part_names
    for word in part_names:
        tokens_positive.append([[len(caption_string), len(caption_string) + len(word)]])
        caption_string += word
        caption_string += sep
    tokenized = glip_demo.tokenizer([caption_string], return_tensors="pt")
    positive_map = create_positive_map(tokenized, tokens_positive)
    plus = 1 if glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0
    positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(
        positive_map, plus=plus
    )
    glip_demo.plus = plus
    glip_demo.positive_map_label_to_token = positive_map_label_to_token

    if batch_size is None or batch_size <= 0:
        batch_size = num_views

    predictions_out = []
    for start in range(0, num_views, batch_size):
        end = min(start + batch_size, num_views)
        imgs_np = []
        imgs_t = []
        sizes = []
        for i in range(start, end):
            img = load_img(f"{save_dir}/rendered_img/{i}.png")  # BGR HWC uint8
            imgs_np.append(img)
            sizes.append(img.shape[:2])  # (H, W)
            imgs_t.append(glip_demo.transforms(img))
        image_list = to_image_list(
            imgs_t, glip_demo.cfg.DATALOADER.SIZE_DIVISIBILITY
        ).to(glip_demo.device)
        B = end - start
        with torch.no_grad():
            preds = glip_demo.model(
                image_list,
                captions=[caption_string] * B,
                positive_map=positive_map_label_to_token,
            )
            preds = [p.to(glip_demo.cpu_device) for p in preds]

        for bi, pred in enumerate(preds):
            view_id = start + bi
            H, W = sizes[bi]
            pred = pred.resize((W, H))
            # Confidence threshold.
            scores_all = pred.get_field("scores")
            keep = (scores_all > confidence).nonzero(as_tuple=False).squeeze(1)
            pred = pred[keep]
            scores_all = pred.get_field("scores")
            _, idx = scores_all.sort(0, descending=True)
            pred = pred[idx]

            bbox = pred.bbox.cpu().numpy()
            score = pred.get_field("scores").cpu().numpy()
            labels = pred.get_field("labels").cpu().numpy()

            if save_pred_img:
                # Lightweight visual: just draw kept boxes on the loaded image.
                vis = imgs_np[bi][:, :, [2, 1, 0]].copy()  # to RGB
                for j in range(len(bbox)):
                    x0, y0, x1, y1 = [int(v) for v in bbox[j].tolist()]
                    vis = draw_rectangle(vis, x0, y0, x1, y1)
                plt.imsave(f"{pred_dir}/{view_id}.png", vis)

            for j in range(len(bbox)):
                x1, y1, x2, y2 = bbox[j].tolist()
                predictions_out.append({
                    "image_id": view_id,
                    "category_id": labels[j].item(),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score[j].item(),
                })

    if save_pred_json:
        with open("%s/pred.json" % pred_dir, "w") as outfile:
            json.dump(predictions_out, outfile)
    return predictions_out
