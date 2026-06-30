import glob
import os
import re
import random
from torch.utils.data._utils.collate import default_collate
import open3d as o3d
import cv2
import pyvista as pv
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import trimesh

# Repo-relative data root. Dataset logs (train_dataset.log / val_dataset*.log) and
# the CAD dataset are git-ignored; by default we resolve relative paths against the
# repo root (parent of utils/). Set MVGEL_ROOT to point at a separate data location.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_ROOT = os.environ.get("MVGEL_ROOT", _REPO_ROOT)


def _resolve_dataset_log(path):
    """Resolve a (possibly relative) dataset-log path.

    Absolute paths are returned unchanged. Relative paths are tried, in order,
    against: the current working dir, the data root (MVGEL_ROOT), the repo root,
    and the repo's ``configs/`` dir (where the checked-in split logs live). The
    first existing candidate wins; if none exist we fall back to the data-root
    join so the original (informative) FileNotFoundError is raised downstream.
    """
    if os.path.isabs(path):
        return path
    candidates = [
        path,
        os.path.join(_DATA_ROOT, path),
        os.path.join(_REPO_ROOT, path),
        os.path.join(_REPO_ROOT, "configs", path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return os.path.join(_DATA_ROOT, path)


try:
    from torch_cluster import fps
except ImportError:  # torch_cluster is optional; only needed by fps_fast
    fps = None
# The model.llava imports below are only needed by the LLM-preprocessing paths
# (multimodal_conv*, encode_*). The zero-shot point-cloud baselines (PartSLIP,
# Find3D, PatchAlign3D) only use CAD_ViewRank_Dataset and run in conda envs whose
# transformers version is incompatible with model.llava. Guard the import so the
# dataset class stays importable everywhere; the LLM helpers will only be used in
# the LISA env where these imports succeed.
try:
    from model.llava import conversation as conversation_lib
    from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                       IMAGE_TOKEN_INDEX)
    from model.llava.mm_utils import tokenizer_image_token
except Exception as _llava_import_err:  # pragma: no cover
    conversation_lib = None
    DEFAULT_IMAGE_TOKEN = "<image>"
    IGNORE_INDEX = -100
    IMAGE_TOKEN_INDEX = -200
    tokenizer_image_token = None
    print(f"[dataset_] model.llava unavailable ({_llava_import_err}); "
          f"LLM-preprocessing paths disabled, CAD_ViewRank_Dataset still works.")

from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)
import ast
import PIL, cv2
from torchvision.transforms.functional import to_pil_image
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
import contextlib

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def extract_el_az_from_view_desc(view_desc):
    """
    Extracts elevation and azimuth from a view description string.
    Example view description: ".../view_e30_a-60.png"
    """
    el,az = view_desc.split('_e')[1].split('_')[0], view_desc.split('_a')[1].split('.')[0].split('_')[0]
    return int(el), int(az)

def remap_azimuth_old_convention(path):
    """Rewrite the azimuth in an image filename using (360 - az) % 360.

    Old-convention ('original') captions reference azimuths under a flipped
    convention; the image actually saved on disk uses (360 - az) % 360.
    Only the azimuth token (``_a<az>``) in the basename is changed.
    """
    directory = os.path.dirname(path)
    basename = os.path.basename(path)

    def _repl(match):
        az = int(match.group(1))
        return f"_a{(360 - az) % 360}"

    basename = re.sub(r'_a(-?\d+)', _repl, basename)
    return os.path.join(directory, basename)

def remove_smallest_component(mask):
    """
    Removes the smallest non-background connected component.
    Assumes mask is binary (0/255 or 0/1).
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    # If only background or single component → nothing to remove
    if num_labels <= 2:
        return mask_uint8

    # Ignore background (label 0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    # Find smallest component
    smallest_label = 1 + np.argmin(component_areas)

    cleaned = mask_uint8.copy()
    cleaned[labels == smallest_label] = 0

    return cleaned

def keep_largest_component(mask):
    """
    Keeps only the largest non-background connected component.
    """
    mask_uint8 = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    # If no components or only background
    if num_labels <= 1:
        return mask_uint8

    # Ignore background (label 0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]

    # Largest component
    largest_label = 1 + np.argmax(component_areas)

    cleaned = np.zeros_like(mask_uint8)
    cleaned[labels == largest_label] = 1

    return cleaned

def extract_mask(image_input, feature):
    """
    Robust color segmentation for blue (faces) or red (edges)
    Accepts either an image path (str) or a PyVista numpy array (RGB).
    """
    import cv2
    import numpy as np

    if isinstance(image_input, str):
        # Read from disk (returns BGR)
        img = cv2.imread(image_input)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif isinstance(image_input, np.ndarray):
        # Read from memory (PyVista returns RGB)
        hsv = cv2.cvtColor(image_input, cv2.COLOR_RGB2HSV)
    else:
        raise TypeError("image_input must be a file path string or a numpy array")

    if feature == "faces":  # BLUE
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif feature == "edges":  # RED (two HSV ranges)
        lower1 = np.array([0, 50, 50])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 50, 50])
        upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    else:
        raise ValueError("feature must be 'faces' or 'edges'")

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Retain Largest connected component
    mask = keep_largest_component(mask)

    return mask.astype(np.float32)

def sample_points_from_mesh(vertices, faces, num_samples):
    """
    vertices: [V, 3]
    faces: [F, 3]
    returns: [num_samples, 3]
    """

    # 1. Get triangle vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 2. Compute triangle areas
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = torch.norm(cross, dim=1) * 0.5

    # 3. Sample faces proportional to area
    face_probs = face_areas / face_areas.sum()
    sampled_face_indices = torch.multinomial(face_probs, num_samples, replacement=True)

    v0 = v0[sampled_face_indices]
    v1 = v1[sampled_face_indices]
    v2 = v2[sampled_face_indices]

    # 4. Barycentric sampling
    u = torch.rand(num_samples, 1, device=vertices.device)
    v = torch.rand(num_samples, 1, device=vertices.device)

    mask = (u + v > 1)
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]

    sampled_points = v0 + u * (v1 - v0) + v * (v2 - v0)

    return sampled_points

def farthest_point_sampling(points, n_samples):
    """
    points: [N, 3]
    return: [n_samples] indices
    """

    device = points.device
    N = points.shape[0]

    centroids = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10

    farthest = torch.randint(0, N, (1,), device=device)
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest].unsqueeze(0)

        dist = torch.sum((points - centroid) ** 2, dim=1)
        distances = torch.minimum(distances, dist)

        farthest = torch.argmax(distances)

    return centroids

# def mesh_to_fps_points(vertices, faces, total_samples=8192, fps_samples=1024):
#     """
#     vertices: [V, 3]
#     faces: [F, 3]
#     """

#     surface_points = sample_points_from_mesh(
#         vertices, faces, total_samples
#     )

#     fps_indices = farthest_point_sampling(
#         surface_points, fps_samples
#     )

#     fps_points = surface_points[fps_indices]

#     return fps_points  # [fps_samples, 3]

def fps_fast(points, K):
    """
    points: [N, 3]
    """
    if fps is None:
        raise ImportError("torch_cluster is required for fps_fast but is not installed.")
    ratio = K / points.shape[0]
    idx = fps(points, ratio=ratio, random_start=True)
    return points[idx], idx

def normalize_point_cloud(points):
    """
    points: [N, 3]
    """
    centroid = points.mean(dim=0)
    points = points - centroid
    scale = torch.max(torch.norm(points, dim=1))
    points = points / scale
    return points

class CAD_VQA_dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokenizer,
        vision_tower=None,
        split='train',
        caption_variant='corrected'
    ):
        # Resolve `{split}_dataset.log` against DATA_ROOT, the repo root, and
        # configs/ (where the committed train/val splits live).
        dataset_path = _resolve_dataset_log(f'{split}_dataset.log')
        self.VQA_dict_list = []

        # Select which caption log files to read.
        #   'corrected' -> views_and_ques_{edge,face}_augmented_corrected.log (viewpoint-fixed captions)
        #   'original'  -> views_and_ques_{edge,face}_augmented_.log          (old/original captions)
        caption_suffix_map = {
            'corrected': '_augmented_corrected',
            'original': '_augmented_',
        }
        if caption_variant not in caption_suffix_map:
            raise ValueError(
                f"caption_variant must be one of {list(caption_suffix_map.keys())}, got {caption_variant!r}"
            )
        caption_suffix = caption_suffix_map[caption_variant]
        self.caption_variant = caption_variant
        print(f"[CAD_VQA_dataset:{split}] using caption_variant={caption_variant!r} "
              f"-> views_and_ques_{{edge,face}}{caption_suffix}.log")

        self.tokenizer = tokenizer
        if vision_tower is not None:
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        else:
            self.clip_image_processor = None

        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.img_size = 1024
        self.ignore_label = 255
        
        
        feature_dict_list = []
        with open(dataset_path, 'r') as fread:
            for line_cad in fread:
                cad_folder_path = line_cad.strip()
                cad_edges_vqa_path = f'{cad_folder_path}/views_and_ques_edge{caption_suffix}.log'
                cad_faces_vqa_path = f'{cad_folder_path}/views_and_ques_face{caption_suffix}.log'
                features_path = [('edges', cad_edges_vqa_path), ('faces', cad_faces_vqa_path)]

                for feature, fpath in features_path:
                    with open(fpath, 'r') as f_read:
                        for line_edge_dict in f_read:
                            dict_read = ast.literal_eval(line_edge_dict)
                            dict_read['feature'] = feature
                            feature_dict_list.append(dict_read)
        
        for dict_read in tqdm(feature_dict_list, total=len(feature_dict_list)):
            input_img_path = f'{os.path.dirname(dict_read["unmarked_image"])}/mesh_views_corrected/{os.path.basename(dict_read["unmarked_image"])}'
            #input_img_clip, input_img = self.process_input_img(input_img_path)
            target_img_path = dict_read['marked_image']
            #target_mask = extract_mask(target_img_path, feature)

            # Old-convention ('original') captions store the azimuth under a flipped
            # convention; the image actually on disk uses (360 - az) % 360.
            if self.caption_variant == 'original':
                input_img_path = self._remap_azimuth_old_convention(input_img_path)
                target_img_path = self._remap_azimuth_old_convention(target_img_path)

            self.VQA_dict_list.append({
                "input_img_path":input_img_path,
                #"input_img": input_img, # FOR SAM
                #"input_img_clip":input_img_clip,
                "target_img_path":target_img_path,
                #"target_mask":target_mask,
                "question": dict_read["question"],
                "answer": dict_read["answer"],
                "feature": dict_read['feature'],
                "top_views": dict_read["top_views_desc"],
                "inference": False if split=='train' else True
                }
            )

    def _remap_azimuth_old_convention(self, path):
        """Rewrite the azimuth in an image filename using (360 - az) % 360.

        Old-convention ('original') captions reference azimuths under a flipped
        convention; the image actually saved on disk uses (360 - az) % 360.
        Only the azimuth token (``_a<az>``) in the basename is changed.
        """
        directory = os.path.dirname(path)
        basename = os.path.basename(path)

        def _repl(match):
            az = int(match.group(1))
            new_az = (360 - az) % 360
            return f"_a{new_az}"

        basename = re.sub(r'_a(-?\d+)', _repl, basename)
        return os.path.join(directory, basename)

    def process_input_img(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip
        
        image = np.array(to_pil_image(image))  # preprocess image for sam
        image_sam = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_sam = (image_sam - self.pixel_mean) / self.pixel_std

        # Pad
        # h, w = x.shape[-2:]
        # padh = self.img_size - h
        # padw = self.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return image_clip, image_sam

    def __getitem__(self, idx):

        sample = self.VQA_dict_list[idx]

        if self.clip_image_processor is not None:
            input_img_clip, input_img = self.process_input_img(sample["input_img_path"])
            target_mask = extract_mask(sample["target_img_path"], sample["feature"])
            mask = torch.from_numpy(target_mask).unsqueeze(0)
            label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
            resize = mask.shape[1:] #All images of shape 1024X1024 at the moment

            ## Preprocess mesh point cloud 
            # cad_folder_path = '/'.join(sample['input_img_path'].split('/')[:-2])
            # for filename in os.listdir(cad_folder_path):
            #     if filename.endswith('.obj'):
            #         mesh_path = os.path.join(cad_folder_path, filename)
            #         break
            # mesh = trimesh.load(mesh_path)
            # ## Sample points from mesh, resolution
            # points = sample_points_from_mesh(
            #     torch.from_numpy(mesh.vertices).float(),
            #     torch.from_numpy(mesh.faces).long(),
            #     num_samples=8192
            # )
            # fps_points, _ = fps_fast(points, K=1024)
            # points = normalize_point_cloud(fps_points)

            cad_folder_path = '/'.join(sample['input_img_path'].split('/')[:-2])
            el, az = extract_el_az_from_view_desc(sample["input_img_path"])
            #depth_map = np.load(f"{cad_folder_path}/depth_views/depth_e{el}_a{az}.npy")
            # deep_map_pickle = np.load(f"{cad_folder_path}/deep_views/DN_e{el}_a{az}.npz")
            # depth =  deep_map_pickle["depth"].astype(np.float32)[None,:]
            # normal = deep_map_pickle["normal"].astype(np.float32).transpose(2,0,1)/255.0
            # normal = normal * 2.0 -1.0
            # deep_map = np.concatenate([depth, normal], axis=0)
            return {
                    "input_img_path":sample["input_img_path"],
                    "input_img": input_img, # FOR SAM
                    "input_img_clip":input_img_clip,
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "target_img_path":sample["target_img_path"],
                    "target_mask":mask,
                    "label": label,
                    "resize":resize,
                    "feature": sample["feature"],
                    "top_views": sample["top_views"],
                    "inference": sample["inference"],
                    #"mesh_points": points,
                    #"depth_map": torch.from_numpy(deep_map)
            }
        else:
            return {
                "input_img_path":sample["input_img_path"],
                "question": sample["question"],
                "answer": sample["answer"],
                "target_img_path":sample["target_img_path"],
                "feature": sample["feature"],
                "top_views": sample["top_views"],
                "inference": sample["inference"]
            }
    
    def __len__(self):

        return len(self.VQA_dict_list)



def cad_collate_fn(batch, tokenizer=None,  use_mm_start_end=True, local_rank=-1): #using conv_type="llava_v1", " USER:", " ASSISTANT:", "</s>"
    image_path_list = []
    images_list = []
    images_clip_list = []
    question_list, answer_list = [], []
    conversation_list = []
    target_masks_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    top_views_list = []
    label_list = []
    resize_list = []
    #points_list = []
    depth_map_list = []
    for sample in batch:
        
        image_path_list.append(sample["input_img_path"])
        images_list.append(sample["input_img"])
        images_clip_list.append(sample["input_img_clip"])
        question_list.append(sample["question"])
        # answer_list.append(answer)
        conversation = (
            "A chat between a curious human and an artificial intelligence assistant." 
            "The assistant gives helpful, detailed,"
            "and polite answers to the human's questions."
            f" USER: <image>\n{sample['question']}.Please output the segmentation mask."
            f" ASSISTANT: {sample['answer']} It is given by [SEG].</s>"
        )
        conversation_list.append(conversation)
        ## Later for multi feature segmentation (multiple [SEG] tokens), no need for {answer} in assistance response
        target_masks_list.append(sample["target_mask"])
        #question_list.append(question)
        label_list.append(sample['label'])
        resize_list.append(sample['resize'])
        cnt += 1
        offset_list.append(cnt)
        inferences.append(sample["inference"])
        top_views_list.append(sample["top_views"])
        #points_list.append(sample["mesh_points"])
        if "depth_map" in sample and sample["depth_map"] is not None:
            depth_map_list.append(sample["depth_map"])
        else:
            has_depth = False

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    targets = input_ids.clone()

    sep = " ASSISTANT:"
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split('</s>')
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255

        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]


    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": target_masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": question_list,
        "top_views_list": top_views_list,
        #"sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        #"geometry_points": torch.stack(points_list, dim=0),
        **(
            {"depth_maps": torch.stack(depth_map_list, dim=0)}
            if len(depth_map_list) > 0
            else {}
        )
    }

def inspect_masks2(pred_mask, gt_mask, image_path, prompt=None):
    save_dir = "debug_masks"
    os.makedirs(save_dir, exist_ok=True)

    pred_mask = (pred_mask * 255).cpu().numpy()
    pred_uint8 = pred_mask.astype(np.uint8)

    image_id = os.path.basename(image_path)
    Image.fromarray(pred_uint8).save(
        os.path.join(save_dir, f"{image_id}_pred.png")
    )

    gt_mask = (gt_mask * 255).cpu().numpy()
    gt_uint8 = gt_mask.astype(np.uint8)

    Image.fromarray(gt_uint8).save(
        os.path.join(save_dir, f"{image_id}_gt.png")
    )

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

    plt.savefig(os.path.join(save_dir, f"{image_id}_pred_overlay.png"),
                bbox_inches="tight",
                pad_inches=0)
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

    plt.savefig(os.path.join(save_dir, f"{image_id}_comparison.png"))
    plt.close()

    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()

    iou = intersection / (union + 1e-6)

    print(f"{image_id} IoU: {iou:.4f}")

    print(prompt) if prompt is not None else print("")

class CAD_ViewRank_Dataset(torch.utils.data.Dataset):
    def __init__(self, clip_image_processor, split='train', dataset_log_path=None, caption_variant='corrected',
                 entity_allowlist=None):
        # `entity_allowlist`: optional set of (cad_name, feature, feature_idx) keys
        # (feature is the singular 'edge'/'face') to restrict evaluation to. When given,
        # only those exact entities are kept, so a clean single-pass re-run can target a
        # handful of entities instead of every entity in their CAD folders. When None,
        # all entities in the dataset log are used (default behaviour).

        # `dataset_log_path` lets callers point at an arbitrary dataset log (e.g.
        # val_dataset.log). When omitted we fall back to the
        # default `{split}_dataset.log` convention.
        if dataset_log_path is not None:
            dataset_path = dataset_log_path
            if not os.path.isabs(dataset_path):
                dataset_path = _resolve_dataset_log(dataset_path)
        else:
            dataset_path = os.path.join(_DATA_ROOT, f'{split}_dataset.log')

        # Which per-CAD caption log to read:
        #   'corrected' -> views_and_ques_{edge,face}_augmented_corrected.log (viewpoint-fixed)
        #   'original'  -> views_and_ques_{edge,face}_augmented_.log          (old captions)
        # For 'original', the view azimuths in the captions use a flipped convention; the
        # image actually on disk lives at (360 - az) % 360, so every constructed view path
        # is remapped via remap_azimuth_old_convention in __getitem__.
        caption_suffix_map = {
            'corrected': '_augmented_corrected',
            'original': '_augmented_',
        }
        if caption_variant not in caption_suffix_map:
            raise ValueError(
                f"caption_variant must be one of {list(caption_suffix_map.keys())}, got {caption_variant!r}"
            )
        caption_suffix = caption_suffix_map[caption_variant]
        self.caption_variant = caption_variant
        print(f'[CAD_ViewRank_Dataset:{split}] reading dataset log: {dataset_path} '
              f'| caption_variant={caption_variant!r} -> views_and_ques_{{edge,face}}{caption_suffix}.log')
        self.VQA_dict_list = []
        self.clip_image_processor = clip_image_processor

        feature_dict_list = []
        with open(dataset_path, 'r') as fread:
            for line_cad in fread:
                cad_folder_path = line_cad.strip()
                cad_name = os.path.basename(cad_folder_path)
                cad_edges_vqa_path = f'{cad_folder_path}/views_and_ques_edge{caption_suffix}.log'
                cad_faces_vqa_path = f'{cad_folder_path}/views_and_ques_face{caption_suffix}.log'
                features_path = [('edges', cad_edges_vqa_path), ('faces', cad_faces_vqa_path)]

                for feature, fpath in features_path:
                    with open(fpath, 'r') as f_read:
                        for line_edge_dict in f_read:
                            dict_read = ast.literal_eval(line_edge_dict)
                            dict_read['feature'] = feature
                            dict_read['cad_name'] = cad_name
                            feature_dict_list.append(dict_read)

        # Optional entity-level subset: keep only entities whose
        # (cad_name, feature_singular, feature_idx) key is in `entity_allowlist`.
        if entity_allowlist is not None:
            allow = set(entity_allowlist)
            feature_to_singular = {'edges': 'edge', 'faces': 'face'}
            filtered = []
            for dict_read in feature_dict_list:
                feat_singular = feature_to_singular.get(dict_read['feature'], dict_read['feature'])
                try:
                    feat_idx = int(os.path.basename(dict_read['marked_image']).split('[')[1].split(']')[0])
                except (IndexError, ValueError):
                    continue
                key = (str(dict_read['cad_name']), feat_singular, feat_idx)
                if key in allow:
                    filtered.append(dict_read)
            print(f'[CAD_ViewRank_Dataset:{split}] entity_allowlist active: '
                  f'{len(filtered)}/{len(feature_dict_list)} entities kept '
                  f'(allowlist size {len(allow)}).')
            feature_dict_list = filtered

        for dict_read in tqdm(feature_dict_list, total=len(feature_dict_list)):
            input_img_path = f'{os.path.dirname(dict_read["unmarked_image"])}/mesh_views_corrected/{os.path.basename(dict_read["unmarked_image"])}'
            #input_img_clip, input_img = self.process_input_img(input_img_path)
            target_img_path = dict_read['marked_image']
            #target_mask = extract_mask(target_img_path, feature)
            
            self.VQA_dict_list.append({
                "input_img_path":input_img_path,
                #"input_img": input_img, # FOR SAM
                #"input_img_clip":input_img_clip,
                "target_img_path":target_img_path,
                #"target_mask":target_mask,
                "question": dict_read["question"],
                "answer": dict_read["answer"],
                "feature": dict_read['feature'],
                "top_views": dict_read["top_views_desc"],
                "inference": False if split=='train' else True
                }
            )

            if dict_read['question']==None: 
                print(dict_read)

    def __len__(self):
        return len(self.VQA_dict_list)

    def __getitem__(self, idx):

        sample = self.VQA_dict_list[idx]

        img_paths = [
            f"{os.path.dirname(img_path)}/mesh_views_corrected/"
            f"{os.path.basename(img_path.split('_marked')[0])}.png"
            for img_path in sample["top_views"]
        ]

        # 'original' captions store azimuths under the flipped convention; remap each
        # constructed view path to the file that actually exists on disk (and so the
        # azimuth later extracted from this path is the corrected, geometrically-valid one).
        if getattr(self, 'caption_variant', 'corrected') == 'original':
            img_paths = [remap_azimuth_old_convention(p) for p in img_paths]

        # highest rank = best view = 0
        ranks = list(range(len(img_paths) - 1, -1, -1))

        image_orient_dict = {}
        el_set, az_set = set(), set()

        for i, img_path in enumerate(img_paths):

            img = Image.open(img_path).convert("RGB")

            img_tensor = self.clip_image_processor(
                images=img,
                return_tensors="pt"
            )["pixel_values"].squeeze(0)

            el, az = extract_el_az_from_view_desc(img_path)
            el_set.add(el)
            az_set.add(az)

            image_orient_dict[f"{el}_{az}"] = (
                img_tensor,
                ranks[i],
                img_path
            )

        el_list_sorted = sorted(el_set, reverse=True)
        az_list_sorted = sorted(az_set)

        sample_images = []
        sample_ranks = []
        sample_paths = []

        for el in el_list_sorted:
            for az in az_list_sorted:
                key = f"{el}_{az}"
                if key in image_orient_dict:
                    img_tensor, rank, path = image_orient_dict[key]
                    sample_images.append(img_tensor)
                    sample_ranks.append(rank)
                    sample_paths.append(path)

        images_tensor = torch.stack(sample_images)  # (V,3,H,W)
        ranks_tensor = torch.tensor(sample_ranks, dtype=torch.float32)

        return {
            "images": images_tensor,
            "ranks": ranks_tensor,
            "question": sample["question"],
            "image_paths": sample_paths
        }


from torch.utils.data._utils.collate import default_collate

def views_collate_fn(batch):
    return {
        "images": default_collate([b["images"] for b in batch]),
        "ranks": default_collate([b["ranks"] for b in batch]),
        "question": [b["question"] for b in batch],
        "image_paths": [b["image_paths"] for b in batch],
    }

def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict
