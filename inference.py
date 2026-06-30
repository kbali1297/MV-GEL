import os
import shutil
import json
import torch
import tqdm
import argparse
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from utils.dataset import ValDataset, collate_fn
from utils.utils import dict_to_cuda, intersectionAndUnionGPU
from functools import partial
from inspect_masks import inspect_masks

from tqdm import tqdm

# Repo-relative paths: ROOT is this file's dir (code); DATA_ROOT (override via
# MVGEL_ROOT) holds the git-ignored base weights / dataset.
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.path.join(DATA_ROOT, 'model/load_files&weights/LISA-7B-v1-explanatory_finetuned_deepspeed_2_CAD'), type=str)
    parser.add_argument("--dataset_dir", default=os.path.join(DATA_ROOT, 'dataset'), type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|train", type=str)
    parser.add_argument("--vision_tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--vision_pretrained", default=os.path.join(DATA_ROOT, "model/segment_anything/load_files&weights/sam_vit_h_4b8939.pth"), type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--precision", default="bf16", choices=["fp32","bf16","fp16"], type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    if 'finetuned_deepspeed' in os.path.basename(args.model_path):
        model_dir_new = process_model(args.model_path)
        build_index_file(model_dir_new)
        config_dir_path = model_dir_new.split('_finetuned_deepspeed')[0]
        shutil.copy(f'{config_dir_path}/added_tokens.json', model_dir_new)  
        shutil.copy(f'{config_dir_path}/config.json', model_dir_new)
        shutil.copy(f'{config_dir_path}/generation_config.json', model_dir_new)
        shutil.copy(f'{config_dir_path}/special_tokens_map.json', model_dir_new)
        shutil.copy(f'{config_dir_path}/tokenizer_config.json', model_dir_new)
        shutil.copy(f'{config_dir_path}/tokenizer.model', model_dir_new)

        args.model_path = model_dir_new

     # ------------------------
    # Load tokenizer
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False
    )

    # ------------------------
    # Precision setup
    # ------------------------
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # ------------------------
    # Load model
    # ------------------------
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    model_args = {
        "out_dim": args.out_dim,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    model = LISAForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **model_args
    )

    # 🔥 CRITICAL FIX: initialize delayed vision modules
    model.get_model().initialize_vision_modules(model.get_model().config)

    # Move vision tower explicitly to device/dtype
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=torch_dtype)

    model.to(device)
    model.eval()

    # ------------------------
    # Validation Dataset
    # ------------------------

    ## Standard use case
    # val_dataset = ValDataset(
    #     args.dataset_dir,
    #     tokenizer,
    #     args.vision_tower,
    #     args.val_dataset,
    #     args.image_size,
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=partial(
    #         collate_fn,
    #         tokenizer=tokenizer,
    #         conv_type="llava_v1",
    #         use_mm_start_end=True,
    #         local_rank=0,
    #     ),
    # )

    ## Cad use case
    from utils.dataset_ import CAD_VQA_dataset, cad_collate_fn
    val_dataset = CAD_VQA_dataset( 
        split="val",
        tokenizer=tokenizer,
        vision_tower=args.vision_tower
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=partial(
            cad_collate_fn,
            tokenizer=tokenizer,
            use_mm_start_end=True,
            local_rank=0,
        ),
    )

    # ------------------------
    # Metrics
    # ------------------------
    total_intersection = 0
    total_union = 0
    total_ciou = 0
    count = 0

    print("Running inference...")

    for input_dict in tqdm(val_loader):
        input_dict = dict_to_cuda(input_dict)

        # precision conversion
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        gt_masks = output_dict["gt_masks"][0].int()
        pred_binary = (pred_masks[0] > 0).int()

        for idx, (mask_i, pred_i) in enumerate(zip(gt_masks, pred_binary)):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                pred_i.contiguous().clone(),
                mask_i.contiguous(),
                2,
                ignore_index=255,
            )

            total_intersection += intersection_i[1].item()
            total_union += union_i[1].item()

            if union_i[1] > 0:
                total_ciou += (intersection_i[1] / union_i[1]).item()
            else:
                total_ciou += 1.0

            count += 1

            inspect_masks(pred_i, mask_i, input_dict['image_paths'][idx], input_dict['conversation_list'][idx])


    giou = total_ciou / count
    ciou = total_intersection / (total_union + 1e-10)

    print("================================")
    print(f"gIoU: {giou:.4f}")
    print(f"cIoU: {ciou:.4f}")
    print("================================")

def merge_lora_into_state_dict(state_dict, alpha=16, r=8):
    new_state_dict = {}
    lora_groups = {}

    # First pass: collect base weights and lora components
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
        # elif "lora_alpha" in key:
        #     prefix = key.replace(".lora_alpha", "")
        #     lora_groups.setdefault(prefix, {})["alpha"] = val
        else:
            new_state_dict[key_] = val

    # Second pass: merge LoRA into base weights
    scaling = alpha / r
    for prefix, group in lora_groups.items():
        #base_key = prefix + ".weight"
        base_key_clean = prefix.replace("base_model.model.", "")

        if base_key_clean not in new_state_dict.keys():
            continue  # skip if base weight not in shard

        W = new_state_dict[base_key_clean]
        A = group["A"]
        B = group["B"]
        #alpha = group.get("alpha", A.shape[0])  # default alpha=r
        #r = A.shape[0]

        delta = (torch.matmul(B.to(W.dtype), A.to(W.dtype))) * scaling

        new_state_dict[base_key_clean] = W + delta

    return new_state_dict


def process_model(model_path):
    model_path_new = f"{model_path}_merged"
    os.makedirs(model_path_new, exist_ok=True)

    state_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".bin"):
            print(f"Processing {file}")
            state_dict_temp = torch.load(f"{model_path}/{file}", map_location="cpu")
            state_dict.update(state_dict_temp)

    merged_state_dict = merge_lora_into_state_dict(state_dict)

    torch.save(merged_state_dict, f"{model_path_new}/pytorch_model.bin")

    return model_path_new
    #print("Done.")

        
def build_index_file(model_dir, index_filename="pytorch_model.bin.index.json"):
    weight_map = {}
    total_size = 0

    bin_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".bin")])

    for bin_file in bin_files:
        shard_path = os.path.join(model_dir, bin_file)
        state_dict = torch.load(shard_path, map_location="cpu")

        for key, tensor in state_dict.items():
            weight_map[key] = bin_file
            total_size += tensor.numel() * tensor.element_size()

    index_data = {
        "metadata": {
            "total_size": total_size
        },
        "weight_map": weight_map
    }

    with open(os.path.join(model_dir, index_filename), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Saved new {index_filename}")

if __name__ == "__main__":
    main()




        

    
