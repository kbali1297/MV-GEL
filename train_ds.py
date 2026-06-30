import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import tqdm
from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset_ import CAD_VQA_dataset, cad_collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

import torch.distributed as dist

# Repo-relative paths: ROOT is this file's dir (code); DATA_ROOT (override via
# MVGEL_ROOT) holds the git-ignored base weights (LISA-7B / SAM) and the CAD
# dataset logs (train_dataset.log / val_dataset.log).
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("MVGEL_ROOT", ROOT)


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        print(f"[Rank {rank}] Distributed initialized. World size: {world_size}")
    else:
        print("Not using distributed mode.")

if "LOCAL_RANK" in os.environ:
    print(f"Running rank {os.environ['LOCAL_RANK']}")

def parse_args(args):
    #nohup /data/1bali/miniforge3/envs/LISA_multi_view/bin/deepspeed   --include localhost:0,1,2   train_ds_.py   --exp_name lisa   --resume runs/lisa/ckpt_model_CAD_LISA_universal   --load_universal   --epochs 60   > resume_CAD_3gpu_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default=os.path.join(DATA_ROOT, "model/load_files&weights/LISA-7B-v1-explanatory")
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=96, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default=os.path.join(DATA_ROOT, "model/segment_anything/load_files&weights/sam_vit_h_4b8939.pth"), type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--load_universal", action="store_true", default=False)
    parser.add_argument(
        "--caption_variant",
        default="corrected",
        type=str,
        choices=["corrected", "original"],
        help="Which caption logs to use: 'corrected' -> *_augmented_corrected.log (viewpoint-fixed), "
             "'original' -> *_augmented_.log (old captions).",
    )
    parser.add_argument(
        "--reset_schedule",
        action="store_true",
        default=False,
        help="Load only model weights from --resume (skip optimizer/LR-scheduler state) and start a fresh schedule at epoch 0.",
    )
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    """
    cd /data/1bali/Other_LLM_projects/ECCV_2026/LISA

nohup /data/1bali/miniforge3/envs/LISA_multi_view/bin/deepspeed \
  --include localhost:0,1,2,3 \
  train_ds_.py \
  --exp_name lisa \
  --resume runs/lisa/ckpt_model_CAD_LISA_universal \
  --load_universal \
  --epochs 60 \
  > resume_CAD_4gpu_20260621_162010.log 2>&1 &
    """
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    init_distributed()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    args.distributed = world_size > 1
    train_dataset = CAD_VQA_dataset(tokenizer, args.vision_tower, 'train', caption_variant=args.caption_variant)

    if args.no_eval == False:
        val_dataset = CAD_VQA_dataset(tokenizer, args.vision_tower, 'val', caption_variant=args.caption_variant)
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    temp_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
    )

    steps_per_epoch = len(temp_loader) // world_size
    total_steps = args.epochs * steps_per_epoch
    args.steps_per_epoch = steps_per_epoch
    
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "train_batch_size": args.batch_size * args.grad_accumulation_steps * world_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": total_steps,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 2,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    if args.load_universal:
        ds_config["checkpoint"] = {"load_universal": True}


    # if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    #     args.distributed = True
    #     args.local_rank = int(os.environ["LOCAL_RANK"])
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend="nccl")
    # else:
    #     args.distributed = False
    #     args.local_rank = 0
    #     torch.cuda.set_device(0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    #print("Trainable tensors passed to DeepSpeed:", len(trainable_params))
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters= trainable_params,
        training_data=train_dataset,
        collate_fn=partial(
            cad_collate_fn,
            tokenizer=tokenizer,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # print("Optimizer param groups:")
    # for i, group in enumerate(optimizer.param_groups):
    #     print(f"Group {i} param count:", len(group["params"]))

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    args.steps_per_epoch = steps_per_epoch

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        if args.reset_schedule:
            load_path, client_state = model_engine.load_checkpoint(
                args.resume,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
            args.start_epoch = 0
            print(
                "loaded weights only from {}, fresh schedule starting at epoch 0".format(
                    args.resume
                )
            )
        else:
            load_path, client_state = model_engine.load_checkpoint(args.resume)
            with open(os.path.join(args.resume, "latest"), "r") as f:
                ckpt_dir = f.readlines()[0].strip()
            args.start_epoch = (
                int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
            )
            print(
                "resume training from {}, start from epoch {}".format(
                    args.resume, args.start_epoch
                )
            )

    # validation dataset
    val_loader = None
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                cad_collate_fn,
                tokenizer=tokenizer,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    top_checkpoints = []   # will store best 1
    max_keep = 1
    save_root = os.path.join(args.log_dir, f"ckpt_model")
    
    for epoch in tqdm.tqdm(range(args.start_epoch, args.epochs), total=args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            val_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        
        if not args.no_eval:

            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)

            # 🔥 Use real DeepSpeed global step
            global_step = model_engine.global_steps
            tag = f"global_step{global_step}"

            # Save checkpoint (ALL ranks must call this)
            model_engine.save_checkpoint(save_root, tag=tag)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Only rank 0 manages ranking + deletion
            if args.local_rank == 0:

                top_checkpoints.append({
                    "score": giou,
                    "ciou": ciou,
                    "tag": tag
                })

                # Sort by giou (primary), ciou (secondary)
                top_checkpoints = sorted(
                    top_checkpoints,
                    key=lambda x: (x["score"], x["ciou"]),
                    reverse=True
                )

                # Keep only top 3
                if len(top_checkpoints) > max_keep:
                    to_remove = top_checkpoints[max_keep:]
                    top_checkpoints = top_checkpoints[:max_keep]

                    for ckpt in to_remove:
                        remove_path = os.path.join(save_root, ckpt["tag"])
                        print(f"Removing old checkpoint: {remove_path}")
                        if os.path.exists(remove_path):
                            try:
                                shutil.rmtree(remove_path)
                            except: pass
                print(f"Current Top-{max_keep} Checkpoints:")
                for ckpt in top_checkpoints:
                    print(
                        f"  {ckpt['tag']} | giou={ckpt['score']:.4f} | ciou={ckpt['ciou']:.4f}"
                    )


def train(
    train_loader,
    val_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step, input_dict in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

    # for global_step in range(args.steps_per_epoch):
    #     for i in range(args.grad_accumulation_steps):
    #         try:
    #             input_dict = next(train_iter)
    #         except:
    #             train_iter = iter(train_loader)
    #             input_dict = next(train_iter)

        data_time.update(time.time() - end)
        input_dict = dict_to_cuda(input_dict)

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        output_dict = model(**input_dict)

        loss = output_dict["loss"]
        ce_loss = output_dict["ce_loss"]
        mask_bce_loss = output_dict["mask_bce_loss"]
        mask_dice_loss = output_dict["mask_dice_loss"]
        mask_loss = output_dict["mask_loss"]

        losses.update(loss.item(), input_dict["images"].size(0))
        ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
        mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
        mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
        mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
        
        model.backward(loss)
        model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

        if global_step % 100==0:
            if not args.no_eval and val_loader is not None:
                giou, ciou = validate(val_loader, model, epoch, writer, args)

        # if global_step > 0 and global_step % 20 == 0:

        #     save_dir = os.path.join(args.log_dir, "ckpt_model")
        #     tag = f"step_{global_step}"

        #     # 1️⃣ All ranks save first
        #     model.save_checkpoint(save_dir, tag=tag)

        #     # 2️⃣ Wait until all ranks finished writing
        #     torch.distributed.barrier()

        #     # 3️⃣ Only rank 0 does cleanup AFTER saving is fully done
        #     if args.local_rank == 0:
        #         max_keep = 3

        #         step_dirs = [
        #             d for d in os.listdir(save_dir)
        #             if d.startswith("step_")
        #         ]

        #         step_dirs = sorted(
        #             step_dirs,
        #             key=lambda x: int(x.replace("step_", ""))
        #         )

        #         if len(step_dirs) > max_keep:
        #             to_delete = step_dirs[0]
        #             full_path = os.path.join(save_dir, to_delete)
        #             if os.path.exists(full_path):
        #                 shutil.rmtree(full_path)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
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
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    model_engine.train()
    torch.cuda.empty_cache()
    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])
