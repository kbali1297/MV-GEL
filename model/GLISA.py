from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from model.Point_MAE import Group, Encoder, TransformerEncoder

class PointMAEBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.group_divider = Group(
            num_group=config["model"]["num_group"],
            group_size=config["model"]["group_size"]
        )

        self.encoder = Encoder(
            encoder_channel=config["model"]["transformer_config"]["encoder_dims"]
        )
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config["model"]["transformer_config"]["trans_dim"]),
        )

        self.blocks = TransformerEncoder(
            embed_dim=config["model"]["transformer_config"]["trans_dim"],
            depth=config["model"]["transformer_config"]["depth"],
            num_heads=config["model"]["transformer_config"]["num_heads"],
        )

        self.norm = nn.LayerNorm(config["model"]["transformer_config"]["trans_dim"])

    def forward(self, pts):
        ## pts need to be fp32
        assert pts.dtype == torch.float32, "Need fp32 for KNN+FPS"
        neighborhood, center = self.group_divider(pts)

        ##----------------convert to model.dtype----------
        self.dtype = self.encoder.first_conv[0].weight.dtype
        neighborhood = neighborhood.to(self.dtype)
        center = center.to(self.dtype)
        ##------------------------------------------------
        
        tokens = self.encoder(neighborhood)  # B G C
        pos = self.pos_embed(center)

        tokens = self.blocks(tokens, pos)
        tokens = self.norm(tokens)

        return tokens  # B G C

class GeometryEncoder(nn.Module):
    """
    Wrapper around ULIP / PointMAE / mesh transformer.
    Must output tokens of shape [B, N, D]
    """

    def __init__(self, backbone, out_dim):
        super().__init__()
        self.backbone = backbone
        embed_dim = self.backbone.config["model"]["transformer_config"]["trans_dim"]
        for p in self.backbone.parameters(): ## Freeze encoder
            p.requires_grad = False

        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, points):
        """
        points: [B, N_points, 3] or mesh features
        returns: [B, N_geom_tokens, out_dim]
        """
        geom_tokens = self.backbone(points)   # [B, N, D_backbone]
        geom_tokens = self.proj(geom_tokens)
        return geom_tokens

class GatedGeometryCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        # Per-channel gate
        self.alpha = nn.Parameter(0.01 * torch.ones(hidden_size))

    def forward(self, hidden_states, geom_tokens):

        h = self.norm(hidden_states)

        cross_out, _ = self.cross_attn(
            query=h,
            key=geom_tokens,
            value=geom_tokens
        )

        gate = torch.tanh(self.alpha)[None, None, :]

        h = hidden_states + gate * cross_out
        h = h + self.ffn(self.norm(h))

        return h

class DepthEncoder(nn.Module):
    """
    Encode depth map [B,4,1024,1024] #Depth + Normal map
    to SAM-compatible embedding [B,256,64,64]
    """

    def __init__(self, out_channels=256):
        super().__init__()

        self.encoder = nn.Sequential(
            # 1024 → 512
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 512 → 256
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 256 → 128
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # 128 → 64
            nn.Conv2d(192, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth):
        """
        depth: [B,4,1024,1024]
        returns: [B,256,64,64]
        """
        return self.encoder(depth)
    
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_2d_sincos_position_embedding(H, W, C, device):
    """
    Returns: [1, H*W, C]
    """

    if C % 4 != 0:
        raise ValueError("Channels must be divisible by 4 for 2D sin-cos encoding.")

    y_embed = torch.linspace(0, 1, H, device=device)
    x_embed = torch.linspace(0, 1, W, device=device)

    yy, xx = torch.meshgrid(y_embed, x_embed, indexing="ij")

    pos_dim = C // 4
    omega = torch.arange(pos_dim, device=device) / pos_dim
    omega = 1.0 / (10000 ** omega)

    out_y = torch.einsum("hw,d->hwd", yy, omega)
    out_x = torch.einsum("hw,d->hwd", xx, omega)

    pos = torch.cat(
        [
            torch.sin(out_y),
            torch.cos(out_y),
            torch.sin(out_x),
            torch.cos(out_x),
        ],
        dim=-1,
    )

    pos = pos.reshape(H * W, C)
    return pos.unsqueeze(0)  # [1, HW, C]

class DepthFusionModule(nn.Module):
    """
    General fusion module for combining image and depth features.

    Supported modes:
        - add
        - gated_add
        - concat
        - film
        - cross_attn
        - geo_aware
    """

    def __init__(
        self,
        channels=256,
        mode="gated_add",
        num_heads=8,
    ):
        super().__init__()

        self.mode = mode
        self.channels = channels

        # ------------------------
        # Simple modes
        # ------------------------
        if mode == "add":
            pass

        elif mode == "gated_add":
            self.gate = nn.Parameter(torch.zeros(1))

        elif mode == "concat":
            self.proj = nn.Conv2d(2 * channels, channels, kernel_size=1)

        elif mode == "film":
            self.gamma = nn.Conv2d(channels, channels, kernel_size=1)
            self.beta = nn.Conv2d(channels, channels, kernel_size=1)

        elif mode == "cross_attn":
            self.norm_img = nn.LayerNorm(channels)
            self.norm_depth = nn.LayerNorm(channels)

            self.cross_attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                batch_first=True,
            )

            self.proj = nn.Linear(channels, channels)
            
            self.ffn = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, channels * 4),
                nn.GELU(),
                nn.Linear(channels * 4, channels),
            )
        # ------------------------
        # 🔥 Geometry-Aware Mode
        # ------------------------
        elif mode == "geo_aware":

            # 1️⃣ Spatial gate (depth → spatial mask)
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, 1, 1),
                nn.Sigmoid()
            )

            # 2️⃣ Channel gate (SE-style)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            )

            # 3️⃣ Local mixer
            self.local_mixer = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )

            # 4️⃣ Output projection
            self.out_proj = nn.Conv2d(channels, channels, 1)

        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, image_feats, depth_feats):
        """
        image_feats: [B, C, H, W]
        depth_feats: [B, C, H, W]
        """

        # ------------------------------------------------------
        # Simple modes
        # ------------------------------------------------------

        if self.mode == "add":
            return image_feats + depth_feats

        elif self.mode == "gated_add":
            alpha = torch.tanh(self.gate)
            return image_feats + alpha * depth_feats

        elif self.mode == "concat":
            fused = torch.cat([image_feats, depth_feats], dim=1)
            return self.proj(fused)

        elif self.mode == "film":
            gamma = self.gamma(depth_feats)
            beta = self.beta(depth_feats)
            return image_feats * (1 + gamma) + beta

        elif self.mode == "cross_attn":

            B, C, H, W = image_feats.shape
            device = image_feats.device
            dtype = image_feats.dtype  # ← important

            img_tokens = image_feats.flatten(2).transpose(1, 2)
            depth_tokens = depth_feats.flatten(2).transpose(1, 2)

            pos = build_2d_sincos_position_embedding(H, W, C, device).to(dtype)

            img_tokens = img_tokens + pos
            depth_tokens = depth_tokens + pos

            # ---- Cast norms safely ----
            img_norm = self.norm_img(img_tokens.to(self.norm_img.weight.dtype))
            depth_norm = self.norm_depth(depth_tokens.to(self.norm_depth.weight.dtype))

            # Cast back to model dtype
            img_norm = img_norm.to(dtype)
            depth_norm = depth_norm.to(dtype)

            attn_out, _ = self.cross_attn(
                query=img_norm,
                key=depth_norm,
                value=depth_norm,
            )

            attn_out = self.proj(attn_out)

            fused_tokens = img_tokens + attn_out
            fused_tokens = fused_tokens + self.ffn(fused_tokens)

            fused = fused_tokens.transpose(1, 2).reshape(B, C, H, W)

            return fused
        # ------------------------------------------------------
        # 🔥 Geometry-Aware Fusion
        # ------------------------------------------------------

        elif self.mode == "geo_aware":

            # 1️⃣ Spatial modulation
            spatial_alpha = self.spatial_gate(depth_feats)
            img_mod = image_feats * (1 + spatial_alpha)

            # 2️⃣ Channel modulation
            channel_alpha = self.channel_gate(depth_feats)
            img_mod = img_mod * (1 + channel_alpha)

            # 3️⃣ Local geometry mixing
            fused = img_mod + depth_feats
            fused = self.local_mixer(fused)

            # 4️⃣ Residual stabilization
            fused = fused + image_feats

            return self.out_proj(fused)
        
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            #self.geometry_pretrained = kwargs.get("geometry_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            #self.geometry_pretrained = kwargs.get("geometry_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config, geometry_backbone=None, depth_backbone=None, depth_fusion_mode='gated_add'):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        #------Depth Encoder-------
        self.depth_encoder = depth_backbone
        if self.depth_encoder is not None:
            self.depth_fusion = DepthFusionModule(mode=depth_fusion_mode)
        
        # ---- Geometry Encoder ----
        # self.geometry_backbone = geometry_backbone
        # self.geometry_encoder = None
        # self.geometry_cross_attn_layers = None
        # self.num_geom_cross_layers = 4  # last 4 layers

        # if self.geometry_backbone is not None:
        #     for param in self.geometry_backbone.parameters():
        #         param.requires_grad = False

        #     self.geometry_encoder = GeometryEncoder(
        #         backbone=self.geometry_backbone,
        #         out_dim=config.hidden_size,
        #     )

        #     self.geometry_cross_attn_layers = nn.ModuleList([
        #         GatedGeometryCrossAttention(
        #             hidden_size=config.hidden_size,
        #             num_heads=config.num_attention_heads,
        #         )
        #         for _ in range(self.num_geom_cross_layers)
        #     ])


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )    
        else:
            config.mm_vision_tower = config.vision_tower
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        depth_maps: torch.FloatTensor,
        #geometry_points:torch.FloatTensor=None,
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        ###############Depth map insertion###################
        depth_map_embeddings = self.model.depth_encoder(depth_maps)
        image_embeddings_fused = self.model.depth_fusion(image_embeddings, depth_map_embeddings) #or other interesting fusion, shape of image_embeddings:[B,256,64,64]
        #####################################################
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                model_output = output_i
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
            ## Replace the above forward with below
# #-----------------------forward decode with geometry cross attention------------------------            
#             llama_model = self.model.model  # this is LlamaModel

#             inputs_embeds = llama_model.embed_tokens(input_ids)

#             hidden_states = inputs_embeds
#             all_hidden_states = []

#             geom_tokens = None
#             if self.model.geometry_encoder is not None and geometry_points is not None:
#                 geom_tokens = self.model.geometry_encoder(geometry_points)
#                 geom_tokens = geom_tokens.to(hidden_states.dtype)

#             num_layers = len(llama_model.layers)
#             k = self.model.num_geom_cross_layers
#             inject_start = num_layers - k

#             for i, layer in enumerate(llama_model.layers):

#                 hidden_states = layer(
#                     hidden_states,
#                     attention_mask=attention_masks,
#                     position_ids=None,
#                     past_key_value=None,
#                     output_attentions=False,
#                     use_cache=False,
#                 )[0]

#                 # 🔥 TRUE in-layer injection
#                 if geom_tokens is not None and i >= inject_start:
#                     cross_idx = i - inject_start
#                     hidden_states = self.model.geometry_cross_attn_layers[cross_idx](
#                         hidden_states,
#                         geom_tokens
#                     )

#                 all_hidden_states.append(hidden_states)
# #-------------------------------------------------------------------------------------


        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        
        
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        #text_features = output_hidden_states[-1]   # [B, L, D]

# #---------------------------------Geometry cross attention insertion point---------------------------------
#         if (
#             self.model.geometry_encoder is not None
#             and geometry_points is not None
#         ):
#             geom_tokens = self.model.geometry_encoder(geometry_points)

#             k = self.model.num_geom_cross_layers
#             num_layers = len(output_hidden_states)

#             # Start from earliest layer in the fusion range
#             text_features = output_hidden_states[num_layers - k]

#             for i in range(k):
#                 text_features = self.model.geometry_cross_attn_layers[i](
#                     text_features,
#                     geom_tokens
#                 )
# #---------------------------------Geometry cross attention insertion point---------------------------------

#         hidden_states.append(
#             self.model.text_hidden_fcs[0](text_features)
#         )

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None, 
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            ## Depthmap fusion to image and dense_prompt_embeddings (1,256,64,64)
            # dense_embeddings_fused = self.model.depth_fusion(
            #                             dense_embeddings,
            #                             depth_map_embeddings[i].unsqueeze(0))

            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings_fused[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            #pred_ids = torch.argmax(model_output.logits, axis=-1) # (B, seq_len, vocab) -> (B,seq_len)
            
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

#--------------------------------Debugging logs for gated geometry cross attention---------------------------------
        # if self.model.geometry_cross_attn_layers is not None:
        #     for i, layer in enumerate(self.model.geometry_cross_attn_layers):
        #         gate_norm = torch.norm(torch.tanh(layer.alpha)).item()
        #         print(f"[Gate Layer {i}] L2 norm = {gate_norm:.6f}")

        # if self.model.geometry_cross_attn_layers is not None:
        #     for i, layer in enumerate(self.model.geometry_cross_attn_layers):
        #         if layer.alpha.grad is not None:
        #             grad_norm = layer.alpha.grad.norm().item()
        #             print(f"[Gate {i}] grad_norm={grad_norm:.6f}")
#--------------------------------Debugging logs for gated geometry cross attention--------------------------------- 
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
        **kwargs,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks

## Cross attention module insertion

## with point cloud embeddings for spatial context
