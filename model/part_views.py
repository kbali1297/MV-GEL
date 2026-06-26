import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# ---------------------------------------------------
# View Self Attention
# ---------------------------------------------------

class ViewSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, depth=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, view_feats):
        return self.encoder(view_feats)


# ---------------------------------------------------
# Prompt Conditioned View Selector
# ---------------------------------------------------

class PromptConditionedViewSelector(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-large-patch14",
        depth=2,
        unfreeze_last_vision_block=False,
        alpha=0.7
    ):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        dim = self.clip_model.config.projection_dim
        
        self.alpha = alpha
        # Freeze CLIP by default
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Optional: unfreeze last vision block
        # if unfreeze_last_vision_block:
        #     for name, param in self.clip_model.named_parameters():
        #         if "vision_model.encoder.layers.23" in name:
        #             param.requires_grad = True

        # View self-attention
        self.view_self_attn = ViewSelfAttention(dim, num_heads=8, depth=depth)

        # Cross-attention (text attends to views)
        self.cross_attn_tv = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )

        # Cross-attention (views attend to text)
        self.cross_attn_vt = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            batch_first=True
        )

        # Small MLP head for extra capacity
        self.view_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Learnable temperature (CLIP-style)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    def forward(self, images, text_tokens):
        B, V = images.shape[:2]

        # ---------------------------------------------------
        # Encode Images
        # ---------------------------------------------------
        images_flat = images.view(B * V, *images.shape[2:])
        #images_flat = F.interpolate(images_flat, size=(224, 224), mode='bilinear')

        with torch.no_grad():
            image_feats = self.clip_model.get_image_features(images_flat)

        image_feats = image_feats.view(B, V, -1)

        # ---------------------------------------------------
        # Encode Text
        # ---------------------------------------------------
        with torch.no_grad():
            text_feats = self.clip_model.get_text_features(**text_tokens)

        # ---------------------------------------------------
        # View Self-Attention + MLP
        # ---------------------------------------------------
        view_feats = self.view_self_attn(image_feats)
        view_feats = self.view_mlp(view_feats)

        # ---------------------------------------------------
        # Dual Cross Attention
        # ---------------------------------------------------

        text_query = text_feats.unsqueeze(1)  # [B,1,D]

        # 1️⃣ Views attend to text
        v_prime, _ = self.cross_attn_vt(
            query=view_feats,
            key=text_query,
            value=text_query
        )  # [B,V,D]

        # 2️⃣ Text attends to views
        t_prime, _ = self.cross_attn_tv(
            query=text_query,
            key=view_feats,
            value=view_feats
        )  # [B,1,D]

        t_prime = t_prime.squeeze(1)  # [B,D]

        # ---------------------------------------------------
        # Dual Scoring (Correct Form)
        # ---------------------------------------------------

        # Term 1: prompt-conditioned views
        score_v = torch.einsum("bvd,bd->bv", v_prime, text_feats)

        # Term 2: scene-conditioned text
        score_t = torch.einsum("bvd,bd->bv", view_feats, t_prime)

        scores = self.alpha * score_v + (1 - self.alpha) * score_t

        # Temperature scaling
        scores = scores * self.logit_scale.exp()

        return scores

def pairwise_ranking_loss_(scores, gt_ranks):
    """
    scores: [B, V]
    gt_ranks: [B, V]  (higher = better)
    """
    s_i, r_i = scores.unsqueeze(1), gt_ranks.unsqueeze(1) #[B,1,V]
    s_j, r_j = scores.unsqueeze(2), gt_ranks.unsqueeze(2) #[B,V,1]

    loss_mask = (r_i > r_j).float()
    diff = (s_i - s_j)
    
    loss = torch.nn.functional.softplus(-diff) * loss_mask

    return loss.sum()/loss_mask.sum().clamp(min=1.0) #clamp because edge case all views are equally ranked 

def pairwise_ranking_loss(
    scores,
    gt_ranks,
    topk_ratio=0.01,     # top % considered "good"
    margin=0.1,         # ranking margin
):
    """
    scores:   [B, V]
    gt_ranks: [B, V]  (higher = better)
    """

    B, V = scores.shape 
    device = scores.device

    # -------------------------
    # 1️⃣ Determine top-k mask
    # -------------------------
    k = max(1, int(V * topk_ratio))

    # Get threshold rank per batch
    topk_vals, _ = torch.topk(gt_ranks, k, dim=1)
    threshold = topk_vals[:, -1].unsqueeze(1)  # [B,1]

    # Good vs bad masks
    good_mask = gt_ranks >= threshold   # [B,V]
    bad_mask  = gt_ranks < threshold    # [B,V]

    # -------------------------
    # 2️⃣ Pairwise construction
    # -------------------------
    s_i = scores.unsqueeze(2)  # [B,V,1]
    s_j = scores.unsqueeze(1)  # [B,1,V]

    good_i = good_mask.unsqueeze(2)  # [B,V,1]
    bad_j  = bad_mask.unsqueeze(1)   # [B,1,V]

    # Only enforce: good view should rank above bad view
    pair_mask = good_i & bad_j  # [B,V,V]

    # -------------------------
    # 3️⃣ Margin ranking loss
    # -------------------------
    diff = s_i - s_j  # want positive

    # softplus for smooth hinge
    loss = F.softplus(-(diff - margin)) * pair_mask.float()

    # -------------------------
    # 4️⃣ Normalize safely
    # -------------------------
    denom = pair_mask.sum().clamp(min=1.0)
    return loss.sum() / denom

import torch
import torch.nn as nn
from transformers import CLIPModel

class GeometricViewAdapter(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        num_heads=4,
        max_views=128,
        adapter_dim=128, 
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # FREEZE EVERYTHING IN CLIP
        for p in self.clip.parameters():
            p.requires_grad = False
            
        self.vision_dim = self.clip.vision_model.config.hidden_size # 768
        self.text_dim = self.clip.text_model.config.hidden_size     # 512
        
        # 1. Projections to Adapter Space
        self.geo_vis_proj = nn.Sequential(
            nn.Linear(self.vision_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim)
        )

        self.geo_txt_proj = nn.Sequential(
            nn.Linear(self.text_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim)
        )

        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_views, adapter_dim) * 0.02)

        # ---------------------------------------------------------
        # 3. NEW: Non-Linear Fusion Layer (Replaces aggressive Sigmoid)
        # ---------------------------------------------------------
        # Input size is adapter_dim * 2 because we concat (Vis, Txt)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(adapter_dim * 2, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, adapter_dim)
            # No sigmoid here. We want a feature shift, not a gate.
        )

        # 4. Geometric Interaction
        self.geo_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=adapter_dim, nhead=num_heads, batch_first=True, norm_first=True),
            num_layers=2
        )
        
        # 5. Final Scoring
        self.geo_score_head = nn.Linear(adapter_dim, 1)

    def forward(self, images, text_tokens):
        B, V = images.shape[:2]
        
        images_flat = images.view(B * V, *images.shape[2:])
        
        # --- A. Extract Raw CLIP Features ---
        with torch.no_grad():
            # Get raw [CLS] tokens BEFORE projection
            vis_out = self.clip.vision_model(pixel_values=images_flat)
            raw_vis_feats = vis_out.pooler_output # [B*V, 768]
            
            txt_out = self.clip.text_model(**text_tokens)
            raw_txt_feats = txt_out.pooler_output # [B, 512]

            # Standard Zero-Shot Scores (Baseline)
            semantic_vis = self.clip.visual_projection(raw_vis_feats).view(B, V, -1)
            semantic_txt = self.clip.text_projection(raw_txt_feats).unsqueeze(1)
            
            semantic_vis = semantic_vis / semantic_vis.norm(dim=-1, keepdim=True)
            semantic_txt = semantic_txt / semantic_txt.norm(dim=-1, keepdim=True)
            
            base_scores = (semantic_vis * semantic_txt).sum(dim=-1)

        # --- B. Geometric Adapter Path ---
        
        # 1. Project
        geo_vis = self.geo_vis_proj(raw_vis_feats).view(B, V, -1) # [B, V, 128]
        geo_txt = self.geo_txt_proj(raw_txt_feats).unsqueeze(1)   # [B, 1, 128]
        
        # 2. Add Positional Embeddings
        geo_vis = geo_vis + self.pos_embed[:, :V, :]
        
        # ---------------------------------------------------------
        # 3. NEW: Non-Linear Fusion Logic
        # ---------------------------------------------------------
        # Expand text to match views: [B, 1, 128] -> [B, V, 128]
        txt_expanded = geo_txt.expand(-1, V, -1)
        
        # Concatenate: [B, V, 256]
        combined = torch.cat([geo_vis, txt_expanded], dim=-1)
        
        # Pass through MLP to learn the interaction
        modulation = self.fusion_mlp(combined)
        
        # Residual Connection:
        # Instead of replacing geo_vis, we ADD the text-conditioned adjustment.
        # This guarantees that even if the MLP outputs garbage initially, 
        # the original geometric features are preserved.
        geo_vis = geo_vis + modulation 
        
        # ---------------------------------------------------------

        # 4. View Interaction (Transformer)
        geo_vis = self.geo_transformer(geo_vis)
        
        # 5. Score
        geo_adjustment = self.geo_score_head(geo_vis).squeeze(-1)
        
        # Combine
        final_scores = base_scores + geo_adjustment
        
        return final_scores * 10.0


from peft import get_peft_model, LoraConfig, TaskType

class LoraCLIPViewSelector_simple(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_heads=8,
        view_layers=2,
        max_views=128,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # 1. Load CLIP
        # ---------------------------------------------------------
        clip = CLIPModel.from_pretrained(clip_model_name)

        # ---------------------------------------------------------
        # 2. Configure LoRA (modify attention patterns only)
        # ---------------------------------------------------------
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],  # attention layers
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None
        )

        self.clip = get_peft_model(clip, peft_config)

        # Projection dimension (aligned embedding space)
        self.embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/16

        # ---------------------------------------------------------
        # 3. View Positional Embeddings (Spiral Awareness)
        # ---------------------------------------------------------
        self.view_pos_embed = nn.Parameter(
            torch.randn(1, max_views, self.embed_dim) * 0.02
        )

        # ---------------------------------------------------------
        # 4. View Interaction Transformer
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )

        self.view_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=view_layers
        )

        # ---------------------------------------------------------
        # 5. Geometric Adjustment Head
        # ---------------------------------------------------------
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1)
        )

        # Learnable temperature (CLIP-style scaling)
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

        # Optional: print trainable params to confirm LoRA works
        self.clip.print_trainable_parameters()

    def forward(self, images, text_tokens):
        """
        images: [B, V, 3, H, W]
        text_tokens: dict with input_ids, attention_mask
        """

        B, V = images.shape[:2]
        images_flat = images.view(B * V, *images.shape[2:])

        # ---------------------------------------------------------
        # 1. LoRA-Adapted Aligned CLIP Features
        # ---------------------------------------------------------
        img_feats = self.clip.get_image_features(pixel_values=images_flat)
        txt_feats = self.clip.get_text_features(**text_tokens)

        # Normalize (important for stability)
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)

        # Reshape
        img_feats = img_feats.view(B, V, -1)   # [B, V, 512]
        txt_feats = txt_feats.unsqueeze(1)     # [B, 1, 512]

        # ---------------------------------------------------------
        # 2. Base Semantic Similarity (CLIP prior)
        # ---------------------------------------------------------
        base_scores = (img_feats * txt_feats).sum(dim=-1)  # [B, V]

        return base_scores

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model


class ModalityFusion_CLS(nn.Module):
    """
    Ablation block testing fusion mechanisms using ONLY the global [CLS] text token.
    """
    def __init__(self, embed_dim, fusion_type="add", num_heads=8):
        super().__init__()
        self.fusion_type = fusion_type.lower() if fusion_type is not None else None
        
        if self.fusion_type == "add":
            pass # No parameters needed
            
        elif self.fusion_type == "gated_add":
            # Initialize to 0. At step 0, it acts exactly like frozen CLIP.
            self.gate = nn.Parameter(torch.zeros(1))
            
        elif self.fusion_type == "film" or self.fusion_type == "film_no_clip":
            # Projects 512D text to 1024D (512 for gamma/scale, 512 for beta/shift)
            self.film_proj = nn.Linear(embed_dim, embed_dim * 2)
            # Initialize to 0 so gamma=0, beta=0 at start (identity transformation)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)
            
        elif self.fusion_type == "cross_attention" or self.fusion_type == 'cross_attention_no_clip':
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm_img = nn.LayerNorm(embed_dim)
            self.norm_txt = nn.LayerNorm(embed_dim)
            # Gate initialized to 0 for stability
            self.ca_gate = nn.Parameter(torch.zeros(1))
            
        else: pass
        #raise ValueError(f"Unknown fusion_type: {fusion_type}. Choose from: add, gated_add, film, cross_attention.")

    def forward(self, img_feats, txt_cls):
        """
        img_feats:[B, V, D]
        txt_cls:   [B, 1, D] (The pooled [CLS] text token)
        """
        if self.fusion_type == "add":
            return img_feats + txt_cls
            
        elif self.fusion_type == "gated_add":
            return img_feats + (self.gate * txt_cls)
            
        elif self.fusion_type == "film" or self.fusion_type == "film_no_clip":
            # Generate scale (gamma) and shift (beta) from the single text token
            film_params = self.film_proj(txt_cls)  #[B, 1, 2*D]
            gamma, beta = film_params.chunk(2, dim=-1)
            # Apply feature-wise linear modulation
            return img_feats * (1 + gamma) + beta
            
        elif self.fusion_type == "cross_attention" or self.fusion_type == 'cross_attention_no_clip':
            norm_img = self.norm_img(img_feats)
            norm_txt = self.norm_txt(txt_cls)
            
            # Query = Image Views [B, V, D], Key/Value = Text CLS [B, 1, D]
            # Note: Softmax across Key length of 1 will equal 1.0. 
            # This tests if the linear projections (W_q, W_k, W_v) offer an advantage.
            attn_out, _ = self.cross_attn(
                query=norm_img,
                key=norm_txt,
                value=norm_txt
            )
            return img_feats + (self.ca_gate * attn_out)


class LoraCLIPViewSelector_Ablation(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        fusion_type=None,  # CHANGE THIS FOR ABLATIONS: "add", "gated_add", "film", "cross_attention", "no_fusion", "only_clip"
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        num_heads=8,
        view_layers=2,
        max_views=128,
    ):
        super().__init__()
        self.fusion_type = fusion_type.lower() if fusion_type is not None else None

        # ---------------------------------------------------------
        # 1. Load CLIP & Configure LoRA
        # ---------------------------------------------------------
        clip = CLIPModel.from_pretrained(clip_model_name)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None
        )

        self.clip = get_peft_model(clip, peft_config)
        self.embed_dim = self.clip.config.projection_dim

        # ---------------------------------------------------------
        # 2. View Positional Embeddings
        # ---------------------------------------------------------
        self.view_pos_embed = nn.Parameter(torch.randn(1, max_views, self.embed_dim) * 0.02)

        # ---------------------------------------------------------
        # 3. Modality Fusion Ablation Module
        # ---------------------------------------------------------
        if self.fusion_type not in ["no_fusion", "only_clip", None]:
            self.modality_fusion = ModalityFusion_CLS(
                embed_dim=self.embed_dim, 
                fusion_type=self.fusion_type, 
                num_heads=num_heads
            )

        # ---------------------------------------------------------
        # 4. View Interaction Transformer
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=self.embed_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=view_layers)

        # ---------------------------------------------------------
        # 5. Geometric Adjustment Head
        # ---------------------------------------------------------
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

    def forward(self, images, text_tokens):
        """
        images:[B, V, 3, H, W]
        text_tokens: dict with input_ids, attention_mask
        """
        B, V = images.shape[:2]
        images_flat = images.view(B * V, *images.shape[2:])

        # ---------------------------------------------------------
        # 1. Feature Extraction (Strictly Global Tokens)
        # ---------------------------------------------------------
        img_feats = self.clip.get_image_features(pixel_values=images_flat)  # [B*V, D]
        txt_feats = self.clip.get_text_features(**text_tokens)              # [B, D]

        # ---------------------------------------------------------
        # 2. Base Semantic Similarity (CLIP Prior)
        # ---------------------------------------------------------
        img_norm = F.normalize(img_feats, dim=-1).view(B, V, -1)  #[B, V, D]
        txt_norm = F.normalize(txt_feats, dim=-1).unsqueeze(1)    # [B, 1, D]

        similarity_scores = (img_norm * txt_norm).sum(-1)  # [B, V]

        if self.fusion_type == 'only_clip':
            return similarity_scores
        # ---------------------------------------------------------
        # 3. Inject View Positional Context
        # ---------------------------------------------------------
        # Reshape images back to multi-view format
        img_feats = img_feats.view(B, V, -1)
        img_feats = img_feats + self.view_pos_embed[:, :V, :]

        # ---------------------------------------------------------
        # 4. Modality Fusion (The Ablation Layer)
        # ---------------------------------------------------------
        # Feed strictly the [CLS] text token [B, 1, D] into the chosen fusion block
        if self.fusion_type not in [None, "no_fusion"]:
            img_feats = self.modality_fusion(
                img_feats=img_feats,
                txt_cls=txt_feats.unsqueeze(1)
            )
        

        # ---------------------------------------------------------
        # 5. View-to-View Context & Geometric Routing
        # ---------------------------------------------------------
        context_feats = self.view_transformer(img_feats)
        geo_scores = self.score_head(context_feats).squeeze(-1)  # [B, V]

        # ---------------------------------------------------------
        # 6. Final Score Calculation
        # ---------------------------------------------------------

        if self.fusion_type.endswith('no_clip'):
            total_score = geo_scores
            print("No clip variant!! Not adding similarity scores!!") 
        else:
            total_score = geo_scores + similarity_scores
        return total_score * self.logit_scale.exp()
    





























#class LoraCLIPViewSelector(nn.Module):
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch16",
#         lora_r=8,
#         lora_alpha=16,
#         lora_dropout=0.1,
#         num_heads=8,
#         view_layers=2,
#         max_views=128,
#     ):
#         super().__init__()

#         # ---------------------------------------------------------
#         # 1. Load CLIP
#         # ---------------------------------------------------------
#         clip = CLIPModel.from_pretrained(clip_model_name)

#         # ---------------------------------------------------------
#         # 2. Configure LoRA (modify attention patterns only)
#         # ---------------------------------------------------------
#         peft_config = LoraConfig(
#             r=lora_r,
#             lora_alpha=lora_alpha,
#             target_modules=["q_proj", "v_proj"],  # attention layers
#             lora_dropout=lora_dropout,
#             bias="none",
#             task_type=None
#         )

#         self.clip = get_peft_model(clip, peft_config)

#         # Projection dimension (aligned embedding space)
#         self.embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/16

#         # ---------------------------------------------------------
#         # 3. View Positional Embeddings (Spiral Awareness)
#         # ---------------------------------------------------------
#         self.view_pos_embed = nn.Parameter(
#             torch.randn(1, max_views, self.embed_dim) * 0.02
#         )

#         # ---------------------------------------------------------
#         # 4. View Interaction Transformer
#         # ---------------------------------------------------------
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=num_heads,
#             dim_feedforward=self.embed_dim * 4,
#             batch_first=True,
#             norm_first=True,
#             dropout=0.1,
#         )

#         self.view_transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=view_layers
#         )

#         # ---------------------------------------------------------
#         # 5. Geometric Adjustment Head
#         # ---------------------------------------------------------
#         self.score_head = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, 1)
#         )

#         # Learnable temperature (CLIP-style scaling)
#         self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

#         # Optional: print trainable params to confirm LoRA works
#         self.clip.print_trainable_parameters()

#     def forward(self, images, text_tokens):
#         """
#         images: [B, V, 3, H, W]
#         text_tokens: dict with input_ids, attention_mask
#         """

#         B, V = images.shape[:2]
#         images_flat = images.view(B * V, *images.shape[2:])

#         # ---------------------------------------------------------
#         # 1. LoRA-Adapted Aligned CLIP Features
#         # ---------------------------------------------------------
#         img_feats = self.clip.get_image_features(pixel_values=images_flat)
#         txt_feats = self.clip.get_text_features(**text_tokens)

#         # Normalize (important for stability)
#         img_feats = F.normalize(img_feats, dim=-1)
#         txt_feats = F.normalize(txt_feats, dim=-1)

#         # Reshape
#         img_feats = img_feats.view(B, V, -1)   # [B, V, 512]
#         txt_feats = txt_feats.unsqueeze(1)     # [B, 1, 512]

#         # ---------------------------------------------------------
#         # 2. Base Semantic Similarity (CLIP prior)
#         # ---------------------------------------------------------
#         base_scores = (img_feats * txt_feats).sum(dim=-1)  # [B, V]

#         # ---------------------------------------------------------
#         # 3. Inject View Positional Context
#         # ---------------------------------------------------------
#         img_feats = img_feats + self.view_pos_embed[:, :V, :]

#         # ---------------------------------------------------------
#         # 4. Condition Views on Text (Residual Conditioning)
#         # ---------------------------------------------------------
#         img_feats = img_feats + txt_feats

#         # ---------------------------------------------------------
#         # 5. View-to-View Interaction
#         # ---------------------------------------------------------
#         context_feats = self.view_transformer(img_feats)

#         # ---------------------------------------------------------
#         # 6. Geometric Adjustment
#         # ---------------------------------------------------------
#         geo_scores = self.score_head(context_feats).squeeze(-1)  # [B, V]

#         # ---------------------------------------------------------
#         # 7. Combine Semantic + Geometric
#         # ---------------------------------------------------------
#         final_scores = base_scores + geo_scores

#         return final_scores * self.logit_scale.exp()

# class CLIPViewSelector(nn.Module):
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch16",
#         num_heads=8,
#         view_layers=2,
#         max_views=128,
#     ):
#         super().__init__()

#         # ---------------------------------------------------------
#         # 1. Load CLIP
#         # ---------------------------------------------------------
      
#         # 1. Load CLIP
#         self.clip = CLIPModel.from_pretrained(clip_model_name)
        
#         # Freeze CLIP
#         for p in self.clip.parameters():
#             p.requires_grad = False

#         # Projection dimension (aligned embedding space)
#         self.embed_dim = self.clip.config.projection_dim  # 512 for ViT-B/16

#         # ---------------------------------------------------------
#         # 3. View Positional Embeddings (Spiral Awareness)
#         # ---------------------------------------------------------
#         self.view_pos_embed = nn.Parameter(
#             torch.randn(1, max_views, self.embed_dim) * 0.02
#         )

#         # ---------------------------------------------------------
#         # 4. View Interaction Transformer
#         # ---------------------------------------------------------
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=num_heads,
#             dim_feedforward=self.embed_dim * 4,
#             batch_first=True,
#             norm_first=True,
#             dropout=0.1,
#         )

#         self.view_transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=view_layers
#         )

#         # ---------------------------------------------------------
#         # 5. Geometric Adjustment Head
#         # ---------------------------------------------------------
#         self.score_head = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, 1)
#         )

#         # Learnable temperature (CLIP-style scaling)
#         self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)

#     def forward(self, images, text_tokens):
#         """
#         images: [B, V, 3, H, W]
#         text_tokens: dict with input_ids, attention_mask
#         """

#         B, V = images.shape[:2]
#         images_flat = images.view(B * V, *images.shape[2:])

#         # ---------------------------------------------------------
#         # 1. LoRA-Adapted Aligned CLIP Features
#         # ---------------------------------------------------------
#         img_feats = self.clip.get_image_features(pixel_values=images_flat)
#         txt_feats = self.clip.get_text_features(**text_tokens)

#         # Normalize (important for stability)
#         img_feats = F.normalize(img_feats, dim=-1)
#         txt_feats = F.normalize(txt_feats, dim=-1)

#         # Reshape
#         img_feats = img_feats.view(B, V, -1)   # [B, V, 512]
#         txt_feats = txt_feats.unsqueeze(1)     # [B, 1, 512]

#         # ---------------------------------------------------------
#         # 2. Base Semantic Similarity (CLIP prior)
#         # ---------------------------------------------------------
#         base_scores = (img_feats * txt_feats).sum(dim=-1)  # [B, V]

#         # ---------------------------------------------------------
#         # 3. Inject View Positional Context
#         # ---------------------------------------------------------
#         img_feats = img_feats + self.view_pos_embed[:, :V, :]

#         # ---------------------------------------------------------
#         # 4. Condition Views on Text (Residual Conditioning)
#         # ---------------------------------------------------------
#         img_feats = img_feats + txt_feats

#         # ---------------------------------------------------------
#         # 5. View-to-View Interaction
#         # ---------------------------------------------------------
#         context_feats = self.view_transformer(img_feats)

#         # ---------------------------------------------------------
#         # 6. Geometric Adjustment
#         # ---------------------------------------------------------
#         geo_scores = self.score_head(context_feats).squeeze(-1)  # [B, V]

#         # ---------------------------------------------------------
#         # 7. Combine Semantic + Geometric
#         # ---------------------------------------------------------
#         final_scores = base_scores + geo_scores

#         return final_scores * self.logit_scale.exp()


# class BiDirectionalPatchViewSelector(nn.Module):
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch16",
#         num_heads=8,
#         view_layers=2,
#         max_views=128,   # must be >= max V in dataset
#         unfreeze_last_block=False,
#         alpha=0.5,
#     ):
#         super().__init__()

#         self.clip = CLIPModel.from_pretrained(clip_model_name)

#         self.embed_dim = self.clip.config.vision_config.hidden_size
#         self.text_dim = self.clip.config.text_config.hidden_size
#         self.alpha = alpha

#         # Freeze CLIP
#         for p in self.clip.parameters():
#             p.requires_grad = False

#         if unfreeze_last_block:
#             for name, param in self.clip.named_parameters():
#                 if "vision_model.encoder.layers.11" in name:
#                     param.requires_grad = True

#         # Text projection
#         if self.text_dim != self.embed_dim:
#             self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
#         else:
#             self.text_proj = nn.Identity()

#         # Cross Attention
#         self.cross_attn_t2v = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )

#         self.cross_attn_v2t = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )

#         # 🔥 View positional embedding (CRITICAL FIX)
#         self.view_pos_embed = nn.Parameter(
#             torch.randn(1, max_views, self.embed_dim)
#         )

#         # View-level transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=num_heads,
#             batch_first=True,
#             norm_first=True
#         )

#         self.view_transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=view_layers
#         )

#         # Scoring head
#         self.view_mlp = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.embed_dim, 1)
#         )

#         self.logit_scale = nn.Parameter(
#             torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
#         )

#     def forward(self, images, text_tokens):
#         """
#         images: [B, V, 3, 224, 224]
#         """

#         B, V = images.shape[:2]

#         # --------------------------------------------------
#         # Vision Encoding
#         # --------------------------------------------------
#         images_flat = images.view(B * V, *images.shape[2:])
#         vision_outputs = self.clip.vision_model(pixel_values=images_flat)

#         patch_tokens = vision_outputs.last_hidden_state  # [B*V,197,D]
#         patch_tokens = patch_tokens[:, 1:, :]            # remove CLS
#         patch_tokens = patch_tokens.view(B, V, 196, self.embed_dim)

#         # --------------------------------------------------
#         # Text Encoding
#         # --------------------------------------------------
#         text_outputs = self.clip.text_model(**text_tokens)
#         text_cls = text_outputs.last_hidden_state[:, 0, :]
#         text_embed = self.text_proj(text_cls)  # [B,D]

#         # --------------------------------------------------
#         # Flatten views for cross attention
#         # --------------------------------------------------
#         patch_tokens = patch_tokens.view(B * V, 196, self.embed_dim)

#         text_query = text_embed.unsqueeze(1)  # [B,1,D]
#         text_query = text_query.repeat_interleave(V, dim=0)  # [B*V,1,D]

#         # Text → Patch
#         t2v_out, _ = self.cross_attn_t2v(
#             query=text_query,
#             key=patch_tokens,
#             value=patch_tokens
#         )
#         t2v_out = t2v_out.squeeze(1)  # [B*V,D]

#         # Patch → Text
#         v2t_out, _ = self.cross_attn_v2t(
#             query=patch_tokens,
#             key=text_query,
#             value=text_query
#         )
#         v2t_out = v2t_out.mean(dim=1)  # [B*V,D]

#         # --------------------------------------------------
#         # Fuse
#         # --------------------------------------------------
#         fused = self.alpha * t2v_out + (1 - self.alpha) * v2t_out

#         # --------------------------------------------------
#         # Restore view dimension
#         # --------------------------------------------------
#         fused = fused.view(B, V, self.embed_dim)

#         # 🔥 ADD POSITIONAL EMBEDDING
#         fused = fused + self.view_pos_embed[:, :V, :]

#         # View self-attention across views
#         fused = self.view_transformer(fused)

#         # --------------------------------------------------
#         # 🔥 Direct text-view similarity path (important)
#         # --------------------------------------------------
#         sim_scores = torch.einsum("bvd,bd->bv", fused, text_embed)

#         # --------------------------------------------------
#         # MLP scoring
#         # --------------------------------------------------
#         mlp_scores = self.view_mlp(fused).squeeze(-1)

#         # Combine both
#         view_scores = sim_scores + mlp_scores

#         view_scores = view_scores * self.logit_scale.exp()

#         return view_scores

# class BiDirectionalPatchViewSelector(nn.Module):
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch16",
#         num_heads=8,
#         view_layers=2,
#         max_views=128,   # must be >= max V in dataset
#         unfreeze_last_block=False,
#         alpha=0.5,
#     ):
#         super().__init__()

#         self.clip = CLIPModel.from_pretrained(clip_model_name)

#         self.embed_dim = self.clip.config.vision_config.hidden_size
#         self.text_dim = self.clip.config.text_config.hidden_size
#         self.alpha = alpha

#         # Freeze CLIP
#         for p in self.clip.parameters():
#             p.requires_grad = False

#         if unfreeze_last_block:
#             for name, param in self.clip.named_parameters():
#                 if "vision_model.encoder.layers.11" in name:
#                     param.requires_grad = True

#         # Text projection
#         if self.text_dim != self.embed_dim:
#             self.text_proj = nn.Linear(self.text_dim, self.embed_dim)
#         else:
#             self.text_proj = nn.Identity()

#         # Cross Attention
#         self.cross_attn_t2v = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )

#         self.cross_attn_v2t = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=num_heads,
#             batch_first=True
#         )

#         # 🔥 View positional embedding (CRITICAL FIX)
#         self.view_pos_embed = nn.Parameter(
#             torch.randn(1, max_views, self.embed_dim)
#         )

#         # View-level transformer
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim,
#             nhead=num_heads,
#             batch_first=True,
#             norm_first=True
#         )

#         self.view_transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=view_layers
#         )

#         # Scoring head
#         self.view_mlp = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.embed_dim, 1)
#         )

#         self.logit_scale = nn.Parameter(
#             torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
#         )

#     def forward(self, images, text_tokens):
#         """
#         images: [B, V, 3, 224, 224]
#         """

#         B, V = images.shape[:2]

#         # --------------------------------------------------
#         # Vision Encoding
#         # --------------------------------------------------
#         images_flat = images.view(B * V, *images.shape[2:])
#         vision_outputs = self.clip.vision_model(pixel_values=images_flat)

#         patch_tokens = vision_outputs.last_hidden_state  # [B*V,197,D]
#         patch_tokens = patch_tokens[:, 1:, :]            # remove CLS
#         patch_tokens = patch_tokens.view(B, V, 196, self.embed_dim)

#         # --------------------------------------------------
#         # Text Encoding
#         # --------------------------------------------------
#         text_outputs = self.clip.text_model(**text_tokens)
#         text_cls = text_outputs.last_hidden_state[:, 0, :]
#         text_embed = self.text_proj(text_cls)  # [B,D]

#         # --------------------------------------------------
#         # Flatten views for cross attention
#         # --------------------------------------------------
#         patch_tokens = patch_tokens.view(B * V, 196, self.embed_dim)

#         text_query = text_embed.unsqueeze(1)  # [B,1,D]
#         text_query = text_query.repeat_interleave(V, dim=0)  # [B*V,1,D]

#         # Text → Patch
#         t2v_out, _ = self.cross_attn_t2v(
#             query=text_query,
#             key=patch_tokens,
#             value=patch_tokens
#         )
#         t2v_out = t2v_out.squeeze(1)  # [B*V,D]

#         # Patch → Text
#         v2t_out, _ = self.cross_attn_v2t(
#             query=patch_tokens,
#             key=text_query,
#             value=text_query
#         )
#         v2t_out = v2t_out.mean(dim=1)  # [B*V,D]

#         # --------------------------------------------------
#         # Fuse
#         # --------------------------------------------------
#         fused = self.alpha * t2v_out + (1 - self.alpha) * v2t_out

#         # --------------------------------------------------
#         # Restore view dimension
#         # --------------------------------------------------
#         fused = fused.view(B, V, self.embed_dim)

#         # 🔥 ADD POSITIONAL EMBEDDING
#         fused = fused + self.view_pos_embed[:, :V, :]

#         # View self-attention across views
#         fused = self.view_transformer(fused)

#         # --------------------------------------------------
#         # 🔥 Direct text-view similarity path (important)
#         # --------------------------------------------------
#         sim_scores = torch.einsum("bvd,bd->bv", fused, text_embed)

#         # --------------------------------------------------
#         # MLP scoring
#         # --------------------------------------------------
#         mlp_scores = self.view_mlp(fused).squeeze(-1)

#         # Combine both
#         view_scores = sim_scores + mlp_scores

#         view_scores = view_scores * self.logit_scale.exp()

#         return view_scores

# class Simple_ViewSelector(nn.Module):
#     def __init__(
#         self,
#         clip_model_name="openai/clip-vit-base-patch16",
#         num_heads=8,
#         view_layers=2,
#         dropout=0.1,
#         unfreeze_last_block = False,
#         max_views=128, # Set >= your max sequence length
#     ):
#         super().__init__()

#         # 1. Load CLIP
#         self.clip = CLIPModel.from_pretrained(clip_model_name)
        
#         # We use the projection_dim (the shared latent space dimension)
#         # For ViT-B/16, this is usually 512. For Large, 768.
#         self.embed_dim = self.clip.config.projection_dim 
        
#         # Freeze CLIP
#         for p in self.clip.parameters():
#             p.requires_grad = False

#         if unfreeze_last_block:
#             last_text_block = self.clip.text_model.encoder.layers[-1]
#             last_vision_block = self.clip.vision_model.encoder.layers[-1]
#             for name, param in last_vision_block.named_parameters():
#                 param.requires_grad = True
#             for name, param in last_text_block.named_parameters():
#                 param.requires_grad = True

#             for param in self.clip.text_model.final_layer_norm.parameters():
#                 param.requires_grad = True

#         # 2. Positional Embeddings for Views (The "Spiral" Context)
#         # The model needs to know: "This is View #0", "This is View #1"
#         self.view_pos_embed = nn.Parameter(torch.randn(1, max_views, self.embed_dim) * 0.02)

#         # 3. Text-to-View Fusion
#         # Since Image and Text are already aligned, we can simply concat or add.
#         # But a CrossAttention layer allows the Text to weigh specific features of the Views.
#         self.cross_attn_t2v = nn.MultiheadAttention(
#             embed_dim=self.embed_dim,
#             num_heads=num_heads,
#             batch_first=True,
#             dropout=dropout
#         )

#         # 4. View Transformer (Context Awareness)
#         # Allows the model to say "If View 0 is bad, and I am View 1, I might be okay."
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.embed_dim, 
#             nhead=num_heads, 
#             batch_first=True, 
#             dim_feedforward=self.embed_dim * 4,
#             dropout=dropout,
#             norm_first=True
#         )
#         self.view_transformer = nn.TransformerEncoder(encoder_layer, num_layers=view_layers)

#         # 5. Scoring Head
#         self.score_head = nn.Sequential(
#             nn.LayerNorm(self.embed_dim),
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.GELU(),
#             nn.Linear(self.embed_dim, 1) # Output scalar score
#         )
        
#         # Standard CLIP logit scale
#         self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052) 

#     def forward(self, images, text_tokens):
#         """
#         images: [B, V, 3, H, W]
#         text_tokens: dict of input_ids, attention_mask
#         """
#         B, V = images.shape[:2]

#         # -----------------------------------------------------------
#         # 1. Get Aligned Global Features (The "Correct" CLIP usage)
#         # -----------------------------------------------------------
        
#         # Flatten images to [B*V, 3, H, W]
#         images_flat = images.view(B * V, *images.shape[2:])
        
#         #with torch.no_grad():
#         # This returns the PROJECTED features (e.g., 512 dim)
#         # These are chemically aligned with text.
#         img_feats = self.clip.get_image_features(images_flat) # [B*V, D]
#         txt_feats = self.clip.get_text_features(**text_tokens) # [B, D]

#         # Reshape images back to [B, V, D]
#         img_feats = img_feats.view(B, V, -1)
        
#         # Normalize features (Standard CLIP practice)
#         # img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
#         # txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

#         # -----------------------------------------------------------
#         # 2. Add Positional Embeddings (The "Spiral" Logic)
#         # -----------------------------------------------------------
#         # This injects the knowledge that Index 0 is "Top" (or whatever your spiral start is)
#         img_feats = img_feats + self.view_pos_embed[:, :V, :]

#         # -----------------------------------------------------------
#         # 3. Fuse Text Context into Views
#         # -----------------------------------------------------------
#         # Prepare text for broadcasting: [B, 1, D]
#         txt_feats_expanded = txt_feats.unsqueeze(1) 

#         # Option A: Residual Addition (Simplest, often works best for aligned feats)
#         # img_feats = img_feats + txt_feats_expanded
        
#         # Option B: Cross Attention (More expressive)
#         # Query = Image Views (We want to update view info based on text)
#         # Key/Val = Text
#         attn_out, _ = self.cross_attn_t2v(
#             query=img_feats,
#             key=txt_feats_expanded,
#             value=txt_feats_expanded
#         )
#         # Residual connection
#         img_feats = img_feats + attn_out

#         # -----------------------------------------------------------
#         # 4. View-to-View Context (Transformer)
#         # -----------------------------------------------------------
#         # Now the views interact. 
#         # "I am the view at Index 5, I see the Text is 'Top View', and my neighbor Index 0 looks very 'Top-ish'."
#         view_context = self.view_transformer(img_feats)

#         # -----------------------------------------------------------
#         # 5. Score
#         # -----------------------------------------------------------
#         scores = self.score_head(view_context).squeeze(-1) # [B, V]
        
#         # Scale by temperature (optional, helps gradients)
#         scores = scores * self.logit_scale.exp()

#         return scores


# def listwise_loss(scores, gt_ranks, temperature=1.0):
#     target_probs = F.softmax(gt_ranks / temperature, dim=-1)
#     pred_log_probs = F.log_softmax(scores, dim=-1)
#     return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")