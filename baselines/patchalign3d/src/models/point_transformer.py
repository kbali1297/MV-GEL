import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN


def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        """
        xyz: (B, N, C) where C>=3 (xyz | [extra])
        Returns:
          neighborhood : (B, G, M, C')
          center       : (B, G, 3)
        """
        batch_size, num_points, C = xyz.shape
        if C > 3:
            data = xyz
            xyz_only = data[:, :, :3].contiguous()
            extra = data[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None
        center = fps(xyz_only, self.num_group)  # (B, G, 3)
        _, idx = self.knn(xyz_only, center)     # (B, G, M)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)
        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :].view(batch_size, self.num_group, self.group_size, 3)
        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :].view(batch_size, self.num_group, self.group_size, -1)
            neighborhood = torch.cat((neigh_xyz - center.unsqueeze(2), neigh_extra), dim=-1)
        else:
            neighborhood = neigh_xyz - center.unsqueeze(2)
        return neighborhood, center


class PatchedGroup(nn.Module):
    """Same as Group but also returns patch membership indices."""

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, C = xyz.shape
        if C > 3:
            data = xyz
            xyz_only = data[:, :, :3].contiguous()
            extra = data[:, :, 3:].contiguous()
        else:
            xyz_only = xyz.contiguous()
            extra = None

        center = fps(xyz_only, self.num_group)  # (B, G, 3)
        _, idx = self.knn(xyz_only, center)     # (B, G, M)
        idx_rel = idx.clone()
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)
        neigh_xyz = xyz_only.view(batch_size * num_points, -1)[idx_flat, :].view(batch_size, self.num_group, self.group_size, 3)

        if extra is not None:
            neigh_extra = extra.view(batch_size * num_points, -1)[idx_flat, :].view(batch_size, self.num_group, self.group_size, -1)
            neighborhood = torch.cat((neigh_xyz - center.unsqueeze(2), neigh_extra), dim=-1)
        else:
            neighborhood = neigh_xyz - center.unsqueeze(2)
        return neighborhood.contiguous(), center.contiguous(), idx_rel


class Encoder(nn.Module):

    def __init__(self, encoder_channel, color=False):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6 if color else 3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c).permute(0, 2, 1)  # (B*G, C, N)
        feature = self.first_conv(point_groups)
        feature_global = torch.max(feature, 2, keepdim=True)[0]
        feature_global = feature_global.repeat(1, 1, n)
        feature = torch.cat([feature_global, feature], 1)
        feature = self.second_conv(feature)
        feature = feature.max(dim=2)[0]  # (B*G, encoder_channel)
        feature = feature.reshape(bs, g, self.encoder_channel).contiguous()
        return feature


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0):
        super().__init__()
        def _drop_for_block(i):
            if isinstance(drop_path_rate, (list, tuple)):
                return drop_path_rate[i]
            return drop_path_rate
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=_drop_for_block(i),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        for i, blk in enumerate(self.blocks):
            x = blk(x + pos)
        return x


class get_model(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        self.color = getattr(config, "color", False)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = PatchedGroup(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, color=self.color)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth, drop_path_rate=dpr, num_heads=self.num_heads)
        self.norm = nn.LayerNorm(self.trans_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_patches(self, pts):
        """
        Args:
            pts: (B, C, N) with C>=3 (xyz | [extra])
        Returns:
            patch_emb    : (B, trans_dim, G)
            patch_centers: (B, 3, G)
            patch_idx    : (B, G, M)
        """
        B, C, N = pts.shape
        pts_bn = pts.transpose(-1, -2).contiguous()  # (B, N, C)
        neighborhood, center, patch_idx = self.group_divider(pts_bn)
        group_tokens = self.encoder(neighborhood)
        group_tokens = self.reduce_dim(group_tokens)

        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        feature = self.blocks(x, pos)
        patch_emb = self.norm(feature)[:, 1:, :].transpose(-1, -2).contiguous()
        patch_centers = center.transpose(-1, -2).contiguous()
        return patch_emb, patch_centers, patch_idx

    def forward(self, pts):
        pe, _, _ = self.forward_patches(pts)
        return pe
