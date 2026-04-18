import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
        # qkv shape:[3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # ---------------------------------------------------------
        # EXCLUSIVE SELF ATTENTION (XSA) IMPLEMENTATION
        # 1. Standard attention output (Y)
        y = attn @ v  # Shape:[B, num_heads, N, head_dim]

        # 2. Normalize Value vectors (Vn)
        vn = F.normalize(v, dim=-1)

        # 3. Subtract the projection of Y onto Vn (Z = Y - (Y · Vn) * Vn)
        z = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        # ---------------------------------------------------------

        x = z.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Uses XSA-enabled Attention
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Note: In standard cross-attention there are usually linear projections 
        # for K and V too. Since your original code did not have them, I left 
        # them out to prevent breaking your model's parameter count/checkpoints.

    def forward(self, query, kv):
        B, N_q, C = query.shape
        _, N_kv, _ = kv.shape

        # 1. Project Query and Reshape for Multi-Head
        # Shape: [B, N_q, C] ->[B, N_q, num_heads, head_dim] -> [B, num_heads, N_q, head_dim]
        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Reshape K and V for Multi-Head (without additional linear projection, per original code)
        # Shape: [B, N_kv, C] -> [B, N_kv, num_heads, head_dim] ->[B, num_heads, N_kv, head_dim]
        k = kv.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = kv.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Calculate Attention Scores
        # q:[B, num_heads, N_q, head_dim] @ k.T: [B, num_heads, head_dim, N_kv] 
        # result: [B, num_heads, N_q, N_kv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 4. Multiply by Values
        # attn:[B, num_heads, N_q, N_kv] @ v:[B, num_heads, N_kv, head_dim]
        # result:[B, num_heads, N_q, head_dim]
        out = attn @ v

        # 5. Concatenate Heads Back Together
        # Shape: [B, num_heads, N_q, head_dim] ->[B, N_q, num_heads, head_dim] -> [B, N_q, C]
        out = out.transpose(1, 2).reshape(B, N_q, C)

        # 6. Final Projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock, self).__init__()
        self.self_attn_layer = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp = Mlp(in_features=dim, hidden_features=dim // 2, act_layer=act_layer, drop=drop)

        self.norm1 = norm_layer(dim)
        self.norm2_q = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

        self.norm_mlp = norm_layer(dim)

    def forward(self, query, img_fea):
        query = query + self.self_attn_layer(self.norm1(query))
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))
        query = query + self.mlp(self.norm_mlp(query))

        return query