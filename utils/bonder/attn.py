import torch
import torch.nn as nn
import torch.nn.functional as F  # Added for normalization
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
        # qkv shape: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # --- EXCLUSIVE SELF ATTENTION (XSA) START ---
        # 1. Standard attention output (Y)
        y = attn @ v  # Shape: [B, num_heads, N, head_dim]

        # 2. Normalize Value vectors (Vn)
        vn = F.normalize(v, dim=-1)

        # 3. Subtract the projection (Z = Y - (Y · Vn) * Vn)
        # This removes information that overlaps with the token's own identity
        z = y - (y * vn).sum(dim=-1, keepdim=True) * vn
        # --- EXCLUSIVE SELF ATTENTION (XSA) END ---

        x = z.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_text = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, query, kv):
        # query: text prototypes, kv: image features
        query_proj = self.q(query)
        k, v = kv, kv

        attn = (query_proj @ k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # --- APPLYING XSA LOGIC TO CROSS-ATTENTION ---
        # In cross-attention, we want the output to be exclusive from the Query 
        # (the text prototype itself) so it only captures *new* visual information.
        y = attn @ v
        
        # Normalize the Query (the "self" in this context)
        qn = F.normalize(query, dim=-1)
        
        # Remove projection of visual info onto the original text prototype direction
        z = y - (y * qn).sum(dim=-1, keepdim=True) * qn
        
        query = self.proj(z)
        query = self.proj_drop(query)

        return query

# Block and CrossAttnBlock remain the same as they use the modified Attention classes above.
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
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
        # Now uses XSA-enhanced Self Attention
        query = query + self.self_attn_layer(self.norm1(query))
        # Now uses XSA-enhanced Cross Attention
        query = query + self.cross_attn(self.norm2_q(query), self.norm2_kv(img_fea))
        query = query + self.mlp(self.norm_mlp(query))
        return query