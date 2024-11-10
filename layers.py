import torch
import torch.nn as nn
from transformer import Block

class AveragePool(nn.Module):

    def __init__(self):
        super(AveragePool, self).__init__()

    def forward(self, x):
        batch, num_nodes, features = x.shape
        pooled_vector = torch.mean(x, dim=1)
        return pooled_vector

class SpatialTransformer(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers):
        super(SpatialTransformer, self).__init__()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias=True, qk_norm=None, norm_layer=nn.LayerNorm)
                                     for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pool = AveragePool()

    def forward(self, x):
        batch, num_nodes, num_time_points_seg = x.shape
        attn_list = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_list.append(attn)

        final_attn = torch.cat(attn_list, dim=1)
        sum_attn = torch.sum(final_attn, dim=1)


        x = self.layer_norm(x)  # (batch, num_nodes, embed_dim)
        x = self.pool(x)     # (batch, embed_dim)
        return x, sum_attn

class TemporalTransformer(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers):
        super(TemporalTransformer, self).__init__()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias=True, qk_norm=None, norm_layer=nn.LayerNorm)
                                     for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch, num_seg, D = x.shape
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.layer_norm(x)  # (batch, num_seg, embed_dim)
        return x

class Decoder(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, qkv_bias=True, qk_norm=None, norm_layer=nn.LayerNorm)
                                     for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch, num_seg, D = x.shape
        for blk in self.blocks:
            x, _ = blk(x)
        x = self.layer_norm(x)  # (batch, num_seg, embed_dim)
        return x