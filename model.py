import torch
import torch.nn as nn
from transformer import Block
from utils import positionalencoding1d
import numpy as np
import random
from config import Config
from layers import SpatialTransformer, TemporalTransformer, Decoder




class TARDRL(nn.Module):

    def __init__(self, cfg: Config):
        super(TARDRL, self).__init__()
        self.num_nodes = cfg.node_sz
        self.num_time_points = cfg.timeseries_sz
        self.num_time_points_seg = cfg.num_time_points_seg


        # num_time_points_seg -> sf_embed_dim
        self.linear1 = nn.Linear(self.num_time_points_seg, cfg.sf_embed_dim)
        # spatial feature learning
        self.sf_pos_embed = nn.Parameter(torch.zeros(1, self.num_nodes, cfg.sf_embed_dim), requires_grad=False)
        self.sfl = SpatialTransformer(cfg.sf_embed_dim, cfg.sf_num_heads, cfg.sf_num_layers)

        # sf_embed_dim -> tf_embed_dim
        self.linear2 = nn.Linear(cfg.sf_embed_dim, cfg.tf_embed_dim)
        # temporal feature learning
        self.tf_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_num_pos, cfg.tf_embed_dim), requires_grad=False)
        self.tfl = TemporalTransformer(cfg.tf_embed_dim, cfg.tf_num_heads, cfg.tf_num_layers)

        # tf_embed_dim -> dec_embed_dim
        self.linear3 = nn.Linear(cfg.tf_embed_dim, cfg.decoder_embed_dim)
        # decoder for reconstruction
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, cfg.max_num_pos, cfg.decoder_embed_dim), requires_grad=False)
        self.decoder = Decoder(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.decoder_num_layers)
        self.head = nn.Linear(cfg.decoder_embed_dim, self.num_nodes * self.num_time_points_seg)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.sf_embed_dim), requires_grad=True)

        # linear head for downstream task
        self.classifier = nn.Sequential(
            nn.Linear(cfg.tf_embed_dim, 128),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 32),
            nn.BatchNorm1d(num_features=32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )

        self.initialize_weights()

    def initialize_weights(self):
        sf_pos_embed = positionalencoding1d(self.sf_pos_embed.shape[-1], self.sf_pos_embed.shape[-2])
        self.sf_pos_embed.data.copy_(sf_pos_embed.float().unsqueeze(0))

        tf_pos_embed = positionalencoding1d(self.tf_pos_embed.shape[-1], self.tf_pos_embed.shape[-2])
        self.tf_pos_embed.data.copy_(tf_pos_embed.float().unsqueeze(0))


        decoder_pos_embed = positionalencoding1d(self.decoder_pos_embed.shape[-1], self.decoder_pos_embed.shape[-2])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.float().unsqueeze(0))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def segment_wise_random_masking(self, x, attn, mask_ratio, ratio_highest_attention):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [batch, num_nodes, D], sequence
        attn: [batch, num_nodes, num_nodes]
        """
        batch, num_nodes, D = x.shape
        num_nodes_keep = int(num_nodes * (1 - mask_ratio * ratio_highest_attention))
        # remove the diagonal values of the attention map while aggregating the column wise attention scores
        attn_weights = torch.sum(attn, dim=1) - torch.diagonal(attn, offset=0, dim1=1, dim2=2)  # (batch, num_nodes)
        # get the indices of important ROIs
        res, indices_1 = attn_weights.topk(int(ratio_highest_attention * num_nodes))   # descend  shape: (batch, ratio_highest_attention * num_nodes)
        # randomly select indices of important ROIs
        mask_indices = torch.tensor(
            [random.sample(indices_i.tolist(), num_nodes - num_nodes_keep) for indices_i in indices_1],
            dtype=torch.long).to(x.device)

        # create the binary mask tensor
        mask = torch.zeros(batch, num_nodes, device=x.device)
        mask.scatter_(1, mask_indices, 1)
        # replace mask token
        x[mask.bool(), :] = self.mask_token
        return x, mask

    # masking strategy
    def random_masking(self, x, attn, mask_ratio, ratio_highest_attention):
        # the same coupled architecture coupling reconstruction tasks with downstream tasks
        # but different mode for random mask and attention-guided mask
        # mask_ratio == 1 and ratio_highest_attention != 1: reconstructing time series of all important ROIs
        # mask_ratio != 1 and ratio_highest_attention == 1: all ROIs are considered as important, so it becomes random mask
        # mask_ratio != 1 and ratio_highest_attention != 1: randomly selecting some important ROIs for reconstruction
        # the final mask ratio is (mask_ratio x ratio_highest_attention)
        batch, num_seg, num_nodes, D = x.shape

        # masking all segments
        masked_x_list = []
        mask_list = []
        for i in range(num_seg):
            masked_x, mask = self.segment_wise_random_masking(x[:, i, :, :], attn[:, i, :, :], mask_ratio, ratio_highest_attention)
            masked_x_list.append(masked_x)
            mask_list.append(mask)

        final_masked_x = torch.stack(masked_x_list, dim=1)
        final_mask = torch.stack(mask_list, dim=1)
        return final_masked_x, final_mask

    # forward_reconstruction
    def forward_reconstruction(self, x, attn, mask_ratio=0.5, ratio_highest_attention=0.5):
        batch, num_seg, num_nodes, num_time_points_seg = x.shape

        x = self.linear1(x)
        masked_x, mask = self.random_masking(x, attn, mask_ratio, ratio_highest_attention)
        reshaped_x = masked_x.view(batch * num_seg, num_nodes, -1)
        reshaped_x = reshaped_x + self.sf_pos_embed
        sf, sum_attn = self.sfl(reshaped_x)
        final_sf = sf.view(batch, num_seg, -1)

        tf = self.linear2(final_sf)
        tf = tf + self.tf_pos_embed[:, :num_seg, :]
        final_tf = self.tfl(tf)

        df = self.linear3(final_tf)
        df = df + self.decoder_pos_embed[:, :num_seg, :]
        final_df = self.decoder(df)

        recon_x = self.head(final_df)
        recon_x = recon_x.view(batch, num_seg, num_nodes, num_time_points_seg)

        return recon_x, mask

    # forward_downstream
    def forward_downstream(self, x):
        batch, num_seg, num_nodes, num_time_points_seg = x.shape

        x = self.linear1(x)

        reshaped_x = x.view(batch * num_seg, num_nodes, -1)
        reshaped_x = reshaped_x + self.sf_pos_embed
        sf, sum_attn = self.sfl(reshaped_x)
        final_sf = sf.view(batch, num_seg, -1)
        final_sum_attn = sum_attn.view(batch, num_seg, num_nodes, num_nodes)

        tf = self.linear2(final_sf)
        tf = tf + self.tf_pos_embed[:, :num_seg, :]
        final_tf = self.tfl(tf)

        mean_tf = torch.mean(final_tf, dim=1)
        logits = self.classifier(mean_tf)

        return logits, final_sum_attn