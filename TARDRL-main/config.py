

class Config():

    def __init__(self):

        # Spatial Feature Learning
        self.sf_embed_dim = 256
        self.sf_num_heads = 4
        self.sf_num_layers = 1

        # Tempral Feature Learning
        self.tf_embed_dim = 128
        self.tf_num_heads = 4
        self.tf_num_layers = 2

        # Deocder
        self.decoder_embed_dim = 64
        self.decoder_num_heads = 2
        self.decoder_num_layers = 2

        self.max_num_pos = 1024
        self.mask_ratio = 0.5
        self.ratio_highest_attention = 0.5
        self.node_sz = 200
        self.timeseries_sz = 100
        self.num_time_points_seg =  20