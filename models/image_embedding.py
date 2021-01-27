import torch
import torch.nn as nn
import torch.nn.functional as F
from .gpp import GPP

class ImageEmbedding(nn.Module):
    def __init__(self, n_out, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super(ImageEmbedding, self).__init__()

        self.conv = nn.Sequential(
            # [BATCH_SIZE, 1, w, h]
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 64, *, *]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [BATCH_SIZE, 64, *, *]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 128, *, *]
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [BATCH_SIZE, 128, *, *]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            # [BATCH_SIZE, 256, *, *]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 512, *, *]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # [BATCH_SIZE, 512, *, *]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.spp = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type)
        pooling_output_size = self.spp.pooling_output_size
        self.mlp = nn.Sequential(
            nn.Linear(pooling_output_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, n_out)
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.spp(y)
        y = self.mlp(y)
        return F.normalize(y, dim=-1)

