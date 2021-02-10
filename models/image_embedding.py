import torch
import torch.nn as nn
import torch.nn.functional as F
from .gpp import GPP
from utils import NestedTensor


class Block(nn.Module):
    '''ResNet Block'''

    def __init__(self, in_size, out_size, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.conv3 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.conv_res = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_size)

    def forward(self, x, maxpool=True):
        residual = x
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.bn_res(self.conv_res(residual))
        if maxpool:
            x = F.max_pool2d(nn.ReLU()(x), 2)
        return x


class ResNet12(nn.Module):
    def __init__(self, n_out, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super(ImageEmbedding, self).__init__()
        self.block1 = Block(in_channels, 64)
        self.block2 = Block(64, 128)
        self.block3 = Block(128, 256)
        self.block4 = Block(256, n_out)

    def forward(self, input_tensor):
        y, mask = input_tensor.decompose()
        assert mask is not None
        # Input 64x64x2
        y = self.block1(y)
        # Input 32x32x64
        y = self.block2(y)
        # Input 16x16x128
        y = self.block3(y)
        # Input 8x8x256
        y = self.block4(y, maxpool=False)

        mask = F.interpolate(mask[None].float(), size=y.shape[-2:]).to(torch.bool)[0]
        mask = ~mask

        y = (mask.unsqueeze(1)*y).sum((-2,-1)) / mask.float().sum((-2,-1)).unsqueeze(-1)

        return F.normalize(y, dim=-1)


class PHOCNet(nn.Module):
    def __init__(self, n_out, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super(ImageEmbeddingPhocNet, self).__init__()

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
        input_tensor, mask = x.decompose()
        y = self.conv(x.tensors)
        assert mask is not None
        mask = F.interpolate(mask[None].float(), size=y.shape[-2:]).to(torch.bool)[0]

        y = self.spp(NestedTensor(y, mask))
        y = self.mlp(y)
        return F.normalize(y, dim=-1)

