"""
Source frin: https://github.com/georgeretsi/pytorch-phocnet/blob/master/src/cnn_ws/spatial_pyramid_layers/gpp.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPP(nn.Module):

    def __init__(self, gpp_type='tpp', levels=3, pool_type='max_pool'):
        super(GPP, self).__init__()

        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown gpp_type. Must be either \'spp\', \'tpp\', \'gpp\'')

        if pool_type not in ['max_pool', 'avg_pool']:
            raise ValueError('Unknown pool_type. Must be either \'max_pool\', \'avg_pool\'')

        if gpp_type == 'spp':
            self.pooling_output_size = sum([4 ** level for level in range(levels)]) * 512
        elif gpp_type == 'tpp':
            self.pooling_output_size = (2 ** levels - 1) * 512
        if gpp_type == 'gpp':
            self.pooling_output_size = sum([h * w for h in levels[0] for w in levels[1]]) * 512

        self.gpp_type = gpp_type
        self.levels = levels
        self.pool_type = pool_type

    def forward(self, input_x):

        if self.gpp_type == 'spp':
            return self._spatial_pyramid_pooling(input_x, self.levels)
        if self.gpp_type == 'tpp':
            return self._temporal_pyramid_pooling(input_x, self.levels)
        if self.gpp_type == 'gpp':
            return self._generic_pyramid_pooling(input_x, self.levels)

    def _pyramid_pooling(self, input_x, output_sizes):
        x_batch, mask = input_x.decompose()
        y = []
        for x, m in zip(x_batch, mask):
            m = ~m

            w = torch.where(m.any(0))[0]
            w = m.shape[1] if w.nelement() == 0 else w.max()+1

            h = torch.where(m.any(1))[0]
            h = m.shape[0] if h.nelement() == 0 else h.max()+1

            x = x[:,:h,:w]

            pyramid_level_tensors = []
            for tsize in output_sizes:
                if self.pool_type == 'max_pool':
                    pyramid_level_tensor = F.adaptive_max_pool2d(x, tsize)
                if self.pool_type == 'avg_pool':
                    pyramid_level_tensor = F.adaptive_avg_pool2d(x, tsize)
                pyramid_level_tensor = pyramid_level_tensor.view(1, -1)
                pyramid_level_tensors.append(pyramid_level_tensor)
            y.append(torch.cat(pyramid_level_tensors, dim=1))
        return torch.cat(y, dim=0)

    def _spatial_pyramid_pooling(self, input_x, levels):

        output_sizes = [(int( 2 **level), int( 2 **level)) for level in range(levels)]

        return self._pyramid_pooling(input_x, output_sizes)

    def _temporal_pyramid_pooling(self, input_x, levels):

        output_sizes = [(1, int( 2 **level)) for level in range(levels)]

        return self._pyramid_pooling(input_x, output_sizes)

    def _generic_pyramid_pooling(self, input_x, levels):

        levels_h = levels[0]
        levels_w = levels[1]
        output_sizes = [(int(h), int(w)) for h in levels_h for w in levels_w]

        return self._pyramid_pooling(input_x, output_sizes)

