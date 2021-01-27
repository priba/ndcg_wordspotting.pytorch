import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .gpp import GPP

class StringEmbedding(nn.Module):
    def __init__(self, n_out, voc_size, num_layers=2):
        super(StringEmbedding, self).__init__()
        self.embedding = nn.Embedding(voc_size+1, n_out, padding_idx=voc_size)
        self.padding_idx = voc_size
        self.hidden_size = n_out
        self.GRU = nn.GRU(n_out, self.hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.mlp = nn.Linear(2*n_out, n_out)
        self.num_layers = num_layers

    def forward(self, x):
        y = self.embedding(x)
        seq_lens = x.shape[1] - (x==self.padding_idx).sum(1)
        packed = pack_padded_sequence(y, seq_lens.tolist(), batch_first=True, enforce_sorted=False)

        h0 = torch.zeros(2*self.num_layers, y.shape[0], self.hidden_size).to(y.device)
        output, hn = self.GRU(packed, h0)
        unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)

        unpacked = unpacked.view(y.shape[0], seq_lens.max(),2, self.hidden_size)
        last_seq = torch.cat([unpacked[e, i-1, 0].unsqueeze(0) for e, i in enumerate(seq_lens)], dim=0)
        first_seq = unpacked[:,0,1]
        y = torch.cat((first_seq, last_seq), dim=1)
        y = self.mlp(y)
        return F.normalize(y, dim=-1)


