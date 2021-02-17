import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from loss import DGCLoss
import Levenshtein
import multiprocessing
from joblib import Parallel, delayed

class StringDataset(Dataset):
    def __init__(self, root, char_to_idx, transform=transforms.ToTensor()):
        self._dataset = 'StringDataset'
        self.root = root
        with open(os.path.join(root, 'lexicon.txt'), 'r', encoding='utf-8', errors='ignore') as f:
            words = f.read().splitlines()
        words = [i.lower() for i in words]
        self.word_list = np.unique(words)
        self.char_to_idx = char_to_idx

    def __getitem__(self, index):
        label = self.word_list[index]

        word = torch.tensor([self.char_to_idx[i] for i in label], dtype=torch.long)
        return word, label

    def __len__(self):
        return len(self.word_list)


class CollateFn():
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        seq_len = [x.shape[0] for x in batch[0]]
        padded_sequences = self.voc_size*torch.ones(len(batch[0]), max(seq_len), dtype=torch.long)
        for i, x in enumerate(batch[0]):
            padded_sequences[i, :seq_len[i]] = x
        batch[0] = padded_sequences
        return tuple(batch)

# Define task for multiprocessing
def multiprocessing_levenshtein(str1, gallery_labels, i):
    out = torch.zeros(len(gallery_labels))
    for j, str2 in enumerate(gallery_labels[i+1:], start=i+1):
        out[j] = Levenshtein.distance(str1, str2)
    return out

def pretrain_string(str_model, device, data_path, similarity, char_to_idx):
    print('Pretraining String Embedding')
    collate = CollateFn(len(char_to_idx))
    data_file = StringDataset(data_path, char_to_idx)
    dataloader = DataLoader(data_file, batch_size=512, shuffle=True, num_workers=8, collate_fn=collate.collate_fn, drop_last=True)
    str_model.train()
    loss_func = DGCLoss(k=1e-2, penalize=False)
    optim = torch.optim.Adam(str_model.parameters(), 1e-4)
    print(f'Number of iterations per epoch {len(dataloader)}')
    for e in range(50):
        epoch_loss = 0
        for step, (data, labels) in enumerate(dataloader):
            optim.zero_grad()

            data = data.to(device)

            output_str = str_model(data)
            ranking_str = similarity(output_str, output_str)


            mask_diagonal = ~ torch.eye(ranking_str.shape[0]).bool()

            # Ground-truth Ranking function
            n_processors = min(multiprocessing.cpu_count(), 8)
            gt = Parallel(n_jobs=n_processors)(delayed(multiprocessing_levenshtein)(str1, labels, i) for i, str1 in enumerate(labels))
            gt = torch.stack(gt).to(ranking_str.device)
            gt = gt + gt.t()

            loss = loss_func(ranking_str, gt, mask_diagonal=mask_diagonal)
            if loss.grad_fn is not None:
                loss.backward()
                optim.step()
            epoch_loss += loss.item()

        print(f'Pretraining epoch {e}, Loss {epoch_loss/len(dataloader)}')

    return str_model
