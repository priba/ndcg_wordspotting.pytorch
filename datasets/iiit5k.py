import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
from scipy.io import loadmat

class IIIT5k(Dataset):
    def __init__(self, root, data, char_to_idx, transform=transforms.ToTensor(), max_length=10, subset='train'):
        self._dataset = 'IIIT5k'
        self.root = root
        self.subset = subset
        self.in_channels = 3

        self.words, self.labels = [], []
        for w in data:
            self.words.append(w['ImgName'][0])
            self.labels.append(w['GroundTruth'][0].lower())

        self.unique_labels = list(set(self.labels))
        self.transform = transform
        self.char_to_idx = char_to_idx
        self.max_length=max_length
        self.height = 100

    def __getitem__(self, index):
        word_id, label = self.words[index], self.labels[index]

        img = Image.open(os.path.join(self.root, word_id))
        hpercent = self.height/float(img.size[1])
        wsize = int((float(img.size[0])*float(hpercent)))
        img = img.resize((wsize, self.height), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        word = torch.tensor([self.char_to_idx[i] for i in label], dtype=torch.long)
        return img, word, label, word_id

    def __len__(self):
        return len(self.words)

    def voc_size(self):
        return len(self.char_to_idx)

    def balance_weigths(self):
        labels = np.array(self.labels)
        weights = torch.zeros(len(self.words))
        for l in self.unique_labels:
            count = np.count_nonzero(labels == l)
            weights[labels==l] = count
        return 1./weights

    def query_to_tensor(self, input_string):
        return torch.tensor([self.char_to_idx[i] for i in input_string], dtype=torch.long)


def build_dataset(root, image_extension='.png', transform=transforms.ToTensor):
    train_path = os.path.join(root, 'traindata.mat')
    test_path = os.path.join(root, 'testdata.mat')

    train_mat = loadmat(train_path)
    test_mat = loadmat(test_path)

    train_mat = train_mat['traindata'][0]
    test_mat = test_mat['testdata'][0]

    train_transcriptions = [[w['ImgName'][0], w['GroundTruth'][0].lower()] for w in train_mat]

    dic = ''
    max_length = 0
    for i in train_transcriptions:
        max_length = len(i[1]) if max_length < len(i[1]) else max_length
        dic += i[1]

    dic = list(set(dic))
    dic.sort()
    dic = {k: v for v, k in enumerate(dic)}

    mean, std = [], []
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for i in train_transcriptions:
        img = Image.open(os.path.join(root, i[0]))
        img = transform(img)
        mean.append(img.mean())
        std.append(img.std())
    mean = torch.stack(mean).mean()
    std = torch.stack(std).mean()

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, scale=(0.9,1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((mean.item()), (std.item())),
    ])

    # Prepare Data
    train_file = IIIT5k(
        root=root,
        data=train_mat,
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
        subset='train',
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean.item()), (std.item())),
    ])
    test_file = IIIT5k(
        root=root,
        data=test_mat,
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
        subset='test',
    )
    val_file = test_file
    print(f'Datasets created:\n\t*Train: {len(train_file)}\n\t*Validation: {len(val_file)}\n\t*Test: {len(test_file)}')
    return train_file, val_file, test_file

