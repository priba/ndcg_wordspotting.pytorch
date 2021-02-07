import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from xml.dom import minidom
import random
import numpy as np

class GeorgeWashington(Dataset):
    def __init__(self, root, char_to_idx, transform=transforms.ToTensor(), max_length=10):
        self._dataset = 'George Washington'
        self.root = root

        with open(os.path.join(root, 'transcriptions.txt')) as f:
            transcriptions = f.read().splitlines()
        word_list = [i.split() for i in transcriptions]

        self.words, self.labels = [], []
        for w, l in word_list:
            self.words.append(w)
            self.labels.append(l)

        self.unique_labels = list(set(self.labels))
        self.transform = transform
        self.char_to_idx = char_to_idx
        self.max_length=max_length

    def __getitem__(self, index):
        word_id, label = self.words[index], self.labels[index]

        img = Image.open(os.path.join(self.root, word_id))
        if self.transform is not None:
            img = self.transform(img)

        word = torch.tensor([self.char_to_idx[i] for i in label], dtype=torch.long)
        word = torch.cat((word, self.voc_size()*torch.ones(self.max_length-word.shape[0]))).long()
        return img, word, label

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

def prepare_dataset(root, word_path, image_extension, partition):
    # Create word folder
    os.mkdir(word_path)

    previous_image_name = ''
    for subset in ['train', 'test']:
        os.mkdir(os.path.join(word_path,subset))

        with open(os.path.join(word_path, subset, 'transcriptions.txt'), 'w') as f_trans:

            xmldoc = minidom.parse(f'./datasets/georgewashington/gw_{partition}_{subset}.xml')
            spotlist = xmldoc.getElementsByTagName('spot')

            for word_id, spot in enumerate(spotlist):
                # Get image id
                img_id = spot.attributes['image'].value.replace('.png', '')
                image_name = img_id + '.tif'

                # Read image
                if image_name != previous_image_name:
                    image_name_previous = image_name
                    img = Image.open(os.path.join(root, image_name))

                x1 = int(spot.attributes['x'].value)
                y1 = int(spot.attributes['y'].value)
                x2 = x1 + int(spot.attributes['w'].value)
                y2 = y1 + int(spot.attributes['h'].value)
                img_word = img.crop(box=(x1,y1,x2,y2))

                img_word_name = f'{img_id}_{word_id:04d}{image_extension}'
                img_word.save(os.path.join(word_path, subset, img_word_name))
                f_trans.write(f'{img_word_name} {spot.attributes["word"].value}\n')


def build_dataset(root, image_extension='.png', transform=transforms.ToTensor, partition='cv1'):
    word_path = os.path.join(root, partition)
    if not os.path.isdir(word_path):
        print(f'Preparing dataset at: {word_path}')
        prepare_dataset(root, word_path, image_extension, partition)

    with open(os.path.join(word_path, 'train', 'transcriptions.txt')) as f:
        train_transcriptions = f.read().splitlines()

    train_transcriptions = [i.split() for i in train_transcriptions]

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
        img = Image.open(os.path.join(word_path, 'train', f'{i[0]}'))
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
    train_file = GeorgeWashington(
        root=os.path.join(word_path, 'train'),
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean.item()), (std.item())),
    ])
    test_file = GeorgeWashington(
        root=os.path.join(word_path, 'test'),
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
    )
    return train_file, test_file, test_file

