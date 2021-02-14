import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from xml.dom import minidom
import random
import numpy as np
import string

class IAM(Dataset):
    def __init__(self, root, char_to_idx, transform=transforms.ToTensor(), max_length=10, subset='train', image_extension = '.png'):
        self._dataset = 'IAM'
        self.root = root
        self.subset = subset
        self.image_extension = image_extension
        self.in_channels = 1

        transcription_name = 'transcriptions.txt'
        with open(os.path.join(root, transcription_name)) as f:
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
        self.height = 100

    def __getitem__(self, index):
        word_id, label = self.words[index], self.labels[index]

        img = Image.open(os.path.join(self.root, word_id + self.image_extension))
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

def readset(set_name):
    with open(f'./datasets/iam/{set_name}.txt', 'r') as f:
        lines = f.read().splitlines()
    return lines

def prepare_dataset(root, word_path, image_extension):
    # Create word folder
    os.mkdir(word_path)
    os.mkdir(os.path.join(word_path, 'train'))
    os.mkdir(os.path.join(word_path, 'validation'))
    os.mkdir(os.path.join(word_path, 'test'))

    train_lines = readset('trainset')
    val_lines = readset('validationset1') # + readset('validationset2')
    test_lines = readset('testset')

    train_list, val_list, test_list = [], [], []

    with open(os.path.join(root, 'ascii', 'words.txt'), 'r') as f:
        iam_words = f.read().splitlines()
    iam_words = [w for w in iam_words if not w.startswith('#')]

    current_form = ''
    for w in iam_words:
        w_list = w.split()
        if w_list[1] == 'ok':
            w_id = w_list[0]
            w_id_split = w_id.split('-')
            form_id = w_id_split[0] + '-' + w_id_split[1]
            line_id = form_id + '-' + w_id_split[2]

            x, y, w, h = int(w_list[3]), int(w_list[4]), int(w_list[5]), int(w_list[6])
            if x==-1 or y==-1 or w==-1 or h==-1:
                continue

            transcription = ''.join(w_list[8:]).lower()
            transcription = transcription.translate(str.maketrans('', '', string.punctuation))
            if len(transcription) == 0:
                transcription = '-'
            if line_id in train_lines:
                subset = 'train'
                train_list.append(f'{w_id} {transcription}\n')
            elif line_id in val_lines:
                subset = 'validation'
                val_list.append(f'{w_id} {transcription}\n')
            elif line_id in test_lines:
                subset = 'test'
                test_list.append(f'{w_id} {transcription}\n')
            else:
                continue

            image_file = os.path.join(root, 'forms', form_id + image_extension)
            if current_form != image_file:
                img_form = Image.open(image_file)
                current_form = image_file

            img_word = img_form.crop(box=(x,y,x+w,y+h))

            img_word.save(os.path.join(word_path, subset, w_id + image_extension))

    save_transcription(word_path, 'train', train_list)
    save_transcription(word_path, 'validation', val_list)
    save_transcription(word_path, 'test', test_list)


def save_transcription(word_path, subset, subset_list, transcription_name='transcriptions.txt'):
    with open(os.path.join(word_path, subset, transcription_name), 'w') as f_trans:
        for line in subset_list:
            f_trans.write(line)

def build_dataset(root, image_extension='.png', transform=transforms.ToTensor):
    word_path = os.path.join(root, 'word_level')
    if not os.path.isdir(word_path):
        print(f'Preparing dataset at: {word_path}')
        prepare_dataset(root, word_path, image_extension)

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
        img = Image.open(os.path.join(word_path, 'train', i[0] + image_extension))
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
    train_file = IAM(
        root=os.path.join(word_path, 'train'),
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
        subset='train',
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean.item()), (std.item())),
    ])
    val_file = IAM(
        root=os.path.join(word_path, 'validation'),
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
        subset='validation',
    )
    test_file = IAM(
        root=os.path.join(word_path, 'test'),
        transform=transform,
        char_to_idx=dic,
        max_length=max_length,
        subset='test',
    )
    print(f'Datasets created:\n\t*Train: {len(train_file)}\n\t*Validation: {len(val_file)}\n\t*Test: {len(test_file)}')
    return train_file, val_file, test_file

