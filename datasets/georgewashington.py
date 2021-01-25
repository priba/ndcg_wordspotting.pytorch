import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random

class GeorgeWashington(Dataset):
    def __init__(self, root, word_id, char_to_idx, image_extension='.png', transform=transforms.ToTensor(), max_length=10):
        self._dataset = 'George Washington'
        self.root = root
        self.word_list = word_id
        self.image_extension = image_extension
        self.transform = transform
        self.char_to_idx = char_to_idx
        self.max_length=max_length

    def __getitem__(self, index):
        word_id, label = self.word_list[index]

        img = Image.open(os.path.join(self.root, f'{word_id}{self.image_extension}'))
        if self.transform is not None:
            img = self.transform(img)

        word = torch.tensor([self.char_to_idx[i] for i in label], dtype=torch.long)
        word = torch.cat((word, self.voc_size()*torch.ones(self.max_length-word.shape[0]))).long()
        return img, word, label

    def __len__(self):
        return len(self.word_list)

    def voc_size(self):
        return len(self.char_to_idx)

def prepare_dataset(root, word_path, image_extension):
    # Create word folder
    os.mkdir(word_path)

    # Read file order
    with open(os.path.join(root, 'file_order.txt')) as f:
        file_order = f.read().splitlines()

    with open(os.path.join(root, 'annotations.txt'), encoding='ISO-8859-15') as f:
        annotations = f.read().splitlines()

    idx_annotations = 0
    with open(os.path.join(root, 'transcriptions.txt'), 'w') as f_trans:
        for img_file in file_order:
            # Get image id
            img_id = os.path.splitext(img_file)[0]

            # Get GT bounding boxes
            with open(os.path.join(root, f'{img_id}_boxes.txt')) as f:
                img_boxes = f.read().splitlines()
            img_boxes.pop(0)

            # Read image
            img = Image.open(os.path.join(root, img_file))
            w, h = img.size

            # Iterate through bounding  boxes
            for id_word, bb_line in enumerate(img_boxes):
                bb_line = list(map(float, bb_line.split()))
                x1 = bb_line[0]*w
                x2 = bb_line[1]*w
                y1 = bb_line[2]*h
                y2 = bb_line[3]*h
                img_word = img.crop(box=(x1,y1,x2,y2))
                img_word.save(os.path.join(word_path, f'{img_id}_{id_word}{image_extension}'))
                f_trans.write(f'{img_id}_{id_word} {annotations[idx_annotations]}\n')
                idx_annotations += 1

def build_dataset(root, image_extension='.png', transform=transforms.ToTensor):
    word_path = os.path.join(root, 'words')
    if not os.path.isdir(word_path):
        prepare_dataset(root, word_path, image_extension)
    with open(os.path.join(root, 'transcriptions.txt')) as f:
        transcriptions = f.read().splitlines()

    random.shuffle(transcriptions)
    transcriptions = [i.split() for i in transcriptions]
    total_words = len(transcriptions)

    dic = ''
    max_length = 0
    for i in transcriptions:
        max_length = len(i[1]) if max_length < len(i[1]) else max_length
        dic += i[1]

    dic = list(set(dic))
    dic.sort()
    dic = {k: v for v, k in enumerate(dic)}

    word_id_train = transcriptions[:round(total_words*0.65)]
    mean = []
    std = []
    transform = transforms.Compose([
        transforms.Resize((100, 500)),
        transforms.ToTensor(),
    ])
    for i in word_id_train:
        img = Image.open(os.path.join(word_path, f'{i[0]}{image_extension}'))
        img = transform(img)
        mean.append(img.mean())
        std.append(img.std())
    mean = torch.stack(mean).mean()
    std = torch.stack(std).mean()


    transform = transforms.Compose([
        transforms.Resize((100, 500)),
        transforms.ToTensor(),
        transforms.Normalize((mean.item()), (std.item())),
    ])
    # Prepare Data
    train_file = GeorgeWashington(
        root=word_path,
        word_id=word_id_train,
        transform=transform,
        char_to_idx=dic,
        max_length = max_length,
    )
    validation_file = GeorgeWashington(
        root=word_path,
        word_id=transcriptions[round(total_words*0.65):round(total_words*0.75)],
        transform=transform,
        char_to_idx=dic,
        max_length = max_length,
    )
    test_file = GeorgeWashington(
        root=word_path,
        word_id=transcriptions[-round(total_words*0.25):],
        transform=transform,
        char_to_idx=dic,
        max_length = max_length,
    )
    return train_file, validation_file, test_file

