import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np

from loss import DGCLoss, MAPLoss
from utils import CosineSimilarityMatrix, cosine_similarity_matrix, ndcg, meanavep
from datasets import build as build_dataset

import pandas as pd
import seaborn as sns

from options import get_args_parser

from models.image_embedding import ImageEmbedding
from models.string_embedding import StringEmbedding
from logger import AverageMeter

def train(img_model, str_model, device, train_loader, optim, lossf, epoch):
    img_model.train()
    str_model.train()

    stats = {
        'train_loss': AverageMeter(),
        'img_loss': AverageMeter(),
        'str_loss': AverageMeter(),
        'cross_loss': AverageMeter(),
    }

    start_time = time.time()
    for step, (data, targets, labels) in enumerate(train_loader):
        optim.zero_grad()

        data, targets = data.to(device), targets.to(device)

        # Batch size
        bz = data.shape[0]

        output_img = img_model(data)
        output_str = str_model(targets)

        loss_img, loss_str, loss_cross = 0,0,0
        for k,loss_func in lossf.items():
            loss_img += loss_func(output_img, labels)
            loss_str += loss_func(output_str, labels)
            loss_cross += loss_func(query=output_str, gallery=output_img, target=labels)
        loss = loss_img + loss_str + loss_cross
        loss.backward()
        optim.step()

        stats['train_loss'].update(loss.item(), bz)
        stats['img_loss'].update(loss_img.item(), bz)
        stats['str_loss'].update(loss_str.item(), bz)
        stats['cross_loss'].update(loss_cross.item(), bz)


    loss_str = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
    print(
                f'EPOCH: {epoch}',
                f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
                f'LOSS: {loss_str}',
    )
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}', end='\n')


def test(img_model, str_model, device, test_loader, lossf, criterion, epoch):
    img_model.eval()
    str_model.eval()

    stats = {}
    for k, v in criterion.items():
        stats[k] = AverageMeter()

    with torch.no_grad():
        queries, gallery, img_labels = [], [], []
        for step, (data, targets, labels) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)

            output_img = img_model(data)
            output_str = str_model(targets)

            queries.append(output_str)
            gallery.append(output_img)
            img_labels.append(list(labels))
        gallery_labels = np.concatenate(img_labels)
        gallery = torch.cat(gallery)

        query_labels, idx_queries = np.unique(gallery_labels, return_index=True)
        queries = torch.cat(queries)
        queries = queries[idx_queries]

        for k, criterion_func in criterion.items():
            stats[k].update(criterion_func(queries, query_labels, gallery=gallery, gallery_labels=gallery_labels))

    stats_str = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
    print(f'\n* TEST set: {stats_str}')


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)

    train_file, val_file, test_file = build_dataset(args.dataset, args.data_path)

    train_loader = DataLoader(
        dataset=train_file,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_file,
        batch_size=args.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_file,
        batch_size=args.batch_size,
        shuffle=False
    )

    img_model = ImageEmbedding(args.out_dim).to(device)
    str_model = StringEmbedding(args.out_dim, train_file.voc_size()).to(device)
    optim = torch.optim.Adam([
        {'params': img_model.parameters()},
        {'params': str_model.parameters()}
    ], args.learning_rate)
    similarity = CosineSimilarityMatrix()

    lossf = {}
    if args.loss == 'ndcg':
        lossf['ndcg_loss'] = DGCLoss(k=args.tau, penalize=args.penalize, similarity=similarity)
    elif args.loss == 'map':
        lossf['map_loss'] = MAPLoss(k=args.tau, similarity=similarity)
    elif args.loss == 'combine':
        lossf['ndcg_loss'] = DGCLoss(k=args.tau, penalize=args.penalize, similarity=similarity)
        lossf['map_loss'] = MAPLoss(k=args.tau, similarity=similarity)
    else:
        raise ValueError(f'loss {args.loss} not supported')

    criterion = {
        'ndcg' : ndcg,
        'map' : meanavep,
    }

    for epoch in range(1, args.epochs+1):
        train(img_model, str_model, device, train_loader, optim, lossf, epoch )
        test(img_model, str_model, device, val_loader, lossf, criterion, epoch)
    test(img_model, str_model, device, test_loader, lossf, criterion, epoch)

if __name__ == '__main__':
    import argparse
    args = get_args_parser()
    main(args)

