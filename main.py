import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from loss import DGCLoss, MAPLoss, L1Loss
from utils import CosineSimilarityMatrix, cosine_similarity_matrix, ndcg, meanavep, collate_fn
from datasets import build as build_dataset

import pandas as pd
import seaborn as sns

from options import get_args_parser

from models import ImageEmbedding, StringEmbedding
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
        bz = len(labels)

        output_img = img_model(data)
        output_str = str_model(targets)

        loss_img, loss_str, loss_cross = 0,0,0
        for k,loss_func in lossf.items():
            if k == 'l1_loss':
                # Regularization
                loss_cross += 0.005*loss_func(output_str, output_img)
                continue
            loss_img += loss_func(output_img, labels)
            if k != 'map_loss':
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

    return stats


def test(img_model, str_model, device, test_loader, lossf, criterion):
    img_model.eval()
    str_model.eval()

    stats = {}
    for k, v in criterion.items():
        stats[k] = AverageMeter()
        stats[f'img_{k}'] = AverageMeter()
        if k != 'map':
            stats[f'str_{k}'] = AverageMeter()

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
            stats[f'img_{k}'].update(criterion_func(gallery, gallery_labels))
            if k != 'map':
                stats[f'str_{k}'].update(criterion_func(queries, query_labels))


    stats_str = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
    print(f'\n* TEST set: {stats_str}')
    return stats


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.manual_seed(args.seed)

    train_file, val_file, test_file = build_dataset(args.dataset, args.data_path, args.partition)

    train_sampler = WeightedRandomSampler(weights= num_samples=, replacement = True)
    train_loader = DataLoader(
        dataset=train_file,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset=test_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    img_model = ImageEmbedding(args.out_dim).to(device)
    str_model = StringEmbedding(args.out_dim, train_file.voc_size()).to(device)

    optim = torch.optim.Adam([
        {'params': img_model.parameters()},
        {'params': str_model.parameters()}
    ], args.learning_rate)
    scheduler = ReduceLROnPlateau(optim, 'max', patience=25, cooldown=5, min_lr=1e-6, verbose=True)

    similarity = CosineSimilarityMatrix()

    lossf = {}
    lossf['l1_loss'] = L1Loss()
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

    if args.save is not None:
        writer = SummaryWriter(log_dir=args.save)
        # Test transforms
        for i in range(5):
            image, _, _ = train_file[0]
            writer.add_image('Train/Images', image, i)

            image, _, _ = test_file[0]
            writer.add_image('Test/Images', image, i)

    best_stats = {'ndcg': 0, 'map': 0}
    start_epoch = 1
    early_stop_counter = 0
    if args.load is not None:
        checkpoint = torch.load(args.load, map_location=device)
        img_model.load_state_dict(checkpoint['img_state_dict'])
        str_model.load_state_dict(checkpoint['str_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        best_stats = checkpoint['best_stats']
        start_epoch = checkpoint['epoch'] 

        print(f'Model load at epoch {start_epoch}\n\t* NDCG = {best_stats["ndcg"]}\n\t* MAP = {best_stats["map"]}\n')


    if not args.test:
        for epoch in range(1, args.epochs+1):
            train_stats = train(img_model, str_model, device, train_loader, optim, lossf, epoch )
            val_stats = test(img_model, str_model, device, val_loader, lossf, criterion)

            scheduler.step(val_stats['ndcg'].avg + val_stats['map'].avg)

            if args.save is not None:
                # Train
                writer.add_scalar('Loss/Global', train_stats['train_loss'].avg, epoch)
                writer.add_scalar('Loss/Image', train_stats['train_loss'].avg, epoch)
                writer.add_scalar('Loss/String', train_stats['train_loss'].avg, epoch)
                writer.add_scalar('Loss/Cross', train_stats['train_loss'].avg, epoch)

                # Test
                writer.add_scalar('TestQbS/NDCG', val_stats['ndcg'].avg, epoch)
                writer.add_scalar('TestQbS/MAP', val_stats['map'].avg, epoch)
                writer.add_scalar('TestQbE/NDCG', val_stats['img_ndcg'].avg, epoch)
                writer.add_scalar('TestSED', val_stats['str_ndcg'].avg, epoch)

                # Learning rate
                writer.add_scalar('Learning Rate', optim.param_groups[0]['lr'], epoch)

            es_count = 1
            if val_stats['ndcg'].avg > best_stats['ndcg']:
                best_stats['ndcg'] = val_stats['ndcg'].avg
                early_stop_counter, es_count = 0, 0
                if args.save is not None:
                    torch.save({
                        'epoch': epoch, 
                        'img_state_dict': img_model.state_dict(), 
                        'str_state_dict': str_model.state_dict(), 
                        'best_stats': best_stats,
                        'optimizer': optim.state_dict(),
                        }, os.path.join(args.save, 'checkpoint_ndcg.pth'))

            if val_stats['map'].avg > best_stats['map']:
                best_stats['map'] = val_stats['map'].avg
                early_stop_counter, es_count = 0, 0
                if args.save is not None:
                    torch.save({
                        'epoch': epoch, 
                        'img_state_dict': img_model.state_dict(), 
                        'str_state_dict': str_model.state_dict(), 
                        'best_stats': best_stats,
                        'optimizer': optim.state_dict(),
                        }, os.path.join(args.save, 'checkpoint_map.pth'))

            early_stop_counter += es_count
            if early_stop_counter >= args.early_stop:
                print('Early Stop at epoch {}'.format(epoch))
                break


    if args.save is not None and not args.test:
        checkpoint = torch.load(os.path.join(args.save, 'checkpoint_map.pth'), map_location=device)
        img_model.load_state_dict(checkpoint['img_state_dict'])
        str_model.load_state_dict(checkpoint['str_state_dict'])

    test(img_model, str_model, device, test_loader, lossf, criterion)

if __name__ == '__main__':
    import argparse
    args = get_args_parser()

    # Check Test and Load
    if args.test and args.load is None:
        raise Exception('Cannot test without loading a model.')

    if args.save is not None and not args.test:
        if not os.path.isdir(args.save):
            os.makedirs(args.save)

    main(args)

