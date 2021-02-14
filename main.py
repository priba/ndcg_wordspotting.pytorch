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
from utils import CosineSimilarityMatrix, cosine_similarity_matrix, ndcg, meanavep, CollateFn
from datasets import build as build_dataset

import pandas as pd
import seaborn as sns
import string

from options import get_args_parser

from models import PHOCNet, ResNet12, StringEmbedding
from logger import AverageMeter
import Levenshtein
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def train(img_model, str_model, device, train_loader, optim, lossf, loss_weights, similarity, epoch):
    img_model.train()
    str_model.train()

    stats = {
        'train_loss': AverageMeter(),
        'img_loss': AverageMeter(),
        'str_loss': AverageMeter(),
        'cross_loss': AverageMeter(),
    }

    start_time = time.time()
    for step, (data, targets, labels, _) in enumerate(train_loader):
        optim.zero_grad()

        data, targets = data.to(device), targets.to(device)

        # Batch size
        bz = len(labels)

        output_img = img_model(data)
        output_str = str_model(targets)

        # Similarity matrices
        ranking_img = similarity(output_img, output_img)
        ranking_str = similarity(output_str, output_str)
        ranking_cross = similarity(output_str, output_img)

        mask_diagonal = ~ torch.eye(ranking_img.shape[0]).bool()

        # Ground-truth Ranking function
        gt = torch.zeros((ranking_img.shape[0], ranking_img.shape[1]), device=ranking_img.device)
        for i, str1 in enumerate(labels):
            for j, str2 in enumerate(labels[i+1:], start=i+1):
                gt[i,j] = gt[j, i] = Levenshtein.distance(str1, str2)

        loss_img, loss_str, loss_cross = 0,0,0
        for k,loss_func in lossf.items():
            if k == 'l1_loss':
                # Regularization
                loss_cross += loss_weights[k]*loss_func(output_str, output_img)
                continue
            loss_img += loss_weights[k]*loss_func(ranking_img, gt, mask_diagonal=mask_diagonal)
            if k != 'map_loss':
                loss_str += loss_weights[k]*loss_func(ranking_str, gt, mask_diagonal=mask_diagonal)
            loss_cross += loss_weights[k]*loss_func(ranking_cross, gt)

        loss = loss_img + loss_str + loss_cross

        loss.backward()
        optim.step()

        stats['train_loss'].update(loss.item(), bz)
        stats['img_loss'].update(loss_img.item(), bz)
        if torch.is_tensor(loss_str):
            loss_str = loss_str.item()
        stats['str_loss'].update(loss_str, bz)
        stats['cross_loss'].update(loss_cross.item(), bz)

        if step % 1500 == 0:
            log_stats = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
            print(
                        f'EPOCH: {epoch}',
                        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
                        f'LOSS: {log_stats}',
            )


    log_stats = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
    print(
                f'EPOCH: {epoch}',
                f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}',
                f'LOSS: {log_stats}',
    )
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}', end='\n')

    return stats


def test(img_model, str_model, device, test_loader, lossf, criterion):
    img_model.eval()
    str_model.eval()

    start_time = time.time()

    stats = {}
    for k, v in criterion.items():
        stats[k] = AverageMeter()
        stats[f'img_{k}'] = AverageMeter()
        if k != 'map':
            stats[f'str_{k}'] = AverageMeter()

    with torch.no_grad():
        queries, gallery, img_labels, img_id = [], [], [], []
        for step, (data, targets, labels, word_id) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)

            output_img = img_model(data)
            output_str = str_model(targets)

            queries.append(output_str)
            gallery.append(output_img)
            img_labels.append(list(labels))
            img_id.append(list(word_id))
        gallery_labels = np.concatenate(img_labels)
        gallery = torch.cat(gallery)

        query_labels, idx_queries = np.unique(gallery_labels, return_index=True)
        queries = torch.cat(queries)
        queries = queries[idx_queries]

        if test_loader.dataset._dataset == 'IAM':
            idx_without_stopwords = [i for i, w in enumerate(query_labels) if (w not in stopwords.words()) and (w not in string.punctuation)]
            query_labels = query_labels[idx_without_stopwords]
            queries = queries[idx_without_stopwords]

            idx_without_stopwords = [i for i, w in enumerate(gallery_labels) if (w not in stopwords.words()) and (w not in string.punctuation)]
            query_gallery_labels = gallery_labels[idx_without_stopwords]
            query_gallery = gallery[idx_without_stopwords]
        else:
            query_gallery_labels = gallery_labels
            query_gallery = gallery

        # Ground-truth Ranking function
        gt_cross = torch.zeros((query_labels.shape[0], gallery_labels.shape[0]), device=queries.device)
        gt_img = torch.zeros((query_gallery_labels.shape[0], gallery_labels.shape[0]), device=queries.device)
        gt_str = torch.zeros((query_labels.shape[0], query_labels.shape[0]), device=queries.device)

        for i, str1 in enumerate(query_labels):
            for j, str2 in enumerate(gallery_labels):
                if str1 in string.punctuation or str2 in string.punctuation:
                    gt_cross[i,j] = 15
                    continue
                gt_cross[i,j] = Levenshtein.distance(str1, str2)

            for j, str2 in enumerate(query_labels):
                if str1 in string.punctuation or str2 in string.punctuation:
                    gt_str[i,j] = 15
                    continue
                gt_str[i,j] = Levenshtein.distance(str1, str2)

        for i, str1 in enumerate(query_gallery_labels):
            for j, str2 in enumerate(gallery_labels):
                if str1 in string.punctuation or str2 in string.punctuation:
                    gt_str[i,j] = 15
                    continue
                gt_img[i,j] = Levenshtein.distance(str1, str2)

        for k, criterion_func in criterion.items():
            stats[k].update(criterion_func(queries, gt_cross, gallery=gallery))
            stats[f'img_{k}'].update(criterion_func(query_gallery, gt_img, gallery, drop_first=True))
            if k != 'map':
                stats[f'str_{k}'].update(criterion_func(queries, gt_str))

        img_id = np.concatenate(img_id)

    stats_str = [f'{k}: {v.avg:.4f}' for k,v in stats.items()]
    print(f'\n* TEST set: {stats_str}')
    end_time = time.time()
    print(f'TOTAL-TEST-TIME: {round(end_time-start_time)}', end='\n')
    return stats, (queries, query_labels), (gallery, gallery_labels, img_id)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.cuda.manual_seed(args.seed)

    train_file, val_file, test_file = build_dataset(args.dataset, args.data_path, partition=args.partition)

    train_sampler = WeightedRandomSampler(weights=train_file.balance_weigths(), num_samples=15000, replacement = True)

    collate = CollateFn(train_file.voc_size())
    train_loader = DataLoader(
        dataset=train_file,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate.collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate.collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_file,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate.collate_fn,
        pin_memory=True
    )

    if args.arch == 'phoc':
        img_model = PHOCNet(args.out_dim).to(device)
    elif args.arch == 'resnet':
        img_model = ResNet12(args.out_dim).to(device)
    else:
        raise ValueError(f'architecture {args.arch} not supported')

    str_model = StringEmbedding(args.out_dim, train_file.voc_size()).to(device)

    optim = torch.optim.Adam([
        {'params': img_model.parameters()},
        {'params': str_model.parameters()}
    ], args.learning_rate)
    scheduler = ReduceLROnPlateau(optim, 'max', factor=0.5, patience=15, cooldown=5, min_lr=1e-6, verbose=True)

    similarity = CosineSimilarityMatrix()

    lossf, loss_weights = {}, {}
    lossf['l1_loss'] = L1Loss()
    loss_weights['l1_loss'] = 0.01
    if args.loss == 'ndcg':
        lossf['ndcg_loss'] = DGCLoss(k=args.tau, penalize=args.penalize)
        loss_weights['ndcg_loss'] = 1
    elif args.loss == 'map':
        lossf['map_loss'] = MAPLoss(k=args.tau)
        loss_weights['map_loss'] = 1
    elif args.loss == 'combine':
        lossf['ndcg_loss'] = DGCLoss(k=args.tau, penalize=args.penalize)
        lossf['map_loss'] = MAPLoss(k=args.tau)
        loss_weights['ndcg_loss'] = 1
        loss_weights['map_loss'] = 1
    else:
        raise ValueError(f'loss {args.loss} not supported')

    criterion = {
        'ndcg' : ndcg,
        'map' : meanavep,
    }

    if args.save is not None and not args.test:
        writer = SummaryWriter(log_dir=args.save)
        # Test transforms
        for i in range(5):
            image, _, _, _ = train_file[0]
            writer.add_image('Train/Images', image, i)

            image, _, _, _ = test_file[0]
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
            train_stats = train(img_model, str_model, device, train_loader, optim, lossf, loss_weights, similarity, epoch)
            val_stats, str_embedding, img_embedding = test(img_model, str_model, device, val_loader, lossf, criterion)

            scheduler.step(val_stats['ndcg'].avg + val_stats['map'].avg)

            if args.save is not None:
                # Train
                writer.add_scalar('Loss/Global', train_stats['train_loss'].avg, epoch)
                writer.add_scalar('Loss/Image', train_stats['img_loss'].avg, epoch)
                writer.add_scalar('Loss/String', train_stats['str_loss'].avg, epoch)
                writer.add_scalar('Loss/Cross', train_stats['cross_loss'].avg, epoch)

                # Test
                writer.add_scalar('TestQbS/NDCG', val_stats['ndcg'].avg, epoch)
                writer.add_scalar('TestQbS/MAP', val_stats['map'].avg, epoch)

                writer.add_scalar('TestQbE/NDCG', val_stats['img_ndcg'].avg, epoch)
                writer.add_scalar('TestQbE/MAP', val_stats['img_map'].avg, epoch)

                writer.add_scalar('TestSED', val_stats['str_ndcg'].avg, epoch)

                # Confusion Matrix
                sed = torch.zeros((str_embedding[0].shape[0], img_embedding[0].shape[0]), device=str_embedding[0].device)
                for i, str1 in enumerate(str_embedding[1]):
                    for j, str2 in enumerate(img_embedding[1]):
                        sed[i,j] = Levenshtein.distance(str1, str2)
                distance = cosine_similarity_matrix(str_embedding[0],img_embedding[0])

                sed = sed.view(-1).tolist()
                distance = distance.view(-1).tolist()
                fig = plt.figure()
                plt.scatter(sed, distance)
                ax = plt.gca()
                ax.set_ylim(0,1)
                plt.xlabel('String Edit Distance')
                plt.ylabel('Learned Similarity')

                writer.add_figure('Correlation', fig, global_step = epoch)

                data = []
                sed, distance = np.array(sed), np.array(distance)
                for used in np.unique(sed):
                   data.append(distance[used == sed])

                fig = plt.figure()
                plt.boxplot(data)
                ax = plt.gca()
                ax.set_ylim(0,1)
                plt.xlabel('String Edit Distance')
                plt.ylabel('Learned Similarity')

                writer.add_figure('Box Plot', fig, global_step = epoch)

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

    test_stats, str_embedding, img_embedding = test(img_model, str_model, device, test_loader, lossf, criterion)

    if args.save is not None:
        # Embedding
        header = str_embedding[1].shape[0]*['String/'] + img_embedding[1].shape[0]*['Image/']
        embedding = torch.cat((str_embedding[0], img_embedding[0]))
        metadata = np.concatenate((str_embedding[1], img_embedding[1]))
        metadata = np.core.defchararray.add(header, metadata)

        with open(os.path.join(args.save, 'str_embedding.pickle'), 'wb') as handle:
            pickle.dump(str_embedding, handle)
        with open(os.path.join(args.save, 'img_embedding.pickle'), 'wb') as handle:
            pickle.dump(img_embedding, handle)

        # Confusion Matrix
        sed = torch.zeros((str_embedding[0].shape[0], img_embedding[0].shape[0]), device=str_embedding[0].device)
        for i, str1 in enumerate(str_embedding[1]):
            for j, str2 in enumerate(img_embedding[1]):
                sed[i,j] = Levenshtein.distance(str1, str2)
        distance = cosine_similarity_matrix(str_embedding[0],img_embedding[0])

        sed = sed.view(-1).tolist()
        distance = distance.view(-1).tolist()

        data = []
        sed, distance = np.array(sed), np.array(distance)
        for used in np.unique(sed):
           data.append(distance[used == sed])

        fig = plt.figure()
        plt.boxplot(data)
        ax = plt.gca()
        ax.set_ylim(0,1)
        ax.set_xticklabels(list(range(0,int(max(sed))+1)))
        plt.xlabel('String Edit Distance')
        plt.ylabel('Learned Similarity')
        plt.savefig(os.path.join(args.save, 'boxplot_test.png'))


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

