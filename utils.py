import warnings
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein

from sklearn.metrics import ndcg_score, average_precision_score

def sigmoid(x, k=1.0):
    exponent = -x/k
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1./(1. + torch.exp(exponent))
    return y

def show(img, title=''):
    npimg = img.numpy()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.draw()
    plt.pause(0.02)

def show_batch(images, embeddings, title=''):
    # Distance matrix
    dm = torch.abs(embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).sum(-1)
    dm_sorted, dm_indices = dm.sort(1)
    images = images[dm_indices].view(-1, *images.shape[1:])
    show(make_grid(images, nrow = dm.shape[0], padding = 0).cpu(), title)


class CosineSimilarityMatrix(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityMatrix, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return cosine_similarity_matrix(x1, x2, self.dim, self.eps)

def cosine_similarity_matrix(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    '''
    When using cosine similarity the constant value must be positive
    '''
    #Cosine sim:
    xn1, xn2 = torch.norm(x1, dim=dim), torch.norm(x2, dim=dim)
    x1 = x1 / torch.clamp(xn1, min=eps).unsqueeze(dim)
    x2 = x2 / torch.clamp(xn2, min=eps).unsqueeze(dim)
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(1)

    sim = torch.tensordot(x1, x2, dims=([2], [2])).squeeze()

    sim = (sim + 1)/2 #range: [-1, 1] -> [0, 2] -> [0, 1]

    return sim

def meanavep(queries, labels, gallery=None, gallery_labels=None, reducefn='mean'):
    # Similarity matrix
    if gallery is not None:
        ranking = cosine_similarity_matrix(queries, gallery)
    else:
        ranking = cosine_similarity_matrix(queries, queries)
        mask_diagonal = ~ torch.eye(ranking.shape[0]).bool()
        ranking = ranking[mask_diagonal].view(ranking.shape[0], ranking.shape[0]-1)
        gallery_labels = labels

    # Ground-truth comparison
    gt = torch.zeros((labels.shape[0], gallery_labels.shape[0]), device=queries.device)
    for i, str1 in enumerate(labels):
        for j, str2 in enumerate(gallery_labels):
            gt[i,j] = str1 == str2

    if gallery is None:
        gt = gt[mask_diagonal].view(gt.shape[0], gt.shape[0]-1)

    ap_sklearn = []
    for y_gt, y_scores in zip(gt, ranking):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = average_precision_score(y_gt.cpu(), y_scores.cpu())
        if not np.isnan(ap):
            ap_sklearn.append(ap)

    if reducefn == 'mean':
        return np.mean(ap_sklearn)
    elif reducefn == 'sum':
        return np.sum(ap_sklearn)
    elif reducefn == 'none':
        return ap_sklearn

def ndcg(queries, labels, gallery=None, gallery_labels=None, reducefn='mean', penalize: bool = False):
    # Similarity matrix
    if gallery is not None:
        ranking = cosine_similarity_matrix(queries, gallery)
    else:
        ranking = cosine_similarity_matrix(queries, queries)
        mask_diagonal = ~ torch.eye(ranking.shape[0]).bool()
        ranking = ranking[mask_diagonal].view(ranking.shape[0], ranking.shape[0]-1)
        gallery_labels = labels

    # Ground-truth Ranking function
    gt = torch.zeros((labels.shape[0], gallery_labels.shape[0]), device=queries.device)
    for i, str1 in enumerate(labels):
        for j, str2 in enumerate(gallery_labels):
            gt[i,j] = Levenshtein.distance(str1, str2)

    if gallery is None:
        gt = gt[mask_diagonal].view(gt.shape[0], gt.shape[0]-1)

    relevance = 10. / (gt + 1)

    ndcg_sk = []
    for y_gt, y_scores in zip(relevance, ranking):
        y_scores_np = np.asarray([y_scores.cpu().numpy()])
        y_gt_np = np.asarray([y_gt.cpu().numpy()])
        ndcg_sk.append(ndcg_score(y_gt_np, y_scores_np))

    if reducefn == 'mean':
        return np.mean(ndcg_sk)
    elif reducefn == 'sum':
        return np.sum(ndcg_sk)
    elif reducefn == 'none':
        return ndcg_sk

