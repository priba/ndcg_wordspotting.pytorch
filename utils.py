import warnings
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein
from typing import Optional, List

from sklearn.metrics import ndcg_score, average_precision_score


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = torch.stack(batch[1])
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


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

    relevance = torch.clone(gt)

    # Relevance Scores as proposed by Gomez et al. http://www.cvc.uab.cat/~marcal/pdfs/ICDAR17c.pdf
    relevance[gt==0] = 20
    relevance[gt==1] = 15
    relevance[gt==2] = 10
    relevance[gt==3] = 5
    relevance[gt==4] = 3
    relevance[gt>4] = 0

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

