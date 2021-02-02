import warnings

import torch
import torch.nn as nn
from torch import Tensor
from utils import sigmoid, CosineSimilarityMatrix
from collections.abc import Callable

class DGCLoss(nn.Module):
    def __init__(self, k: float = 1e-3, penalize=False, normalize=True, indicator_function: Callable = sigmoid):
        super(DGCLoss, self).__init__()
        self.k = k
        self.penalize = penalize
        self.normalize = normalize
        self.indicator_function = indicator_function

    def forward(self, ranking: Tensor, gt: Tensor, mask_diagonal: Tensor = None) -> Tensor:
        return dgc_loss(ranking, gt, mask_diagonal=mask_diagonal, k=self.k, penalize=self.penalize, normalize=self.normalize, indicator_function=self.indicator_function)

def dgc_loss(ranking: Tensor, gt: Tensor, mask_diagonal: Tensor = None, k: float = 1e-3, penalize: bool = False, normalize: bool = True, indicator_function: Callable = sigmoid) -> Tensor:

    if mask_diagonal is not None:
        ranking = ranking[mask_diagonal].view(ranking.shape[0], ranking.shape[0]-1)
        gt = gt[mask_diagonal].view(gt.shape[0], gt.shape[0]-1)

    # Prepare indicator function
    dij = ranking.unsqueeze(1) - ranking.unsqueeze(-1)
    mask_diagonal = ~ torch.eye(dij.shape[-1]).bool()
    dij = dij[:,mask_diagonal].view(dij.shape[0], dij.shape[1], -1)

    # Indicator function
    # Assuming a perfect step function
    # indicator = (dij > 0).float()
    # indicator = indicator.sum(-1) + 1

    # Smooth indicator function
    indicator = indicator_function(dij, k=k)
    indicator = indicator.sum(-1) + 1

    # Relevance score
#    relevance = 10. / (gt + 1)
    relevance = 4 - gt
    relevance = relevance.clamp(0)

    if penalize:
        relevance = relevance.exp2() - 1

    dcg = torch.sum(relevance / torch.log2(indicator + 1), dim=1)

    if not normalize:
        return -dcg.mean()

    relevance, _ = relevance.sort(descending=True)
    indicator = torch.arange(relevance.shape[-1], dtype=torch.float32, device=relevance.device)
    idcg = torch.sum(relevance / torch.log2(indicator + 2), dim=-1)

    dcg = dcg[idcg!=0]
    idcg = idcg[idcg!=0]

    ndcg = dcg / idcg

    if ndcg.numel() == 0:
        return torch.tensor(1)

    return 1 - ndcg.mean()

class MAPLoss(nn.Module):
    def __init__(self, k: float = 1e-3, indicator_function: Callable = sigmoid):
        super(MAPLoss, self).__init__()
        self.k = k
        self.indicator_function = indicator_function

    def forward(self, ranking: Tensor, gt: Tensor, mask_diagonal: Tensor = None) -> Tensor:
        return map_loss(ranking, gt, mask_diagonal=mask_diagonal, k=self.k, indicator_function=self.indicator_function)


def map_loss(ranking: Tensor, gt: Tensor, mask_diagonal: Tensor = None, k: float = 1e-3, indicator_function: Callable = sigmoid) -> Tensor:

    # Ground-truth comparison
    gt = gt==0

    if mask_diagonal is not None:
        ranking = ranking[mask_diagonal].view(ranking.shape[0], ranking.shape[0]-1)
        gt = gt[mask_diagonal].view(gt.shape[0], gt.shape[0]-1)

    # Prepare indicator function
    dij = ranking.unsqueeze(1) - ranking.unsqueeze(-1)

    # Indicator function
    # Assuming a perfect step function
    #indicator = (dij > 0).float()

    # Smooth indicator function
    indicator = indicator_function(dij, k=k)
    # Remove self comparison
    mask_diagonal = torch.eye(indicator.shape[-1]).bool()
    indicator[:,mask_diagonal] = 0

    accumulated_gt = (gt.unsqueeze(1)*indicator).sum(-1) + 1

    indicator = indicator.sum(-1) + 1
    prec = accumulated_gt / indicator

    ap = torch.sum(gt*prec, dim=1)
    num_positives = gt.sum(-1)

    ap = ap[num_positives>0]
    num_positives = num_positives[num_positives>0]
    ap = ap/num_positives
    if torch.numel(ap) == 0:
        return torch.tensor(1)
    return 1-ap.mean()

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, query: Tensor, gallery: Tensor) -> Tensor:
        return l1_loss(query, gallery)

def l1_loss(query: Tensor, gallery: Tensor):
    loss = torch.abs(query - gallery).sum(-1)
    return loss.mean()

