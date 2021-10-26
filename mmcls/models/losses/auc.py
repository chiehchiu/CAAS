import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import hamming_loss, accuracy_score, roc_auc_score, roc_curve, auc
import pdb
import sys

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def auc_multi_cls(pred, target):
    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
        device_id = torch.cuda.current_device()
        pred = torch.sigmoid(pred)
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        # ForkedPdb().set_trace()
        # accuracy_pre_label = torch.tensor(1.0 - hamming_loss(target, pred)).cuda(device_id)
        # accuracy_pre_sample = torch.tensor(accuracy_score(target, pred)).cuda(device_id)

        auc_total = 0.0
        cls_num = pred.shape[1]
        for idx in range(cls_num):
            try:
                _auc = roc_auc_score(target[:, idx], pred[:, idx])
            # ValueError: Only one class present in y_true.ROC AUC score is not defined in that case.
            # target[:, idx] = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            except ValueError:
                _auc = 0.5

            auc_total += _auc

        auc_mean = auc_total / cls_num
        auc_mean = torch.tensor(auc_mean).cuda(device_id)
        # auc_mean = torch.tensor(roc_auc_score(target, pred)).cuda(device_id)

    return auc_mean


# LJ
class Auc(nn.Module):

    def __init__(self):
        """Module to calculate the auc

        """
        super().__init__()


    def forward(self, pred, target):
        """Forward function to calculate accuracy

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        # LJ
        return auc_multi_cls(pred, target)
