import copy
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

from mmcls.models.losses import accuracy, cross_entropy, binary_cross_entropy
from .pipelines import Compose

from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score, roc_curve, auc
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

class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base dataset.

    Args:
        data_prefix (str): the prefix of data path
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        ann_file (str | None): the annotation file. When ann_file is str,
            the subclass is expected to read from the ann_file. When ann_file
            is None, the subclass is expected to read according to data_prefix
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self, data_prefix, pipeline, ann_file=None, sub_set=None, test_mode=False):
        super(BaseDataset, self).__init__()

        self.ann_file = ann_file
        self.sub_set = sub_set
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        """Map mapping class name to class index.

        Returns:
            dict: mapping from class name to class index.
        """

        return {_class: i for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            list[int]: categories for all images.
        """
        ## LJ 此处有bug 多卡测试时 gt_labels与results的idx不对应
        gt_labels = np.array([data['gt_label'] for data in self.data_infos])

        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index.

        Args:
            idx (int): Index of data.

        Returns:
            int: Image category of specified index.
        """

        return self.data_infos[idx]['gt_label'].astype(np.int)

    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])

        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options={'topk': (1, 5)},
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict: evaluation results
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        #allowed_metrics = ['accuracy']
        #if metric not in allowed_metrics:
            #raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'accuracy':
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
        elif metric == 'acc_and_auc':
            # pdb.set_trace()
            topk = metric_options.get('topk')
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs
            acc = accuracy(results, gt_labels, topk)
            loss = cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = round(float(loss), 3)
            auc = roc_auc_score(gt_labels, results[:,1])
            eval_results = {f'top-{k}': a.item() for k, a in zip(topk, acc)}
            eval_results['loss'] = loss
            eval_results['auc'] = auc
        # LJ
        elif metric == 'auc_multi_cls':
            results = np.vstack(results)
            gt_labels = self.get_gt_labels()
            num_imgs = len(results)
            assert len(gt_labels) == num_imgs

            loss = binary_cross_entropy(torch.Tensor(results), torch.Tensor(gt_labels).long()).mean()
            loss = float(loss)
            eval_results['loss'] = loss
            # acc_pre_label = 1.0 - hamming_loss(gt_labels, results)
            # acc_pre_sample = accuracy_score(gt_labels, results)
            # eval_results['acc_pre_label'] = acc_pre_label
            # eval_results['acc_pre_sample'] = acc_pre_sample
            '''
            auc_pre_cls = {}
            auc_total = 0.0
            for cls in self.CLASSES:
                idx = self.CLASSES.index(cls)
                _auc = roc_auc_score(gt_labels[:, idx], results[:, idx])
                # fpr, tpr, thresholds = roc_curve(gt_labels[:, idx], results[:, idx], pos_label=1)
                # _auc = auc(fpr, tpr)
                auc_total += _auc
                auc_pre_cls[cls] = round(_auc, 4)
            '''
            eval_results['auc_mean'] = round(roc_auc_score(gt_labels,results[:,1]),4)
            # eval_results['auc_mean'] = roc_auc_score(gt_labels, results)
            #eval_results['auc_mean'] = auc_total / len(self.CLASSES)
            #eval_results['auc_pre_cls'] = auc_pre_cls
            # pdb.set_trace()

        return eval_results
