from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import random
import os
import numpy as np
import pandas as pd
from .builder import DATASETS

@DATASETS.register_module()
class XinanDataset(BaseDataset):

    CLASSES = ['viral', 'cov']

    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = os.path.join(self.data_prefix, self.ann_file)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        lines =  [line.strip().split(',') for line in lines[1:]]
        #IDs, paths, labels = list(map(list, zip(*lines)))
        data_infos = []
        for index in range(len(lines)):
            ID, path, label = lines[index]
            filename = os.path.join(self.data_prefix, path)
            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array(float(label), dtype=np.int64)
            data_infos.append(data_info)
        return data_infos

    def backup_load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        # 1. load info
        ann_path = os.path.join(self.data_prefix, self.sub_set)
        info = pd.read_csv(ann_path, index_col='index')
        self.info = info[info['malignancy_mode']!=3]

        # 2. load names
        self.names = pd.read_csv(os.path.join(self.data_prefix, self.ann_file))
        if self.test_mode:
            self.names = self.names['test'].dropna()
        else:
            self.names = self.names['train'].dropna()
        self.names = pd.merge(
                pd.Series(self.info.index.map(lambda x:x+'.npz'), name='index'),
                pd.Series(self.names, name='index'))['index'].unique()
        self.cls_map = {'1':0, '2':0, '4':1, '5':1}
        # 3. generate data_infos
        data_infos = []
        for index in range(len(self.names)):
            filename = os.path.join(self.data_prefix, 'nodule', self.names[index])
            gt_label = self.cls_map[str(self.info.loc[self.names[index][:-4], 'malignancy_mode'])]
            info = {}
            info['img_prefix'] = self.data_prefix
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    #def __len__(self):
        #return len(self.names)


