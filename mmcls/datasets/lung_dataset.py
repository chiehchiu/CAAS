from torch.utils.data import Dataset
from .base_dataset import BaseDataset
import random
import os
import numpy as np
import pandas as pd
from .builder import DATASETS
import pdb

@DATASETS.register_module()
class LungDataset(BaseDataset):

    CLASSES = None

    # LJ
    def load_annotations(self):
        '''Overwrite load_annotations func.
        '''
        ann_file = self.ann_file
        with open(ann_file, 'r') as f:
            lines = f.readlines()

        lines =  [line.strip().split(' ') for line in lines]
        #IDs, paths, labels = list(map(list, zip(*lines)))
        data_infos = []
        for index in range(len(lines)):
            ID, labels = lines[index][0], lines[index][1:]
            filename = os.path.join(self.data_prefix, ID)

            _labels = [0] * len(self.CLASSES)
            for lable in labels:
                _labels[self.CLASSES.index(lable)] = 1

            data_info = {}
            data_info['img_info'] = {'filename': filename}
            data_info['img_prefix'] = self.data_prefix
            data_info['gt_label'] = np.array(_labels, dtype=np.int64)
            data_infos.append(data_info)
        return data_infos

    # def __len__(self):
    #     return len(self.data_infos)



@DATASETS.register_module
class Lung9Dataset(LungDataset):

    CLASSES = ['结节肿块', '肺气肿', '肺大疱', '实变影', '网影',
               '磨玻璃密度影', '条索影', '胸腔积液', '气胸']


@DATASETS.register_module
class Lung6Dataset(LungDataset):

    CLASSES = ['结节肿块', '肺气肿', '实变影', '网影', '胸腔积液', '气胸']