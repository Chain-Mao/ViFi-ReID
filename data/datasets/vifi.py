# encoding: utf-8

import glob
import os.path as osp
import os
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ViFi(ImageDataset):
    dataset_dir = ''
    dataset_name = "vifi"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'ViFi')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "train" under '
                          '"ViFi".')

        # reid or retrival
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir, is_train=True, type='train')
        query = lambda: self.process_dir(self.query_dir, is_train=False, type='query')
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False, type='gallery')

        super(ViFi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, type='train'):
        vision_paths = [osp.join(osp.join(dir_path, "vision"), d) for d in os.listdir(osp.join(dir_path, "vision"))]
        wifi_paths = glob.glob(osp.join(osp.join(dir_path, "wifi"), '*.npy'))
        vision_paths.sort()
        wifi_paths.sort()
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for vision_path, wifi_path in zip(vision_paths, wifi_paths):
            pid, camid = map(int, pattern.search(wifi_path).groups())
            # 重定义camid
            if type == 'train': camid = 0
            if type == 'query': camid = 1
            if type == 'gallery': camid = 2
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((vision_path, wifi_path, pid, camid))

        return data
