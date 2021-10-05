# encoding: utf-8
import os.path as osp
from collections import defaultdict
import random

statistic_message = """f'''Dataset statistics({self.__class__.__name__}):
  {'-'*40}  
  subset   | # ids | # images | # cameras
  {'-'*40}  
  train    | {self.num_train_pids:5d} | {self.num_train_imgs:8d} | {self.num_train_cams:5d}
  query    | {self.num_query_pids:5d} | {self.num_query_imgs:8d} | {self.num_query_cams:5d}
  gallery  | {self.num_gallery_pids:5d} | {self.num_gallery_imgs:8d} | {self.num_gallery_cams:5d}
  {'-'*40}  '''"""

class BaseDataset:
    """
    Base class of reid dataset which should be composed of three sub 
    directories, including train_dir/, query_dir/ and gallery_dir/
    """
    def __init__(self, root, dirname, subdirnames, verbose):
        self.dataset_dir = osp.join(root, dirname)
        self.verbose = verbose
        self.train_dir = osp.join(self.dataset_dir, subdirnames['train'])
        self.query_dir = osp.join(self.dataset_dir, subdirnames['query'])
        self.gallery_dir = osp.join(self.dataset_dir, subdirnames['gallery'])
        
        self._check_before_run()
        self.loading()
        
    def loading(self): #loading or reloading
        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.get_dataset_statistics(self.verbose)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel): #@abc.abstractmethod is recommended
        '''Main processing function'''
        raise NotImplementedError(f'class {self.__class__.__name__} '
            'doesn\'t implement the parent(BaseDataset) method _process_dir(dir_path, relabel)!')

    def _get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_dataset_statistics(self, printing):
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self._get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self._get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self._get_imagedata_info(self.gallery)

        self.statistic = eval(statistic_message, {'self': self})
        if printing:
            print(self.statistic)
