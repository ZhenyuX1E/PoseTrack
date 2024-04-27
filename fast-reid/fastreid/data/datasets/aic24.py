# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re
import os

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class AIC24(ImageDataset):
    """DukeMTMC-reID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'AIC24'
    dataset_name = "aic24"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        train,query,gallery=[],[],[]
        '''img_paths=glob.glob('/mnt/extended/randperson/images/subset/randperson_subset/randperson_subset/*.jpg')
        for img_path in img_paths:
            img_name=img_path.split('/')[-1]
            infos=img_name.split('_')
            pid,camid=int(infos[0]),int(infos[2][1:])
            train.append((img_path, pid, camid))
        img_paths=glob.glob('/root/ywj/fast-reid/datasets/Challenge/challenge_train/*.jpg')
        for img_path in img_paths:
            img_name=img_path.split('/')[-1]
            infos=img_name.split('_')
            pid,camid=int(infos[0]),int(infos[1][1:])
            train.append((img_path, pid+8000, camid))'''

        current_file_path = os.path.abspath(__file__)
        path_arr = current_file_path.split('/')[:-4]
        root_path = '/'.join(path_arr)
        img_paths=glob.glob(root_path+"/"+"dataset/*/*.jpg")
        gallery_id=set()
        for img_path in img_paths:
            scene,img=img_path.split('/')[-2],img_path.split('/')[-1]
            scene=int(scene[-3:])
            info=img.split('_')
            pid,camid,frame_id=int(info[0]),int(info[1]),int(info[-1][:-4])

            if scene in [58,59,60]:
                if frame_id>=3000:
                    gallery.append((img_path, pid+scene*200, camid))
                    gallery_id.add(pid+scene*200)
            else:
                train.append((img_path, pid+9000+scene*200, camid))
        
        for img_path in img_paths:
            scene,img=img_path.split('/')[-2],img_path.split('/')[-1]
            scene=int(scene[-3:])
            info=img.split('_')
            pid,camid,frame_id=int(info[0]),int(info[1]),int(info[-1][:-4])

            if scene in [58,59,60]:
                if frame_id<3000:
                    if pid+scene*200 in gallery_id:
                        query.append((img_path, pid+scene*200, camid+100))
              
        super(AIC24, self).__init__(train, query, gallery, **kwargs)