import argparse
from functools import partial
import os
import os.path as osp
import time
import cv2
import json
import numpy as np
from tqdm import tqdm


result_dir = "result/track"
save_dir = "result"
if not osp.exists(save_dir):
    os.mkdir(save_dir)

scenes = os.listdir(result_dir)
scenes = sorted([s for s in scenes if s[:5]=='scene'])

files = []
for s in tqdm(scenes):
    files.append(np.loadtxt(osp.join(result_dir,s))[:,:-1])

results = np.concatenate(files,axis=0)

np.savetxt(osp.join(save_dir,"track.txt"), results, fmt="%d %d %d %d %d %d %d %f %f")



