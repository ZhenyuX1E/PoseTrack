# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import time
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def infer_one_image(args,frame, bboxes_s, pose_estimator):
    #bboxes_s = bboxes_s[bboxes_s[:,4] > args.bbox_thr]
    pose_results = inference_topdown(pose_estimator, frame, bboxes_s[:,:4])
    records=[]
    for i, result in enumerate(pose_results):
        keypoints = result.pred_instances.keypoints[0]
        scores = result.pred_instances.keypoint_scores.T
        record = (np.concatenate((keypoints,scores),axis=1)).flatten()
        records.append(record)
    records = np.array(records)
    records = np.concatenate((bboxes_s,records),axis=1)
    return records
        


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--start', type=int,default=0)
    parser.add_argument(
        '--end', type=int,default=-1)

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-3]
    root_path = '/'.join(path_arr)
    
    det_root = os.path.join(root_path,"result/detection")
    vid_root = os.path.join(root_path,"dataset/test")
    save_root = os.path.join(root_path,"result/pose")
    scenes = sorted(os.listdir(det_root))
    scenes = [s for s in scenes if s[0]=="s"]
    scenes = scenes[args.start:args.end]
    for scene in tqdm(scenes):
        print(scene)
        det_dir = os.path.join(det_root, scene)
        vid_dir = os.path.join(vid_root, scene)
        save_dir = os.path.join(save_root, scene)
        cams = os.listdir(vid_dir)
        cams = sorted([c for c in cams if c[0]=="c"])
        
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        for cam in tqdm(cams):
            print(cam)
            det_path = os.path.join(det_dir,cam)+".txt"
            vid_path = os.path.join(vid_dir,cam)+"/video.mp4"
            save_path = os.path.join(save_dir,cam+".txt")
            if os.path.exists(save_path):
                continue
            det_annot = np.loadtxt(det_path,delimiter=",")
            #print(det_annot[0])

            frame_id = 0
            ret = True
            #cap=cv2.VideoCapture(vid_path)
            video = mmcv.VideoReader(vid_path)
            #all_results = np.zeros((0,57),dtype=np.float32)
            all_results = []
            line_idx = 0
            
            det_len = len(det_annot)
            #while ret and frame_id<10:
            for frame_id, frame in enumerate(tqdm(video)):
                num_det = 0
                # while det_annot[line_idx,0]<frame_id:
                #     line_idx += 1
                # while line_idx + num_det < det_len and det_annot[line_idx + num_det,0]==frame_id:
                #     num_det += 1

                # if det_annot[line_idx,0]>frame_id:
                #     continue

                # dets = det_annot[line_idx:line_idx+num_det]

                #ret, frame = cap.read()
                dets = det_annot[det_annot[:,0]==frame_id]
                bboxes_s = dets[:,2:7] #x1y1x2y2s
                if len(bboxes_s)==0:
                    continue
                #t1 = time.time()
                result = infer_one_image(args, frame, bboxes_s, pose_estimator)
                #t2 = time.time()
                #print(t2-t1,"  ",len(bboxes_s))
                result = np.concatenate((np.ones((len(result),1))*frame_id,result.astype(np.float32)),axis=1)
                all_results.append(result)
            all_results = np.concatenate(all_results)

                # if frame_id%10==0:
                #     print(frame_id)
                # frame_id+=1
            np.savetxt(save_path, all_results)
    # cap.release()


if __name__ == '__main__':
    main()
