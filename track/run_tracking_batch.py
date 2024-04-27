import argparse
from functools import partial
import os
import os.path as osp
import time
import cv2
import json
import numpy as np
import sys
from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker,TrackState
from tqdm import tqdm
import copy

def main():
    scene_name = sys.argv[1]
    
    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)
    
    det_dir = osp.join(root_path,"result/detection", scene_name)
    pose_dir = osp.join(root_path,"result/pose", scene_name)
    reid_dir = osp.join(root_path,"result/reid", scene_name)
    
    cal_dir = osp.join(root_path,'dataset/test', scene_name)
    save_dir = os.path.join(root_path,"result/track")
    save_path = osp.join(save_dir, scene_name+".txt")
    # if os.path.exists(save_path):
    #     print("exit",scene_name)
    #     exit()
    
    cams = sorted(os.listdir(cal_dir))
    cals = []
    for cam in cams:
        cals.append(Camera(osp.join(cal_dir,cam,"calibration.json")))

    det_data=[]
    files = sorted(os.listdir(det_dir))
    files = [f for f in files if f[0]=='c']
    for f in files:
        if f[0]=='c':       
            det_data.append(np.loadtxt(osp.join(det_dir,f), delimiter=","))
    
    pose_data=[]
    files = sorted(os.listdir(pose_dir))
    files = [f for f in files if f[0]=='c']
    for f in files:
        pose_data.append(np.loadtxt(osp.join(pose_dir,f)))

    reid_data = []
    files = sorted(os.listdir(reid_dir))
    files = [f for f in files if f[0]=='c']
    for f in files:
        reid_data_scene=np.load(osp.join(reid_dir,f),mmap_mode='r')
       
        if len(reid_data_scene):
            reid_data_scene=reid_data_scene/np.linalg.norm(reid_data_scene, axis=1,keepdims=True)
        reid_data.append(reid_data_scene)
    
    print("reading finish")

    
    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:,0]))
    max_frame = int(np.max(max_frame))

    tracker = PoseTracker(cals)
    box_thred = 0.3
    results = []
    #default_reid = np.zeros(2048)

    for frame_id in tqdm(range(max_frame+1),desc = scene_name):
        # if frame_id<1720:
        #     continue
        # if frame_id>1780:
        #     break
        detection_sample_mv = []
        for v in range(tracker.num_cam):
            detection_sample_sv = []
            det_sv = det_data[v]
            if len(det_sv)==0:
                detection_sample_mv.append(detection_sample_sv)
                continue
            idx = det_sv[:,0]==frame_id
            cur_det = det_sv[idx]
            cur_pose = pose_data[v][idx]
            cur_reid = reid_data[v][idx]

            for det, pose, reid in zip(cur_det, cur_pose, cur_reid):
                if det[-1]<box_thred or len(det)==0:
                    continue
                new_sample = Detection_Sample(bbox=det[2:],keypoints_2d=pose[6:].reshape(17,3), reid_feat=reid, cam_id = v, frame_id=frame_id)
                detection_sample_sv.append(new_sample)
            detection_sample_mv.append(detection_sample_sv)

        #import pdb;pdb.set_trace();

        print("frame {}".format(frame_id),"det nums: ",[len(L) for L in detection_sample_mv])

        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)
        print(len([t for t in tracker.tracks if t.state == TrackState.Confirmed]))

        
        frame_results = tracker.output(frame_id)
        results += frame_results
        
        # if frame_id>=2000:
        #     break
        
    results = np.concatenate(results,axis=0)
    sort_idx = np.lexsort((results[:,2],results[:,0]))
    results = np.ascontiguousarray(results[sort_idx])
    np.savetxt(save_path, results)
    
        # if frame_id>=0:
        #     import pdb;pdb.set_trace();


if __name__ == '__main__':
    main()

        






    

    






    


