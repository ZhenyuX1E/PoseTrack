from functools import partial
import numpy as np
import cv2
import sys

#sys.path.append("../")
from Tracker.kalman_filter_box_zya import KalmanFilter_box
from Solver.bip_solver import GLPKSolver
from util.camera import *
from util.process import find_view_for_cluster
from scipy.optimize import linear_sum_assignment
from Tracker.matching import *
import lap
import aic_cpp


class TrackState:
    """
    使用TrackState表示追踪轨迹的状态
    """
    Unconfirmed = 1
    Confirmed = 2
    Missing = 3
    Deleted = 4

class Track2DState:
    Vide = 0
    Detected = 1
    Occluded = 2
    Missing = 3


class Detection_Sample():
    def __init__(self, bbox, keypoints_2d, reid_feat, cam_id, frame_id):
        self.bbox = bbox
        self.keypoints_2d = keypoints_2d
        self.reid_feat=reid_feat
        self.cam_id = cam_id
        self.frame_id = frame_id


class PoseTrack2D():
    
    def __init__(self):
        self.state = []

    def init_with_det(self, Detect_Sample):
        if len(self.state)==10:
            self.state.pop()
        self.state = [Track2DState.Detected]+self.state
        self.bbox = Detect_Sample.bbox
        self.keypoints_2d = Detect_Sample.keypoints_2d
        self.reid_feat=Detect_Sample.reid_feat
        self.cam_id=Detect_Sample.cam_id


class PoseTrack():

    def __init__(self, cameras):
        self.cameras = cameras
        self.num_cam = len(cameras)
        self.confirm_time_left = 0
        self.valid_views = [] # valid view input at current time step
        self.bank_size = 100
        self.feat_bank = np.zeros((self.bank_size, 2048))
        self.feat_count = 0
        self.track2ds = [PoseTrack2D() for i in range(self.num_cam)]
        self.num_keypoints = 17
        self.keypoint_thrd = 0.7
        self.update_age = 0
        self.decay_weight = 0.5
        self.keypoints_3d = np.zeros((self.num_keypoints, 4))
        self.keypoints_mv = np.zeros((self.num_cam, self.num_keypoints, 3))
        self.bbox_mv = np.zeros((self.num_cam, 5))
        self.age_2D = np.ones((self.num_cam, self.num_keypoints)) * np.inf
        self.age_3D = np.ones((self.num_keypoints)) * np.inf
        self.age_bbox = np.ones((self.num_cam)) * np.inf
        self.dura_bbox = np.zeros((self.num_cam))
        self.thred_conf_reid = 0.95
        self.feet_idx = np.array([15,16])
        self.bbox_kalman=[KalmanFilter_box() for i in range(self.num_cam)]
        self.thred_reid = 0.5

        self.output_cord = np.zeros(3)
        self.output_priority = [[5,6],[11,12],[15,16]] #[3,4,5,6],[3,4,11,12],
        self.main_joints = np.array([5,6,11,12,13,14,15,16])
        self.upper_body = np.array([5,6,11,12])

        self.sample_buf = []
        self.unit = np.full((self.num_keypoints,3), 1/np.sqrt(3))
        self.iou_mv = [0 for i in range(self.num_cam)]
        self.ovr_mv = [0 for i in range(self.num_cam)]
        self.oc_state = [False for i in range(self.num_cam)]
        self.oc_idx = [[] for i in range(self.num_cam)]
        self.ovr_tgt_mv = [0 for i in range(self.num_cam)]

    def reactivate(self,newtrack):
        self.state = newtrack.state
        self.valid_views = newtrack.valid_views
        self.track2ds = newtrack.track2ds
        self.age_2D = newtrack.age_2D
        self.age_3D = newtrack.age_3D
        self.bbox_mv = newtrack.bbox_mv
        self.age_bbox = newtrack.age_bbox
        self.confirm_time_left = newtrack.confirm_time_left
        self.keypoints_3d = newtrack.keypoints_3d
        self.keypoints_mv = newtrack.keypoints_mv
        self.output_cord = newtrack.output_cord
        self.dura_bbox = newtrack.dura_bbox
        self.bbox_kalman=newtrack.bbox_kalman
        self.update_age = 0

        if newtrack.feat_count>=1:
            if self.feat_count>=self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count%self.bank_size]

            new_bank = newtrack.feat_bank[:newtrack.feat_count%self.bank_size]
            sim = np.max((new_bank @ bank.T), axis=-1)
            sim_idx = np.where(sim<self.thred_reid)[0]
            for id in sim_idx:
                self.feat_bank[self.feat_count%self.bank_size] = new_bank[id].copy()
                self.feat_count+=1

    def switch_view(self, track, v):
        self.track2ds[v], track.track2ds[v] = track.track2ds[v], self.track2ds[v]
        self.age_2D[v], track.age_2D[v] = track.age_2D[v], self.age_2D[v]
        self.keypoints_mv[v], track.keypoints_mv[v] = track.keypoints_mv[v], self.keypoints_mv[v]
        self.age_bbox[v], track.age_bbox[v] = track.age_bbox[v], self.age_bbox[v]
        self.bbox_mv[v], track.bbox_mv[v] = track.bbox_mv[v], self.bbox_mv[v]
        self.dura_bbox[v], track.dura_bbox[v] = track.dura_bbox[v], self.dura_bbox[v]

        self.oc_state[v], track.oc_state[v] = track.oc_state[v], self.oc_state[v]
        self.oc_idx[v], track.oc_idx[v] = track.oc_idx[v], self.oc_idx[v]
        self.bbox_kalman[v], track.bbox_kalman[v] = track.bbox_kalman[v], self.bbox_kalman[v]
        self.iou_mv[v], track.iou_mv[v] = track.iou_mv[v], self.iou_mv[v]
        self.ovr_mv[v], track.ovr_mv[v] = track.ovr_mv[v], self.ovr_mv[v]
        self.ovr_tgt_mv[v], track.ovr_tgt_mv[v] = track.ovr_tgt_mv[v], self.ovr_tgt_mv[v]
        print("switch ", self.id, track.id, v)


    def get_output(self):
        # 3D kp output

        for comb in self.output_priority:
            if all(self.age_3D[comb]==0):
                self.output_cord = np.concatenate((np.mean(self.keypoints_3d[comb,:2],axis=0),[3]))
                return self.output_cord 
        
        #if no 3D kp comb, choose single-view feet 
        feet_idxs = self.output_priority[-1]
        for v in self.valid_views:
            if all(self.keypoints_mv[v][feet_idxs,-1]>0.7) and all(self.age_2D[v][feet_idxs]==0):
                 
                feet_pos = np.mean(self.keypoints_mv[v][feet_idxs,:2], axis=0)
                feet_homo = self.cameras[v].homo_feet_inv @ np.array([feet_pos[0],feet_pos[1],1])
                feet_homo = feet_homo[:-1]/feet_homo[-1]

                self.output_cord = np.concatenate((feet_homo,[2]))
                return self.output_cord

        # if no single-view feet, then choose bbox bottom point
            
        bottom_points =[]
        for v in self.valid_views:
            bbox = self.bbox_mv[v]
            bp = self.cameras[v].homo_inv @ np.array([(bbox[2]+bbox[0])/2,bbox[3],1])
            bp = bp[:-1]/bp[-1]
            if bbox[3] > 1078:
                bottom_points.append(bp)
                continue
            self.output_cord = np.concatenate((bp ,[1]))
            
            return self.output_cord
            

        bottom_points = np.array(bottom_points).reshape(-1,2)
        self.output_cord = np.concatenate((np.mean(bottom_points,axis=0),[1]))

        return self.output_cord


    def single_view_init(self, detection_sample,id):
        # if initilized only with 1 view 
        self.state = TrackState.Unconfirmed
        self.confirm_time_left = 2
        cam_id = detection_sample.cam_id
        self.valid_views.append(cam_id)

        track2d = self.track2ds[cam_id]
        track2d.init_with_det(detection_sample)

        self.bbox_mv[cam_id] = detection_sample.bbox
        self.keypoints_mv[cam_id] = detection_sample.keypoints_2d
        self.age_2D[cam_id][detection_sample.keypoints_2d[:,-1]>self.keypoint_thrd] = 0
        self.age_bbox[cam_id] = 0
        
        self.bbox_kalman[cam_id].update(detection_sample.bbox[:4].copy())

        self.feat_bank[0] = track2d.reid_feat
        self.feat_count +=1


        self.id = id
        self.update_age = 0
        self.dura_bbox[cam_id] = 1

    def triangulation(self, detection_sample_list):
        keypoints_mv = np.zeros((self.num_cam, self.num_keypoints, 3))
        keypoints_3d = np.zeros((self.num_keypoints, 4))

        age_2D = np.ones((self.num_cam, self.num_keypoints)) * np.inf
        age_3D = np.ones((self.num_keypoints)) * np.inf

        for sample in detection_sample_list:
            keypoints_mv[sample.cam_id] = sample.keypoints_2d

        valid_joint_mask = (keypoints_mv[:,:,2] > self.keypoint_thrd)

        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:,j_idx]) < 2:
                joint_3d = np.zeros(4)
            else:
                A = np.zeros((2*self.num_keypoints,4))
                for v_idx in range(self.num_cam):
                    A[2*v_idx,:] = keypoints_mv[v_idx,j_idx,2] * (keypoints_mv[v_idx,j_idx,0] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[0,:])
                    A[2*v_idx+1,:] = keypoints_mv[v_idx,j_idx,2] * (keypoints_mv[v_idx,j_idx,1] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[1,:])

                u,sigma,vt = np.linalg.svd(A)
                joint_3d = vt[-1]/vt[-1][-1]
                age_3D[j_idx] = 0
            keypoints_3d[j_idx] = joint_3d
            age_2D[valid_joint_mask[:,j_idx]] = 0
            

        return keypoints_3d, keypoints_mv, age_3D, age_2D
    
        
    def multi_view_init(self, detection_sample_list,id):

        self.state = TrackState.Confirmed
        self.keypoints_3d, self.keypoints_mv, self.age_3D, self.age_2D = self.triangulation(detection_sample_list)

        for sample in detection_sample_list:
            cam_id = sample.cam_id
            self.valid_views.append(cam_id)

            track2d = self.track2ds[cam_id]
            track2d.init_with_det(sample)

            self.bbox_mv[cam_id] = sample.bbox
            
            self.bbox_kalman[cam_id].update(sample.bbox[:4].copy())

            if all(sample.keypoints_2d[self.upper_body,-1]>0.5) and sample.bbox[4]>0.9 and np.sum(self.iou_mv[cam_id]>0.15)<1 and np.sum(self.ovr_mv[cam_id]>0.3)<2:
                if self.feat_count:
                    bank = self.feat_bank[:self.feat_count]
                    sim = bank @ sample.reid_feat
                    if np.max(sim)<(self.thred_reid+0.1):
                        self.feat_bank[self.feat_count%self.bank_size] = sample.reid_feat
                        self.feat_count+=1
                else:
                    self.feat_bank[0] = track2d.reid_feat
                    self.feat_count +=1


            self.age_bbox[cam_id] = 0
            self.dura_bbox[cam_id] = 1

        self.update_age=0
        self.id = id
        self.iou_mv = [0 for i in range(self.num_cam)]

        self.valid_views = sorted(self.valid_views)
        self.get_output()
    
    def single_view_2D_update(self, v, sample,iou, ovr, ovr_tgt, avail_idx):
        
        if np.sum(iou>0.5)>=2 or np.sum(ovr_tgt>0.5)>=2:
            self.oc_state[v] = True
            oc_idx = avail_idx[np.where((iou>0.5) | (ovr_tgt>0.5))]
            self.oc_idx[v] = list(set(self.oc_idx[v] + [i for i in oc_idx if i!=self.id]))
            
        valid_joints = sample.keypoints_2d[:,-1]>self.keypoint_thrd
        self.keypoints_mv[v][valid_joints] = sample.keypoints_2d[valid_joints]
        self.age_2D[v][valid_joints] = 0

        self.bbox_mv[v] = sample.bbox
        self.age_bbox[v] = 0
        self.track2ds[v].init_with_det(sample)
            
        self.bbox_kalman[v].update(sample.bbox[:4].copy())
        
        self.dura_bbox[v] +=1
        self.iou_mv[v] = iou
        self.ovr_mv[v] = ovr
        self.ovr_tgt_mv[v] = ovr_tgt


    def multi_view_3D_update(self, avail_tracks):
        valid_views = [v for v in range(self.num_cam) if (self.age_bbox[v]==0)]
        if self.feat_count>=self.bank_size:
            bank = self.feat_bank
        else:
            bank = self.feat_bank[:self.feat_count%self.bank_size]

        for v in valid_views:
            if self.oc_state[v] and np.sum(self.iou_mv[v]>0.15)<2 and np.sum(self.ovr_tgt_mv[v]>0.3)<2 and self.bbox_mv[v][-1]>0.9:
                if self.feat_count==0:
                    self.oc_state[v] = False
                    self.oc_idx[v] = []
                    continue
                print("leaving oc ", self.id, v, self.iou_mv[v], self.ovr_tgt_mv[v],self.oc_idx[v])
                
                self_sim = np.max((self.track2ds[v].reid_feat @ bank.T))
                self.oc_state[v] = False
                oc_tracks = []
                print("self_sim",self_sim)
                if self_sim > 0.5:
                    self.oc_idx[v] = []
                    continue
                for t_id,track in enumerate(avail_tracks):
                    if track.id in self.oc_idx[v]:
                        oc_tracks.append(track)
                if len(oc_tracks)==0:
                    self.oc_idx[v] = []
                    print("miss oc track")
                    continue
                reid_sim = np.zeros(len(oc_tracks))
                for t_id, track in enumerate(oc_tracks):
                    if track.feat_count==0:
                        continue
                    if track.feat_count>=track.bank_size:
                        oc_bank = track.feat_bank
                    else:
                        oc_bank = track.feat_bank[:track.feat_count%track.bank_size]
                    reid_sim[t_id] = np.max(self.track2ds[v].reid_feat @ oc_bank.T)
                print("reid_sim", reid_sim)
                max_idx = np.argmax(reid_sim)
                self.oc_idx[v] = []
                if reid_sim[max_idx] > self_sim and reid_sim[max_idx]>0.5:
                    self.switch_view(oc_tracks[max_idx], v)
                    
        
        valid_joint_mask = (self.keypoints_mv[:,:,2] > self.keypoint_thrd) & (self.age_2D == 0)
        corr_v= []

        for j_idx in range(self.num_keypoints):
            if np.sum(valid_joint_mask[:,j_idx]) < 2:
                joint_3d = np.zeros(4)
                continue
            else:
                A = np.zeros((2*self.num_keypoints,4))
                for v_idx in range(self.num_cam):
                    if valid_joint_mask[v_idx, j_idx]:
                        A[2*v_idx,:] = self.keypoints_mv[v_idx,j_idx,2] * (self.keypoints_mv[v_idx,j_idx,0] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[0,:])
                        A[2*v_idx+1,:] = self.keypoints_mv[v_idx,j_idx,2] * (self.keypoints_mv[v_idx,j_idx,1] * self.cameras[v_idx].project_mat[2,:] - self.cameras[v_idx].project_mat[1,:])

                u,sigma,vt = np.linalg.svd(A)
                joint_3d = vt[-1]/vt[-1][-1]
                # false matching correction

                if (joint_3d[2]<-1 or joint_3d[2]>2.5) or (j_idx in self.feet_idx and (joint_3d[2]<-1 or joint_3d[2]>1)):
                    if np.min(self.dura_bbox[self.age_bbox==0])>=10:
                        continue
                    # views to be corrected are often new entering people with the minimum bbox tracking duration
                    v_cand = [v for v in range(self.num_cam) if (self.dura_bbox[v]==np.min(self.dura_bbox[self.age_bbox==0]))]
                        
                    print(self.dura_bbox[self.dura_bbox>0])
                    print(v_cand)
                    print(j_idx, joint_3d)
                    for v in  v_cand:
                        if valid_joint_mask[v,j_idx]:

                            
                            self.age_bbox[v] = np.inf
                            self.dura_bbox[v] = 0
                            

                            self.keypoints_mv[v]=0
                            self.age_2D[v]=np.inf
                            valid_joint_mask[v]=0
                            corr_v.append(v)                            
                            print("correction ","id ",self.id,"v ", v)

                            break
                    continue
                self.age_3D[j_idx] = np.min(self.age_2D[valid_joint_mask[:,j_idx],j_idx])
            self.keypoints_3d[j_idx] = joint_3d
            
        valid_views = [v for v in range(self.num_cam) if (self.age_bbox[v]==0 and (not v in corr_v))]
        self.update_age= 0

        for v in valid_views:
            if self.feat_count>=self.bank_size:
                bank = self.feat_bank
            else:
                bank = self.feat_bank[:self.feat_count]
            
            sample = self.track2ds[v]
            if all(sample.keypoints_2d[self.upper_body,-1]>0.5) and sample.bbox[4]>0.9 and np.sum(self.iou_mv[v]>0.15)<2 and np.sum(self.ovr_mv[v]>0.3)<2:
                if self.feat_count==0:
                    self.feat_bank[0] = sample.reid_feat
                    self.feat_count+=1
                else:
                    sim = bank @ sample.reid_feat
                    if np.max(sim)<(self.thred_reid+0.1):
                        self.feat_bank[self.feat_count%self.bank_size] = sample.reid_feat
                        self.feat_count+=1


        if self.state == TrackState.Unconfirmed:
            if any(self.bbox_mv[self.age_bbox==0][:,-1]>0.9):
                self.confirm_time_left-=1
                if self.confirm_time_left<=0:
                    self.state = TrackState.Confirmed
        
        self.iou_mv = [0 for i in range(self.num_cam)]
        self.ovr_mv = [0 for i in range(self.num_cam)]
        return corr_v

    def CalcTargetRays(self, v):
        if self.age_bbox[v]>1:
            return self.unit
        cam = self.cameras[v]

        return aic_cpp.compute_joints_rays(self.keypoints_mv[v], cam.project_inv, cam.pos)

def calcRays_sv(keypoints_2d, cam):
    joints_h = np.vstack((keypoints_2d[:,:-1].T, np.ones((1,keypoints_2d.shape[0])))) #3*n
    joints_rays =  cam.project_inv @ joints_h
    joints_rays /= joints_rays[-1]
    joints_rays = joints_rays[:-1]
    joints_rays -= np.repeat(cam.pos.reshape(3,1),keypoints_2d.shape[0], axis=1)
    joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0)+1e-5)
    joints_rays_norm = joints_rays_norm.T

    return joints_rays_norm

            

class PoseTracker():
    def __init__(self,cameras):
        self.cameras = cameras
        self.num_cam = len(cameras)
        self.tracks = []
        self.reid_thrd = 0.5
        self.num_keypoints = 17
        self.decay_weight = 0.5
        self.thred_p2l_3d = 0.3
        self.thred_2d = 0.3
        self.thred_epi = 0.2
        self.thred_homo = 1.5
        self.thred_bbox = 0.4
        self.keypoint_thrd = 0.7
        self.glpk_bip = GLPKSolver(min_affinity=-1e5)
        self.main_joints = np.array([5,6,11,12,13,14,15,16])
        self.bank_size = 30
        self.thred_reid = 0.5
        self.upper_body = np.array([5,6,11,12])

    def compute_reid_aff(self, detection_sample_list_mv, avail_tracks):
        reid_sim_mv = []
        reid_weight = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            reid_sim_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))
            reid_weight_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))+1e-5
    
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                if sample.bbox[-1]<0.88:
                    continue
                for t_id, track in enumerate(avail_tracks):
                    if not len(track.track2ds[v].state):
                        continue
                    if (Track2DState.Occluded not in track.track2ds[v].state) and (Track2DState.Missing not in track.track2ds[v].state):
                        continue 
                    reid_sim = track.feat_bank @ sample.reid_feat
                    reid_sim = reid_sim[reid_sim>0]
                    if reid_sim.size: ##
                        reid_sim_sv[s_id,t_id] = np.max(reid_sim)
                        reid_weight_sv[s_id,t_id] = 1

                        reid_sim_sv[s_id,t_id] -= self.reid_thrd

            reid_sim_mv.append(reid_sim_sv)
            reid_weight.append(reid_weight_sv)
        return reid_sim_mv,reid_weight
    def compute_3dkp_aff(self,detection_sample_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))
            cam = self.cameras[v]
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_h = np.vstack((sample.keypoints_2d[:,:-1].T, np.ones((1,self.num_keypoints))))
                joints_rays =  cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3,1),self.num_keypoints, axis=1)
                joints_rays = joints_rays / (np.linalg.norm(joints_rays, axis=0)+1e-5)
                joints_rays = joints_rays.T # 17*3
                for t_id, track in enumerate(avail_tracks):
                    aff = np.zeros(self.num_keypoints)
                    kp_3d = track.keypoints_3d
                    k_idx = np.where(sample.keypoints_2d[:,-1] < self.keypoint_thrd)[0]
                    
                    aff[k_idx] = Point2LineDist(kp_3d[k_idx, :-1], cam.pos, joints_rays[k_idx])

                    valid = (sample.keypoints_2d[:,-1] > self.keypoint_thrd) * (kp_3d[:,-1]>0)
                    aff  = 1- aff/self.thred_p2l_3d
                    aff = aff * sample.keypoints_2d[:,-1]* np.exp(-track.age_3D)

                    aff_sv[s_id,t_id] = np.sum(aff)/ (np.sum(valid* np.exp(-track.age_3D))+1e-5)
            aff_mv.append(aff_sv)

        return aff_mv
    
    def compute_2dkp_aff(self,detection_sample_list_mv, avail_tracks):
        aff_mv = []
        n_track = len(avail_tracks)

        for v in range(self.num_cam):
            aff_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))
            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_s = sample.keypoints_2d
                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv[v]

                    dist = np.linalg.norm(joints_t[:,:-1] - joints_s[:,:-1], axis=1)
                    aff = 1 - dist/ (self.thred_2d * (np.linalg.norm(track.bbox_mv[v][2:4] - track.bbox_mv[v][:2])+1e-5))
                    valid = (joints_t[:,-1]> self.keypoint_thrd) * (joints_s[:,-1]> self.keypoint_thrd)
                    aff = aff * valid * np.exp(-track.age_2D[v])
                    
                    aff_sv[s_id,t_id] = np.sum(aff)/(np.sum(valid*np.exp(-track.age_2D[v]))+1e-5)

            aff_mv.append(aff_sv)

        return aff_mv    
    
    def compute_epi_homo_aff(self,detection_sample_list_mv, avail_tracks):
        aff_mv = []
        aff_homo = [] 
        n_track = len(avail_tracks)
        mv_rays = self.CalcJointRays(detection_sample_list_mv)
        age_2D_thr = 1

        feet_idxs = [15,16]

        for v in range(self.num_cam):
            pos = self.cameras[v].pos
            aff_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))
            cam = self.cameras[v]
            sv_rays = mv_rays[v]
            aff_homo_sv = np.zeros((len(detection_sample_list_mv[v]),n_track))

            for s_id, sample in enumerate(detection_sample_list_mv[v]):
                joints_s = sample.keypoints_2d

                feet_valid_s = np.all(joints_s[feet_idxs,-1] > self.keypoint_thrd)

                feet_s = aic_cpp.compute_feet_s(joints_s, feet_idxs, cam.homo_feet_inv)
                box_pos_s = aic_cpp.compute_box_pos_s(sample.bbox,cam.homo_inv)
                box_valid_s = True


                for t_id, track in enumerate(avail_tracks):
                    joints_t = track.keypoints_mv
                    aff_sv[s_id,t_id], aff_homo_sv[s_id,t_id] = aic_cpp.loop_t_homo_full(
                        joints_t,
                        joints_s,
                        track.age_bbox,
                        track.age_2D,
                        feet_s,
                        feet_valid_s,
                        v,
                        self.thred_epi,
                        self.thred_homo,
                        self.keypoint_thrd,
                        age_2D_thr,
                        sv_rays[s_id],
                        self.cameras,
                        box_pos_s,
                        box_valid_s,
                        track.bbox_mv)
                    continue
                    aff_ss = []
                    aff_homo_ss = []
                    if feet_valid_s:
                        feet_valid_t = (joints_t[:,feet_idxs[0],-1] > self.keypoint_thrd) & (joints_t[:,feet_idxs[1],-1] > self.keypoint_thrd)

                    valid = (joints_t[:, :,-1]> self.keypoint_thrd) & (joints_s[:,-1]> self.keypoint_thrd)
                    for vj in range(self.num_cam):

                        if v == vj or track.age_bbox[vj]>=2:
                            continue

                        pos_j = self.cameras[vj].pos
                        track_rays_sv = track.CalcTargetRays(vj)

                        aff_temp = aic_cpp.epipolar_3d_score_norm(pos,sv_rays[s_id], pos_j, track_rays_sv, self.thred_epi)
                        _aff_ss = aic_cpp.aff_sum(aff_temp, valid[vj], track.age_2D[vj], 1)
                        if _aff_ss != 0:
                            aff_ss.append(_aff_ss)

                        if feet_valid_s and feet_valid_t[vj]:
                            _aff_homo_ss = aic_cpp.compute_feet_distance(
                                joints_t[vj], feet_idxs, self.cameras[vj].homo_feet_inv, feet_s, self.thred_homo)
                            aff_homo_ss.append(_aff_homo_ss)


                    aff_homo_sv[s_id,t_id] = sum(aff_homo_ss) / (len(aff_homo_ss) + 1e-5)
                    

                    aff_sv[s_id,t_id] = sum(aff_ss) / (len(aff_ss)+ 1e-5)

            aff_homo.append(aff_homo_sv)
            aff_mv.append(aff_sv)
        return aff_mv, aff_homo
                            

    def compute_bboxiou_aff(self,detection_sample_list_mv,avail_tracks):
        aff_mv = []
        iou_mv = []
        ovr_det_mv = []
        ovr_tgt_mv = []
        n_track = len(avail_tracks)
        for v in range(self.num_cam):
            iou = np.zeros((len(detection_sample_list_mv[v]),n_track))
            ovr_det = np.zeros((len(detection_sample_list_mv[v]),len(detection_sample_list_mv[v])))

            if iou.size==0:
                aff_mv.append(iou)
                iou_mv.append(iou)
                ovr_det_mv.append(ovr_det)
                ovr_tgt_mv.append(iou)
                continue
            
            detection_bboxes=np.stack([detection.bbox for detection in detection_sample_list_mv[v]])[:,:5]
            multi_mean=np.stack([track.bbox_kalman[v].mean.copy() if track.bbox_kalman[v].mean is not None else np.array([1,1,1,1,0,0,0,0]) for track in avail_tracks])
            multi_covariance=np.stack([track.bbox_kalman[v].covariance.copy() if track.bbox_kalman[v].covariance is not None else np.eye(8) for track in avail_tracks])
            multi_mean,multi_covariance=avail_tracks[0].bbox_kalman[v].multi_predict(multi_mean,multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                if avail_tracks[i].bbox_kalman[v].mean is not None:
                    avail_tracks[i].bbox_kalman[v].mean=mean
                    avail_tracks[i].bbox_kalman[v].covariance=cov

            score = detection_bboxes[:,-1]
            detection_bboxes = detection_bboxes[:,:4]
            track_bboxes=self.xyah2ltrb(multi_mean[:,:4].copy())
            for i in range(len(track_bboxes)):
                if avail_tracks[i].bbox_kalman[v].mean is None:
                    track_bboxes[i]=avail_tracks[i].bbox_mv[v][:4]
            iou= ious(detection_bboxes.copy(),track_bboxes.copy())
            iou[np.isnan(iou)]=0
            age=np.array([track.age_bbox[v] for track in avail_tracks])
            ovr = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(),track_bboxes.copy())
            ovr_tgt_mv.append(ovr * (age<=15))
            ovr_det = aic_cpp.bbox_overlap_rate(detection_bboxes.copy(),detection_bboxes.copy())
            ovr_det_mv.append(ovr_det)
            iou_mv.append(iou * (age<=15)) 

            
            iou = (((iou-0.5) * (age<=15)).T * score).T
            aff_mv.append(iou)
        return aff_mv, iou_mv, ovr_det_mv, ovr_tgt_mv

    def CalcJointRays(self,detection_sample_list_mv):
        mv_rays = []
        for v in range(self.num_cam):
            cam  = self.cameras[v]
            sv_rays = []
            n_detect= len(detection_sample_list_mv[v])
            sample_sv = detection_sample_list_mv[v]
            for s_id, sample in enumerate(sample_sv):
                joints_h = np.vstack((sample.keypoints_2d[:,:-1].T, np.ones((1,self.num_keypoints)))) #3*n
                joints_rays =  cam.project_inv @ joints_h
                joints_rays /= joints_rays[-1]
                joints_rays = joints_rays[:-1]
                joints_rays -= np.repeat(cam.pos.reshape(3,1),self.num_keypoints, axis=1)

                joints_rays_norm = joints_rays / (np.linalg.norm(joints_rays, axis=0)+1e-5)
                joints_rays_norm = joints_rays_norm.T
                sv_rays.append(joints_rays_norm) #17*3
            mv_rays.append(sv_rays)
        
        return mv_rays
    
    def match_with_miss_tracks(self, new_track, miss_tracks):
        if len(miss_tracks)==0:
            self.tracks.append(new_track)
            print("new init",new_track.id, new_track.valid_views)
            return
        
        reid_sim = np.zeros(len(miss_tracks))
        for t_id, track in enumerate(miss_tracks):
            if track.feat_count == 0 or new_track.feat_count == 0:
                continue
            if track.feat_count>=self.bank_size:
                bank = track.feat_bank
            else:
                bank = track.feat_bank[:track.feat_count%self.bank_size]
            new_bank = new_track.feat_bank[:new_track.feat_count%self.bank_size]
            
            reid_sim[t_id] = np.max(new_bank @ bank.T)
        
        t_id = np.argmax(reid_sim)

        print("init reid score: ", reid_sim)
        if reid_sim[t_id]>0.5:

            miss_tracks[t_id].reactivate(new_track)
            print("reactivate", miss_tracks[t_id].id, miss_tracks[t_id].valid_views)

        else:
            self.tracks.append(new_track)
            print("new init",new_track.id, new_track.valid_views)



                
    def target_init(self,detection_sample_list_mv, miss_tracks, iou_det_mv, ovr_det_mv, ovr_tgt_mv):
        cam_idx_map = [] # cam_idx_map for per det
        det_count = [] # per view det count
        det_all_count=[0]
        for v in range(self.num_cam):
            det_count.append(len(detection_sample_list_mv[v]))
            det_all_count.append(det_all_count[-1]+det_count[-1])
            cam_idx_map+=[v]*det_count[-1]
        
        if det_all_count[-1]==0:
            return self.tracks

        det_num = det_all_count[-1]

        aff_homo = np.ones((det_num,det_num)) * (-10000)
        aff_epi = np.ones((det_num,det_num)) * (-10000)

        mv_rays = self.CalcJointRays(detection_sample_list_mv)
        feet_idxs = [15,16]

        for vi in range(self.num_cam):
            samples_vi = detection_sample_list_mv[vi]
            pos_i = self.cameras[vi].pos
            for vj in range(vi, self.num_cam):
                if vi==vj:
                    continue
                else:
                    pos_j = self.cameras[vj].pos
                    samples_vj = detection_sample_list_mv[vj]

                    aff_temp = np.zeros((det_count[vi],det_count[vj]))
                    reid_sim_temp = np.zeros((det_count[vi],det_count[vj]))
                    aff_homo_temp = np.zeros((det_count[vi],det_count[vj]))
                    # calculate for each det pair
                    for a in range(det_count[vi]):
                        sample_a = samples_vi[a]
                        feet_valid_a = np.all(sample_a.keypoints_2d[feet_idxs,-1]>self.keypoint_thrd)
                        if feet_valid_a:
                            feet_a = np.mean(sample_a.keypoints_2d[feet_idxs,:-1],axis=0)
                            feet_a = self.cameras[vi].homo_feet_inv @ np.array([feet_a[0],feet_a[1],1])
                            feet_a = feet_a[:-1]/feet_a[-1]
                        else:
                            feet_a = np.array([(sample_a.bbox[0]+sample_a.bbox[0])/2, sample_a.bbox[3]])
                            feet_a = self.cameras[vi].homo_inv @ np.array([feet_a[0],feet_a[1],1])
                            feet_a = feet_a[:-1]/feet_a[-1]
                        
                        feet_valid_a = True
                        
                        for b in range(det_count[vj]):
                            sample_b = samples_vj[b]
                            aff = np.zeros(self.num_keypoints)
                            valid_kp = (sample_a.keypoints_2d[:,-1]>self.keypoint_thrd) & (sample_b.keypoints_2d[:,-1]>self.keypoint_thrd)
                            j_id = np.where(valid_kp)[0]
                            aff[j_id] = epipolar_3d_score_norm(pos_i, mv_rays[vi][a][j_id,:],pos_j, mv_rays[vj][b][j_id,:], self.thred_epi)

                            if feet_valid_a and np.all(sample_b.keypoints_2d[feet_idxs,-1]>self.keypoint_thrd):

                                feet_b = np.mean(sample_b.keypoints_2d[feet_idxs,:-1],axis=0)
                                feet_b = self.cameras[vj].homo_feet_inv @ np.array([feet_b[0],feet_b[1],1])
                                feet_b = feet_b[:-1]/feet_b[-1]

                                aff_homo_temp[a,b] = 1 - np.linalg.norm(feet_b - feet_a)/ self.thred_homo
                            else:
                                feet_b = np.array([(sample_b.bbox[0]+sample_b.bbox[0])/2, sample_b.bbox[3]])
                                feet_b = self.cameras[vj].homo_feet_inv @ np.array([feet_b[0],feet_b[1],1])
                                feet_b = feet_b[:-1]/feet_b[-1]

                                aff_homo_temp[a,b] = 1 - np.linalg.norm(feet_b - feet_a)/ self.thred_homo

                            aff_temp[a,b] = np.sum(aff* sample_a.keypoints_2d[:,-1] * sample_b.keypoints_2d[:,-1])/ (np.sum(valid_kp * sample_a.keypoints_2d[:,-1] * sample_b.keypoints_2d[:,-1])+1e-5)
                            reid_sim_temp[a,b] = (sample_a.reid_feat @ sample_b.reid_feat )
                    
                    aff_epi[det_all_count[vi]:det_all_count[vi+1],det_all_count[vj]:det_all_count[vj+1]] = aff_temp
                    aff_homo[det_all_count[vi]:det_all_count[vi+1],det_all_count[vj]:det_all_count[vj+1]] = aff_homo_temp


        aff_final = 2*aff_epi + aff_homo
        aff_final[aff_final<-1000] = -np.inf
        
        clusters, sol_matrix = self.glpk_bip.solve(aff_final,True)


        for cluster in clusters:
            if len(cluster)==1:
               
               view_list, number_list = find_view_for_cluster(cluster,det_all_count)
               det = detection_sample_list_mv[view_list[0]][number_list[0]]
               
               if det.bbox[-1]>0.9 and all(det.keypoints_2d[self.main_joints,-1]>0.5) and np.sum(iou_det_mv[view_list[0]][number_list[0]]>0.15)<1 and np.sum(ovr_det_mv[view_list[0]][number_list[0]]>0.3)<2:

                   new_track = PoseTrack(self.cameras)
                   new_track.single_view_init(det,id=len(self.tracks)+1)

                   self.match_with_miss_tracks(new_track, miss_tracks)

            else:
                view_list, number_list = find_view_for_cluster(cluster,det_all_count)
                sample_list = [detection_sample_list_mv[view_list[idx]][number_list[idx]] for idx in range(len(view_list))]
                for i, sample in enumerate(sample_list):
                    if all(sample.keypoints_2d[self.main_joints,-1]>0.5) and sample.bbox[-1]>0.9 and np.sum(iou_det_mv[view_list[i]][number_list[i]]>0.15)<1 and np.sum(ovr_det_mv[view_list[i]][number_list[i]]>0.3)<2:
                        new_track = PoseTrack(self.cameras)
                        for j in range(len(view_list)):
                            new_track.iou_mv[view_list[j]] = iou_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_mv[view_list[j]] = ovr_det_mv[view_list[j]][number_list[j]]
                            new_track.ovr_tgt_mv[view_list[j]] = ovr_tgt_mv[view_list[j]][number_list[j]]

                        new_track.multi_view_init(sample_list, id=len(self.tracks)+1)
                        self.match_with_miss_tracks(new_track, miss_tracks)
                        
                        break
    
    def xyah2ltrb(self,ret):
        ret[...,2] *= ret[...,3]
        ret[...,:2] -= ret[...,2:] / 2
        ret[...,2:] +=ret[...,:2]
        return ret

    def mv_update_wo_pred(self, detection_sample_list_mv, frame_id = None):

        um_iou_det_mv = []
        um_ovr_det_mv = []
        um_ovr_tgt_mv = []

        a_epi= 1
        a_box = 5
        a_homo = 1
        a_reid=5
        
        # vide the valid view list
        for track in self.tracks :
            track.valid_views=[]

        # 1st step, matching with confirmed and unconfirmed tracks
        avail_tracks = [track for track in self.tracks if track.state < TrackState.Missing]
        avail_idx = np.array([track.id for track in avail_tracks])

        aff_reid, reid_weight = self.compute_reid_aff(detection_sample_list_mv, avail_tracks)
        aff_epi, aff_homo = self.compute_epi_homo_aff(detection_sample_list_mv, avail_tracks)
        aff_box, iou_mv, ovr_det_mv, ovr_tgt_mv = self.compute_bboxiou_aff(detection_sample_list_mv , avail_tracks)

        updated_tracks = set()

        unmatched_det = []
        match_result = []


        for v in range(self.num_cam):

            iou_sv = iou_mv[v]
            ovr_det_sv = ovr_det_mv[v]
            ovr_tgt_sv = ovr_tgt_mv[v]

            matched_det_sv = set()

            

            aff_epi[v][aff_epi[v] < (-a_box+0.5*a_box)] = -a_box+0.5*a_box
            aff_homo[v][aff_homo[v] < (-a_box+0.5*a_box)] = -a_box+0.5*a_box

            norm = a_epi *(aff_epi[v]!=0).astype(float) + a_box *(aff_box[v]!=0).astype(float)+ a_homo *(aff_homo[v]!=0).astype(float)
            aff_final = (a_epi*aff_epi[v] + a_box*aff_box[v] + a_homo*aff_homo[v]+aff_reid[v]*a_reid*reid_weight[v])/(1+reid_weight[v])

            idx = np.where(norm>0)
            aff_final[idx] -= (a_box-norm[idx])*0.1
            sample_list_sv = detection_sample_list_mv[v]

            aff_final[aff_final<0] =0

            row_idxs, col_idxs = linear_sum_assignment(-aff_final)
            
            match_result.append((row_idxs,col_idxs))
            if iou_sv.size:
                
                colmax=iou_sv.max(0)
                argcolmax=iou_sv.argmax(0)

                occlusion_row=set()
                for i in range(iou_sv.shape[1]):
                    if i not in col_idxs:
                        if colmax[i]>0.5:
                            state=Track2DState.Occluded
                            occlusion_row.add(argcolmax[i])

                        else:
                            state=Track2DState.Missing
                        if len(avail_tracks[i].track2ds[v].state)==10:
                            avail_tracks[i].track2ds[v].state.pop()
                        avail_tracks[i].track2ds[v].state=[state]+avail_tracks[i].track2ds[v].state
            elif len(iou_sv)==0:
                for i in range(iou_sv.shape[1]):
                    if len(avail_tracks[i].track2ds[v].state)==10:
                        avail_tracks[i].track2ds[v].state.pop()
                    avail_tracks[i].track2ds[v].state=[Track2DState.Missing]+avail_tracks[i].track2ds[v].state

            for row, col in zip(row_idxs, col_idxs):
                # only update 2D info
                if row in occlusion_row:
                    state=Track2DState.Occluded
                    if len(avail_tracks[col].track2ds[v].state)==10:
                        avail_tracks[col].track2ds[v].state.pop()
                    avail_tracks[col].track2ds[v].state=[state]+avail_tracks[col].track2ds[v].state
                if aff_final[row,col]<=0:
                    continue
                iou = iou_sv[row]
                ovr = ovr_det_sv[row]
                ovr_tgt = ovr_tgt_sv[row]
                if True:
                    avail_tracks[col].single_view_2D_update(v,sample_list_sv[row],iou, ovr, ovr_tgt, avail_idx)
                    updated_tracks.add(col)
                    matched_det_sv.add(row)
                    
            unmatched_det_sv = list(set(range(len(sample_list_sv))) - matched_det_sv)
            unmatched_sv = [sample_list_sv[u] for u in unmatched_det_sv]
            unmatched_det.append(unmatched_sv)

            unmatched_iou_sv = iou_sv[unmatched_det_sv]
            um_iou_det_mv.append(unmatched_iou_sv)

            unmatched_ovr_det_sv = ovr_det_sv[unmatched_det_sv]
            um_ovr_det_mv.append(unmatched_ovr_det_sv)

            unmatched_ovr_tgt_sv = ovr_tgt_sv[unmatched_det_sv]
            um_ovr_tgt_mv.append(unmatched_ovr_tgt_sv)
    
        for t_id in updated_tracks:
            corr_v = avail_tracks[t_id].multi_view_3D_update(avail_tracks)

        for t_id, track in enumerate(avail_tracks):
            track.valid_views = [v for v in range(self.num_cam) if track.age_bbox[v]==0]
            if track.state == TrackState.Unconfirmed and (not t_id in updated_tracks):
                track.state = TrackState.Deleted
            if track.update_age >=15:
                track.state = TrackState.Missing
            if track.state == TrackState.Confirmed:
                track.get_output()

        # perform association for unmatched detections and matching with missing tracks
        miss_tracks = [track for track in self.tracks if track.state == TrackState.Missing]
        if len(unmatched_det):
            self.target_init(unmatched_det, miss_tracks,um_iou_det_mv, um_ovr_det_mv, um_ovr_tgt_mv)

        feat_cnts= []
        for track in self.tracks:
            for i in range(self.num_cam):
                if track.age_bbox[i]>=15:
                    track.bbox_kalman[i]=KalmanFilter_box()

            track.age_2D[track.age_2D>=3] = np.inf
            track.age_3D[track.age_3D>=3] = np.inf
            track.age_bbox[track.age_bbox>=15] = np.inf
            track.dura_bbox[track.age_bbox>=15] = 0

            track.age_2D +=1
            track.age_3D +=1
            track.age_bbox +=1
            track.update_age +=1
            if track.state  == TrackState.Confirmed:
                feat_cnts.append((track.id, track.feat_count))


    def output(self,frame_id):
        frame_results=[]
        for track in self.tracks:
            if track.state == TrackState.Confirmed:
                if track.update_age==1:
                    for v in track.valid_views:
                        bbox = track.bbox_mv[v]

                        record = np.array([[self.cameras[v].idx_int,track.id, frame_id, bbox[0], bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1], track.output_cord[0], track.output_cord[1],track.output_cord[2]]])
                        frame_results.append(record)

        return frame_results
    
            

        
        
        
        
        


        
        
        

        

        

                



        






        









        






