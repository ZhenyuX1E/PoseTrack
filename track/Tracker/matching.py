import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
from torchvision.ops import nms
from cython_bbox import bbox_overlaps as bbox_ious
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1])), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b, x, y
def myiou(atracks,btracks):
    ious = np.zeros((len(atracks), len(btracks)), dtype=np.float)
    if ious.size == 0:
        return ious
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        altrbs = atracks
        bltrbs = btracks
    else:
        altrbs = [track.ltrb for track in atracks]
        bltrbs = [track.ltrb for track in btracks]
    bbox_p=np.ascontiguousarray(altrbs, dtype=np.float)
    bbox_g=np.ascontiguousarray(bltrbs, dtype=np.float)
    
    pl,pt,pr,pb=bbox_p[:,0:1],bbox_p[:,1:2],bbox_p[:,2:3],bbox_p[:,3:4]
    gl,gt,gr,gb=bbox_g[:,0][None,:],bbox_g[:,1][None,:],bbox_g[:,2][None,:],bbox_g[:,3][None,:]
    outl,outt,outr,outb=np.minimum(pl,gl),np.minimum(pt,gt),np.maximum(pr,gr),np.maximum(pb,gb)
    #il,it,ir,ib=np.maximum(pl,gl),np.maximum(pt,gt),np.minimum(pr,gr),np.minimum(pb,gb)
    #inter=(ir-il).clip(0)*(ib-it).clip(0)
    outer=(outr-outl)*(outb-outt)
    union=(pr-pl)*(pb-pt)+(gr-gl)*(gb-gt)
    return 1-union/outer/2
def Giou_np(bbox_p, bbox_g):
    bbox_p=np.ascontiguousarray(bbox_p, dtype=np.float)
    bbox_g=np.ascontiguousarray(bbox_g, dtype=np.float)
    x1p = np.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = np.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = np.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = np.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    bbox_p = np.concatenate([x1p, y1p, x2p, y2p], axis=1)
    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])
    x1I = np.maximum(bbox_p[:, 0], bbox_g[:, 0])
    y1I = np.maximum(bbox_p[:, 1], bbox_g[:, 1])
    x2I = np.minimum(bbox_p[:, 2], bbox_g[:, 2])
    y2I = np.minimum(bbox_p[:, 3], bbox_g[:, 3])
    I = np.maximum((y2I - y1I), 0) * np.maximum((x2I - x1I), 0)
    x1C = np.minimum(bbox_p[:, 0], bbox_g[:, 0])
    y1C = np.minimum(bbox_p[:, 1], bbox_g[:, 1])
    x2C = np.maximum(bbox_p[:, 2], bbox_g[:, 2])
    y2C = np.maximum(bbox_p[:, 3], bbox_g[:, 3])
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    iou = 1.0 * I / U
    giou = iou - (area_c - U) / area_c
    return giou
def ious(altrbs, bltrbs):
    """
    Compute cost based on IoU
    :type altrbs: list[ltrb] | np.ndarray
    :type altrbs: list[ltrb] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(altrbs), len(bltrbs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(altrbs, dtype=np.float),
        np.ascontiguousarray(bltrbs, dtype=np.float)
    )

    return ious


def touched(bboxes):
    eps=1e-10
    n=len(bboxes)
    l,t,r,b=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
    h,w=b-t,r-l
    xc=(l+r)/2
    mh=(h[None,:]+h[:,None])/2
    xdis=np.abs(xc[None,:]-xc[:,None])/mh
    ydis=(np.abs(t[None,:]-t[:,None])*3)/mh
    dis=(xdis**2+ydis**2)**0.5
    return (dis<1).astype(np.int32)

def bboxes_ciou(tracks1,tracks2):
    if (len(tracks1)>0 and isinstance(tracks1[0], np.ndarray)):
        ltrbs1=tracks1
        ltrbs2=tracks2
    else:
        ltrbs1 = [track.ltrb for track in tracks1]
        ltrbs2 = [track.ltrb for track in tracks2]
    boxes1=np.ascontiguousarray(ltrbs1, dtype=np.float)
    boxes2=np.ascontiguousarray(ltrbs2, dtype=np.float)
    ious = np.zeros((len(boxes1), len(boxes2)), dtype=np.float)
    if ious.size == 0:
        return ious

    #cal the box's area of boxes1 and boxess
    boxes1Area = ((boxes1[:,2]-boxes1[:,0])*(boxes1[:,3]-boxes1[:,1]))[:,None]
    boxes2Area = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))[None,:]

    # cal Intersection
    left_up = np.maximum(boxes1[:,:2,None],boxes2.transpose(1,0)[None,:2])
    right_down = np.minimum(boxes1[:,2:,None],boxes2.transpose(1,0)[None,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[:,0] * inter_section[:,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = inter_area/(union_area+np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxes1[:,:2,None],boxes2.transpose(1,0)[None,:2])
    outer_right_down = np.maximum(boxes1[:,2:,None],boxes2.transpose(1,0)[None,2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[:,0]) + np.square(outer[:,1])

    # cal center distance
    boxes1_center = ((boxes1[:, :2] +  boxes1[:,2:]) * 0.5)[:,:,None]
    boxes2_center = (((boxes2[:, :2] +  boxes2[:,2:]) * 0.5).transpose(1,0))[None,:,:]
    center_dis = np.square(boxes1_center[:,0]-boxes2_center[:,0]) +\
                 np.square(boxes1_center[:,1]-boxes2_center[:,1])

    # cal penalty term
    # cal width,height
    boxes1_size = (np.maximum(boxes1[:,2:]-boxes1[:,:2],0.0))[:,:,None]
    boxes2_size = (np.maximum(boxes2[:, 2:] - boxes2[:, :2], 0.0).transpose(1,0))[None,:,:]
    #v1=1-np.maximum(boxes1_size[:,0],boxes2_size[:,0])
    v2=1-np.minimum(boxes1_size[:,1],boxes2_size[:,1])/np.maximum(boxes1_size[:,1],boxes2_size[:,1])
    #cal ciou
    cious = 1-(ious - (center_dis**0.5 / outer_diagonal_line**0.5 +v2))

    return cious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        altrbs = atracks
        bltrbs = btracks
    else:
        altrbs = [track.ltrb for track in atracks]
        bltrbs = [track.ltrb for track in btracks]
    _ious = ious(altrbs, bltrbs)
    cost_matrix = 1 - _ious

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        altrbs = atracks
        bltrbs = btracks
    else:
        altrbs = [track.ltwh_to_ltrb(track.pred_bbox) for track in atracks]
        bltrbs = [track.ltwh_to_ltrb(track.pred_bbox) for track in btracks]
    _ious = ious(altrbs, bltrbs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost