import numpy as np
import tensorflow as tf
import utils
import config
import batch_generate

def roi_proposal(rpn_cls_prob_reshape, rpn_bbox_pred, H, W, ANCHOR_PER_GRID, ANCHOR_BOX, TOP_N_DETECTION, NMS_THRESH, IM_H, IM_W):
    """
    clip the predict results fron rpn output
    appply nms
    proposal topN results as final layer output, no backward operation need here
    """
    box_probs = np.reshape(rpn_cls_prob_reshape,[-1,2])[:,1]
    box_delta = np.reshape(rpn_bbox_pred,[H * W * ANCHOR_PER_GRID,4])
    anchor_box = ANCHOR_BOX
    pred_box_xyxy = utils.bbox_delta_convert_inv(anchor_box, box_delta)
    box_nms, probs_nms, pick = utils.non_max_suppression_fast(pred_box_xyxy, box_probs, TOP_N_DETECTION, overlap_thresh=NMS_THRESH)
    #box_nms = box_nms[probs_nms>0.90]
    box = box_nms
    #box = batch_generate.bbox2cxcy(box)
    #clip box
    proposal_region = clip_box(mc, box, IM_H, IM_W)
    #print('the shape of proposaled region is ',proposal_region.shape)
    #print('the proposaled region value is ', proposal_region)
    batch_inds = np.zeros((proposal_region.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposal_region.astype(np.float32, copy=False)))
    return blob, probs_nms

def clip_box(mc,box, IM_H, IM_W):
    h = IM_H
    w = IM_W
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], w - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], h - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], w - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], h - 1), 0)
    return boxes

def proposaled_target_layer(mc, rois, rpn_scores, gt_boxes):
    all_rois = rois
    all_scores = rpn_scores
    _num_classes = mc.CLASSES
    zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
    all_scores = np.vstack((all_scores, zeros))
    num_images = 1
    rois_per_image = mc.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(mc.RPN_FRACTION * rois_per_image)
    sampling_methods(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes)

def sampling_methods(mc, all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, _num_classes):
    negative_threshould = mc.neg_max_overlaps
    postive_threshould = mc.pos_min_overlaps
    groundtruth = batch_generate.coord2box(gt_boxes[:,:4])
    proposed_rois = batch_generate.coord2box(all_rois[:,:4])
    num_of_groundtruth = len(groundtruth)
    num_of_rois = len(proposed_rois)
    overlaps_table = np.zeros((num_of_rois, num_of_groundtruth))
    for i in range(num_of_rois):
        for j in range(num_of_groundtruth):
            overlaps_table[i,j] = utils.box_iou(proposed_rois[i], groundtruth[j])
    gt_assignment = overlaps_table.argmax(axis=1)
    max_overlaps = overlaps_table.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    fg_inds = np.where(max_overlaps >= postive_threshould)[0]
    bg_inds = np.where(max_overlaps < negative_threshould)[0]
    
def target_delta():
    pass
