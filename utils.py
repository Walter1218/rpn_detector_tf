import numpy as np
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def bgr_to_rgb(ims):
    """Convert a list of images from BGR format to RGB format."""
    out = []
    for im in ims:
        out.append(im[:,:,::-1])
    return out

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))

        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)

            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)

                    overlaps[n, k] = iw * ih / ua

    return overlaps

def bbox_delta_convert(anchor_box, gt_box):
    """
    compute delta value
    """
    ref_cx = anchor_box[:,0]
    ref_cy = anchor_box[:,1]
    ref_w = anchor_box[:,2]
    ref_h = anchor_box[:,3]

    gt_cx = gt_box[:,0]
    gt_cy = gt_box[:,1]
    gt_w = gt_box[:,2]
    gt_h = gt_box[:,3]

    dx = (gt_cx - ref_cx) / ref_w
    dy = (gt_cy - ref_cy) / ref_h
    dw = np.log(gt_w / ref_w)
    dh = np.log(gt_h / ref_h)

    target_delta = np.stack((dx, dy, dw, dh))
    target_delta = np.transpose(target_delta)
    return target_delta

def bbox_delta_convert_inv(anchor_box, trans_boxes):
    dx = trans_boxes[:, 0::4]
    dy = trans_boxes[:, 1::4]
    dw = trans_boxes[:, 2::4]
    dh = trans_boxes[:, 3::4]

    cx = anchor_box[:,0]
    cy = anchor_box[:,1]
    w = anchor_box[:,2]
    h = anchor_box[:,3]

    pred_ctr_x = dx * w[:, np.newaxis] + cx[:, np.newaxis]
    pred_ctr_y = dy * h[:, np.newaxis] + cy[:, np.newaxis]
    pred_w = np.exp(dw) * w[:, np.newaxis]
    pred_h = np.exp(dh) * h[:, np.newaxis]

    pred_boxes = np.zeros(trans_boxes.shape, dtype=trans_boxes.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    return pred_boxes

def batch_iou(boxes, box):
    lr = np.maximum(np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),0)
    tb = np.maximum(np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),0)
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union

def non_max_suppression_fast(boxes, probs, max_boxes, overlap_thresh=0.9):
    if len(boxes) == 0:
        return []
    max_boxes = max_boxes
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #print('coordinates', x1, y1, x2, y2)
    #print('coordinates shape', x1.shape, y1.shape, x2.shape, y2.shape)
    #np.testing.assert_array_less(x1, x2)
    #np.testing.assert_array_less(y1, y2)

    #boxes = boxes.astype('float')
    pick = []
    #print('probs',probs)
    #print('shape of probs', probs.shape)
    probs = probs.reshape(-1)
    #print(probs.shape)
    idx = np.argsort(probs[:])
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #print('sorted index',idx)
    while(len(idx)> 0):
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)
        # find the intersection
        xx1_int = np.maximum(x1[i], x1[idx[:last]])
        yy1_int = np.maximum(y1[i], y1[idx[:last]])
        xx2_int = np.minimum(x2[i], x2[idx[:last]])
        yy2_int = np.minimum(y2[i], y2[idx[:last]])

        # find the union
        xx1_un = np.minimum(x1[i], x1[idx[:last]])
        yy1_un = np.minimum(y1[i], y1[idx[:last]])
        xx2_un = np.maximum(x2[i], x2[idx[:last]])
        yy2_un = np.maximum(y2[i], y2[idx[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0.0, xx2_int - xx1_int + 1)
        h = np.maximum(0.0, yy2_int - yy1_int + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[idx[:last]] - inter)

        # delete all indexes from the index list that have
        idx = np.delete(idx, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))
        if len(pick) >= max_boxes:
            break
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    #print('bbox', boxes)
    #print('probs',probs)
    return boxes, probs, pick

def nms(boxes, probs, threshold):
    order = probs.argsort()[::-1]
    keep = [True]*len(order)
    #print(order.shape)
    for i in range(len(order)-1):
        ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False
    return keep
