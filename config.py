import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset = 'PASCAL_VOC'):
    cfg = edict()
    cfg.DATASET = dataset.upper()

    if cfg.DATASET == 'PASCAL_VOC':
        # object categories to classify
        cfg.CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                           'sofa', 'train', 'tvmonitor')
    else:
        cfg.CLASS_NAMES = ('Left','Right')
    cfg.CLASSES = len(cfg.CLASS_NAMES)

    # batch size
    cfg.BATCH_SIZE = 1

    # image width
    cfg.IMAGE_WIDTH = 224

    # image height
    cfg.IMAGE_HEIGHT = 224

    # anchor box, array of [cx, cy, w, h]. To be defined later
    cfg.ANCHOR_BOX = []

    # number of anchor boxes
    cfg.ANCHORS = len(cfg.ANCHOR_BOX)

    # number of anchor boxes per grid
    cfg.ANCHOR_PER_GRID = -1

    #threshould for target label generate(used to select postive/negative samples)
    cfg.neg_max_overlaps = 0.3
    cfg.pos_min_overlaps = 0.7

    #RPN
    cfg.RPN_BATCH_SIZE = 256
    cfg.RPN_FRACTION = 0.5
    cfg.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    cfg.RPN_POSITIVE_WEIGHT = -1.0

    #lr parameters
    cfg.LEARNING_RATE = 0.001
    cfg.MOMENTUM = 0.9
    cfg.DECAY_STEPS = 10000
    cfg.LR_DECAY_FACTOR = 0.5
    cfg.MAX_GRAD_NORM = 1.0
    cfg.WEIGHT_DECAY = 0.0005
    cfg.KEEP_PROB = 0.7

    cfg.LOAD_PRETRAINED_MODEL = True
    cfg.BATCH_NORM_EPSILON = 1e-5

    cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

    cfg.DEBUG_MODE = True

    cfg.TOP_N_DETECTION = 300
    cfg.NMS_THRESH = 0.3
    return cfg

def model_parameters():
    mc                       = base_model_config('FDDB')
    mc.IMAGE_WIDTH           = 960
    mc.IMAGE_HEIGHT          = 640
    mc.BATCH_SIZE            = 1
    mc.ANCHOR_BOX            = set_anchors(mc)
    mc.ANCHORS               = len(mc.ANCHOR_BOX)
    mc.ANCHOR_PER_GRID       = 9# The K in anchor_box_selected function
    mc.H = 40
    mc.W = 60
    mc.PROB_THRESH           = 0.005
    mc.cls = True
    return mc

def set_anchors(mc):
    #the anchor box scale(w & h ) set here
    H, W, B = 40, 60, 9
    anchor_shapes = np.reshape(
        [np.array(
            [
            #[  36.,  37.], [ 366., 174.], [ 115.,  59.],
            #[ 162.,  87.], [  38.,  90.], [ 258., 173.],
            #[ 224., 108.], [  78., 170.], [  72.,  43.]
             [943., 258.],[54., 167.],[453., 490.],[196., 276.],[303., 79.],[113., 247.],[129., 117.],[117., 128.],[227., 298.]
            ])] * H * W,
        (H, W, B, 2)
    )

    center_x = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B),
                (B, H, W)
            ),
            (1, 2, 0)
        ),
        (H, W, B, 1)
    )
    center_y = np.reshape(
        np.transpose(
            np.reshape(
                np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
                (B, W, H)
            ),
            (2, 1, 0)
        ),
        (H, W, B, 1)
    )
    anchors = np.reshape(
        np.concatenate((center_x, center_y, anchor_shapes), axis=3),
        (-1, 4)
    )
    return anchors
