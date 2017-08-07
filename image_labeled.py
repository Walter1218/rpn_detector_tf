import cv2
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import config
from netarch import *
import utils,batch_generate
img_lists = glob.glob('./Caltech_WebFaces/*.jpg')
#print(img_lists)
#print(len(img_lists))
size=(960,640)
img_channel_mean = [103.939, 116.779, 123.68]
with tf.Graph().as_default():
    mc = config.model_parameters()
    mc.LOAD_PRETRAINED_MODEL = False
    model = ResNet50(mc, '0')
    saver = tf.train.Saver(model.model_params)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, './tf_detection/model.ckpt-99999')
        #img_per_batch = np.expand_dims(img, axis = 0)
        #det_probs, det_boxes = sess.run([model.det_probs, model.det_boxes],feed_dict={model.image_input:[img], model.keep_prob: 1.0})
        for i in range(len(img_lists)):
            img = cv2.imread(img_lists[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,size)
            img_copy = img
            print('process index {}'.format(i))
            img = img.astype(np.float32)
            img[:, :, 0] -= img_channel_mean[0]
            img[:, :, 1] -= img_channel_mean[1]
            img[:, :, 2] -= img_channel_mean[2]
            #img_ = np.expand_dims(img, axis = 0)
            det_probs, det_boxes = sess.run([model.det_probs, model.det_boxes],feed_dict={model.image_input:[img], model.keep_prob: 1.0})
            print('det index {} success'.format(i))
            box_probs = np.reshape(det_probs[0],[-1,2])[:,1]
            box_delta = np.reshape(det_boxes[0],[21600,4])
            anchor_box = mc.ANCHOR_BOX
            pred_box_xyxy = utils.bbox_delta_convert_inv(anchor_box, box_delta)
            box_nms, probs_nms, pick = utils.non_max_suppression_fast(pred_box_xyxy, box_probs, 100, overlap_thresh=0.2)
            box_nms = box_nms[probs_nms>0.90]
            box = box_nms
            #box = batch_generate.bbox2cxcy(box)
            #print(len(box))
            for j in range(len(box)):
                xmin = int(box[j,0])
                ymin = int(box[j,1])
                xmax = int(box[j,2])
                ymax = int(box[j,3])
                #print(x,y,w,h)
                cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), (0,0,0), thickness= 10)
            saving_name = img_lists[i]
            cv2.imwrite(saving_name, img_copy)
