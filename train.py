from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from datetime import datetime
import os.path
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf
from config import *
from netarch import *
import batch_generate
import pandas as pd

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")
tf.app.flags.DEFINE_string('pretrained_model_path', './ResNet/ResNet-50-weights.pkl',"""Path to the pretrained model.""")
tf.app.flags.DEFINE_string('train_dir', './tf_detection',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
def train():
    with tf.Graph().as_default():
        mc = model_parameters()
        mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
        model = ResNet50(mc, FLAGS.gpu)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        data = pd.read_csv('voc_xywh.csv')
        #data = pd.read_csv('FDDB2XYWH.csv')

        data = data.drop('Unnamed: 0', 1)
        #TODO, add trainval split code here;
        img_channel_mean = [103.939, 116.779, 123.68]
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            i_line = np.random.randint(len(data))
            name_str, img, bb_boxes = batch_generate.get_img_by_name(data, i_line, size = (960, 640), dataset = 'PASCAL_VOC')#,dataset = 'FDDB')
            #print(bb_boxes)
            #Normalize
            img = img.astype(np.float32)
            img[:, :, 0] -= img_channel_mean[0]
            img[:, :, 1] -= img_channel_mean[1]
            img[:, :, 2] -= img_channel_mean[2]
            img_per_batch = np.expand_dims(img, axis = 0)
            anchor_box = np.expand_dims(mc.ANCHOR_BOX, axis = 0)
            #if(mc.cls):
            #    labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, cls_map = batch_generate.target_label_generate(bb_boxes, anchor_box, mc, DEBUG = False)
            #else:
            labels, bbox_targets, bbox_inside_weights, bbox_outside_weights , groundtruth = batch_generate.target_label_generate(bb_boxes, anchor_box, mc, DEBUG = False)
            feed_dict = {
                         model.image_input : img_per_batch,
                         model.keep_prob : mc.KEEP_PROB,
                         model.target_label : np.expand_dims(labels, axis = 0),
                         model.target_delta : np.expand_dims(bbox_targets, axis = 0),
                         model.bbox_in_weight : np.expand_dims(bbox_inside_weights, axis = 0),
                         model.bbox_out_weight : np.expand_dims(bbox_outside_weights, axis = 0),
                         model.gt_boxes : groundtruth,
                         #model.cls_map: np.expand_dims(cls_map, axis = 0),# for end2end classification
            }
            losses = sess.run([model._losses, model.train_op], feed_dict = feed_dict)
            print('the training step is {0}, and losses is {1}'.format(step, losses))

            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
              checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=step)




def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()
