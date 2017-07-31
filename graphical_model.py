from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
import utils

class graphical_model:
    def __init__(self, mc):
        self.mc = mc
        #Input X
        self.image_input = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 3],name='image_input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        #Input supervised Y(target label cls & target regression value delta)
        self.target_label = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.H , mc.W , mc.ANCHOR_PER_GRID], name = 'target_label')
        self.target_delta = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.H , mc.W , mc.ANCHOR_PER_GRID * 4], name = 'target_delta')
        self.bbox_in_weight = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.H , mc.W , mc.ANCHOR_PER_GRID * 4], name = 'bbox_in_weight')
        self.bbox_out_weight = tf.placeholder(tf.float32, [mc.BATCH_SIZE, mc.H , mc.W , mc.ANCHOR_PER_GRID * 4], name = 'bbox_out_weight')

        self._predictions = {}
        self._anchor_targets = {}
        self._losses = {}
        self.model_params = []

    def forward_graph(self):
        """
        modify this func in netarch.py
        """
        raise NotImplementedError

    def logits_node(self):
        mc = self.mc
        with tf.variable_scope('predict_output') as scope:
            region_proposal_scores = self.rpn_probs
            region_proposal_delta = self.rpn_delta
            #rpn_cls_prob_reshape = self.reshape_layer(region_proposal_scores,2,'region_proposal_scores')
            rpn_cls_prob = tf.reshape(region_proposal_scores, [-1])
            rpn_cls_prob = tf.nn.softmax(rpn_cls_prob)#self._softmax_layer(region_proposal_scores, "rpn_cls_prob")
            #print(rpn_cls_prob.shape)
            #rpn_cls_prob = self.reshape_layer(rpn_cls_prob, mc.ANCHOR_PER_GRID * 2, "rpn_cls_prob")
            self.rpn_cls_prob = rpn_cls_prob#rpn_cls_prob
            self.rpn_delta = region_proposal_delta

            #self._predictions["rpn_cls_score"] = region_proposal_scores
            self._predictions["rpn_cls_prob"] = self.rpn_cls_prob
            self._predictions["rpn_bbox_pred"] = self.rpn_delta


            self.det_probs = self.rpn_cls_prob#tf.reduce_max(rpn_cls_prob, 2, name='score')
            self.det_boxes = region_proposal_delta

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box,axis=dim))
        return loss_box

    def loss_func(self, sigma_rpn=3.0):
        mc = self.mc
        with tf.variable_scope('cnn-loss') as scope:
            #target label Y
            self._anchor_targets["rpn_labels"] = tf.to_int32(self.target_label)
            #self._anchor_targets["rpn_labels"] = tf.convert_to_tensor(tf.cast(self._anchor_targets["rpn_labels"],tf.int32), name = 'rpn_labels')

            #print(self._anchor_targets["rpn_labels"])
            self._anchor_targets["rpn_bbox_targets"] = self.target_delta
            self._anchor_targets["rpn_bbox_inside_weights"] = self.bbox_in_weight
            self._anchor_targets["rpn_bbox_outside_weights"] = self.bbox_out_weight
            #rpn. fg/bg loss
            rpn_cls_score = tf.reshape(self._predictions["rpn_cls_score_reshape"], [-1, 2])#tf.reshape(self._predictions['rpn_cls_prob'], [-1,2])
            #print(rpn_cls_score.shape)
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            #print(rpn_label.shape)
            fg_keep = tf.equal(rpn_label, 1)
            rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep), [-1, 2]) # shape (N, 2)
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep), [-1])
            rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label)
            rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)
            #rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            #rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1])
            #print(rpn_cls_score.shape)
            #rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            #print(rpn_label.shape)
            #rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            #print(rpn_bbox_pred.shape)
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [mc.BATCH_SIZE, mc.H ,mc.W, mc.ANCHOR_PER_GRID * 4], name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_inside_weights = tf.reshape(rpn_bbox_inside_weights, [mc.BATCH_SIZE, mc.H ,mc.W, mc.ANCHOR_PER_GRID * 4], name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_bbox_outside_weights = tf.reshape(rpn_bbox_outside_weights, [mc.BATCH_SIZE, mc.H ,mc.W, mc.ANCHOR_PER_GRID * 4], name = 'rpn_bbox_outside_weights')

            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box

            loss = rpn_cross_entropy + rpn_loss_box
            self._losses['total_loss'] = loss
        return loss

    def opt_graph(self):
        mc = self.mc
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(mc.LEARNING_RATE,
                                        self.global_step,
                                        mc.DECAY_STEPS,
                                        mc.LR_DECAY_FACTOR,
                                        staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mc.MOMENTUM)
        grads_vars = opt.compute_gradients(self._losses['total_loss'], tf.trainable_variables())

        with tf.variable_scope('clip_gradient') as scope:
            for i, (grad, var) in enumerate(grads_vars):
                grads_vars[i] = (tf.clip_by_norm(grad, mc.MAX_GRAD_NORM), var)

        apply_gradient_op = opt.apply_gradients(grads_vars, global_step=self.global_step)
        with tf.control_dependencies([apply_gradient_op]):
            self.train_op = tf.no_op(name='train')
        #lr = tf.Variable(mc.LEARNING_RATE, trainable=False)
        #momentum = mc.MOMENTUM
        #self.optimizer = tf.train.MomentumOptimizer(lr, momentum)

        # Compute the gradients wrt the loss
        #loss = self._losses['total_loss']
        #gvs = self.optimizer.compute_gradients(loss)
        #self.train_op = self.optimizer.apply_gradients(gvs)

    def spatial_softmax(self, bottom, name):
        input_shape = tf.shape(bottom)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(bottom, [-1, input_shape[3]])),[-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob':
          input_shape = tf.shape(bottom)
          bottom_reshaped = tf.reshape(bottom, [-1])
          reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
          return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def spatial_reshape_layer(self, bottom, d, name):
        input_shape = tf.shape(bottom)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(bottom,[input_shape[0],input_shape[1], -1, int(d)])

    def reshape_layer(self, bottom, d, name):
        input_shape = tf.shape(bottom)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(bottom,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(bottom,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                [0,2,3,1],name=name)

    def reshape_layer_(self, bottom, num_dim, name):
        mc = self.mc
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            #to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            print(bottom.shape)
            reshaped = tf.reshape(bottom,tf.concat(axis=0, values=[[mc.BATCH_SIZE], [num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
        return to_tf

    def filter_prediction(self, boxes, probs):
        mc = self.mc

        if mc.TOP_N_DETECTION < len(probs) and mc.TOP_N_DETECTION > 0:
            order = probs.argsort()[:-mc.TOP_N_DETECTION-1:-1]
            probs = probs[order]
            boxes = boxes[order]
            #cls_idx = cls_idx[order]
        else:
            filtered_idx = np.nonzero(probs>mc.PROB_THRESH)[0]
            probs = probs[filtered_idx]
            boxes = boxes[filtered_idx]
            #cls_idx = cls_idx[filtered_idx]
        return probs, boxes
        #final_boxes = []
        #final_probs = []
        #final_cls_idx = []

        #for c in range(mc.CLASSES):
        #idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == 1]
        #keep = utils.nms(boxes, probs, mc.NMS_THRESH)
        #for i in range(len(keep)):
        #    if keep[i]:
        #        final_boxes.append(boxes[i])
        #        final_probs.append(probs[i])
                #final_cls_idx.append(c)
        #return final_boxes, final_probs#, final_cls_idx
