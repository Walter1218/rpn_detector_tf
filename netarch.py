from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
import utils
from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf
from graphical_model import graphical_model
import tensorflow.contrib.slim as slim
from roi_proposal import roi_proposal

def _variable_with_weight_decay(name, shape, wd, initializer, trainable=True):
    var = _variable_on_device(name, shape, initializer, trainable)
    if wd is not None and trainable:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_device(name, shape, initializer, trainable=True):
    dtype = tf.float32
    if not callable(initializer):
        var = tf.get_variable(name, initializer=initializer, trainable=trainable)
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var

class ResNet50(graphical_model):
    def __init__(self, mc, gpu_id):
        with tf.device('/gpu:{}'.format(gpu_id)):
            graphical_model.__init__(self, mc)
            self.forward_graph()
            #self.logits_node()
            self.loss_func()
            self.opt_graph()

    def forward_graph(self):
        mc = self.mc
        if mc.LOAD_PRETRAINED_MODEL:
            self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

        conv1 = self.conv_bn_layer(self.image_input, 'conv1', 'bn_conv1', 'scale_conv1', filters=64,size=7, stride=2, freeze=True, conv_with_bias=True)
        pool1 = self.pooling_layer('pool1', conv1, size=3, stride=2, padding='VALID')

        with tf.variable_scope('conv2_x') as scope:
            with tf.variable_scope('res2a'):
                branch1 = self.conv_bn_layer(pool1, 'res2a_branch1', 'bn2a_branch1', 'scale2a_branch1',filters=256, size=1, stride=1, freeze=True, relu=False)
                branch2 = self.residual_branch(pool1, layer_name='2a', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res2b'):
                branch2 = self.residual_branch(res2a, layer_name='2b', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2b = tf.nn.relu(res2a+branch2, 'relu')
            with tf.variable_scope('res2c'):
                branch2 = self.residual_branch(res2b, layer_name='2c', in_filters=64, out_filters=256,down_sample=False, freeze=True)
                res2c = tf.nn.relu(res2b+branch2, 'relu')

        with tf.variable_scope('conv3_x') as scope:
            with tf.variable_scope('res3a'):
                branch1 = self.conv_bn_layer(res2c, 'res3a_branch1', 'bn3a_branch1', 'scale3a_branch1',filters=512, size=1, stride=2, freeze=True, relu=False)
                branch2 = self.residual_branch(res2c, layer_name='3a', in_filters=128, out_filters=512,down_sample=True, freeze=True)
                res3a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res3b'):
                branch2 = self.residual_branch(res3a, layer_name='3b', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3b = tf.nn.relu(res3a+branch2, 'relu')
            with tf.variable_scope('res3c'):
                branch2 = self.residual_branch(res3b, layer_name='3c', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3c = tf.nn.relu(res3b+branch2, 'relu')
            with tf.variable_scope('res3d'):
                branch2 = self.residual_branch(res3c, layer_name='3d', in_filters=128, out_filters=512,down_sample=False, freeze=True)
                res3d = tf.nn.relu(res3c+branch2, 'relu')

        with tf.variable_scope('conv4_x') as scope:
            with tf.variable_scope('res4a'):
                branch1 = self.conv_bn_layer(res3d, 'res4a_branch1', 'bn4a_branch1', 'scale4a_branch1',filters=1024, size=1, stride=2, relu=False)
                branch2 = self.residual_branch(res3d, layer_name='4a', in_filters=256, out_filters=1024,down_sample=True)
                res4a = tf.nn.relu(branch1+branch2, 'relu')
            with tf.variable_scope('res4b'):
                branch2 = self.residual_branch(res4a, layer_name='4b', in_filters=256, out_filters=1024,down_sample=False)
                res4b = tf.nn.relu(res4a+branch2, 'relu')
            with tf.variable_scope('res4c'):
                branch2 = self.residual_branch(res4b, layer_name='4c', in_filters=256, out_filters=1024,down_sample=False)
                res4c = tf.nn.relu(res4b+branch2, 'relu')
            with tf.variable_scope('res4d'):
                branch2 = self.residual_branch(res4c, layer_name='4d', in_filters=256, out_filters=1024,down_sample=False)
                res4d = tf.nn.relu(res4c+branch2, 'relu')
            with tf.variable_scope('res4e'):
                branch2 = self.residual_branch(res4d, layer_name='4e', in_filters=256, out_filters=1024,down_sample=False)
                res4e = tf.nn.relu(res4d+branch2, 'relu')
            with tf.variable_scope('res4f'):
                branch2 = self.residual_branch(res4e, layer_name='4f', in_filters=256, out_filters=1024,down_sample=False)
                res4f = tf.nn.relu(res4e+branch2, 'relu')

        dropout4 = tf.nn.dropout(res4f, self.keep_prob, name='drop4')

        #RPN Layer
        #3*3 conv layer
        #with tf.variable_scope('rpn', 'rpn',regularizer=tf.contrib.layers.l2_regularizer(mc.WEIGHT_DECAY)):
        #if(mc.is_training):
        #initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        #initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        #rpn = slim.conv2d(res4f, 512, [3, 3], trainable=True, weights_initializer=initializer, scope="rpn_conv/3x3")
        #rpn_cls_score = slim.conv2d(rpn, mc.ANCHOR_PER_GRID, [1, 1], trainable=True ,weights_initializer=initializer,
        #                            padding='VALID', activation_fn=None, scope='rpn_cls_score')
        #rpn_bbox_pred = slim.conv2d(rpn, mc.ANCHOR_PER_GRID * 4, [1, 1], trainable=True ,weights_initializer=initializer_bbox,
        #                            padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        rpn = self.conv_layer('rpn', dropout4, filters=512, size=3, stride=1, padding='SAME', xavier=False, relu=False, mean = 0.0, stddev=0.001)
        rpn_cls_score = self.conv_layer('rpn_cls_score', rpn, filters = mc.ANCHOR_PER_GRID * 2, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        rpn_bbox_pred = self.conv_layer('rpn_bbox_pred', rpn, filters = mc.ANCHOR_PER_GRID * 4, size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        #cls_pred = self.conv_layer('cls_pred', rpn, filters = 2* mc.ANCHOR_PER_GRID , size = 1, stride = 1, padding = 'VALID', xavier = False, relu = False, mean = 0.0, stddev = 0.001)
        rpn_cls_score_reshape = self.spatial_reshape_layer(rpn_cls_score, 2, name = 'rpn_cls_score_reshape')
        rpn_cls_prob = self.spatial_softmax(rpn_cls_score_reshape, name='rpn_cls_prob')
        rpn_cls_prob_reshape = self.spatial_reshape_layer(rpn_cls_prob , mc.ANCHOR_PER_GRID * 2, name = 'rpn_cls_prob_reshape')

        #cls_score_reshape = self.spatial_reshape_layer(cls_pred, 2, name = 'cls_score_reshape')
        #cls_prob = self.spatial_softmax(cls_score_reshape, name='cls_prob')
        #cls_prob_reshape = self.spatial_reshape_layer(cls_prob , mc.ANCHOR_PER_GRID * 2 , name = 'cls_prob_reshape')

        #proposal_nms here
        rois, rpn_scores = self.proposal_nms_layer(mc, rpn_cls_score_reshape, rpn_bbox_pred)
        #select postive/negative samples from proposaled nms bboxes
        

        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        #################################################
        ########DEBUG PRINT##############################
        #self._predictions["rpn_rois"] = rois
        #self._predictions["rpn_scores"] = rpn_scores
        #self._predictions["cls_pred"] = cls_score_reshape

        self.det_probs = rpn_cls_prob_reshape
        self.det_boxes = rpn_bbox_pred
        #self.det_cls = cls_prob_reshape

    def proposaled_target_layer(self, mc, rois, rpn_scores, gt_boxes, name = 'proposaled_target_layer'):
        with tf.variable_scope(name) as scope:
            pass

    def proposal_nms_layer(self, mc, rpn_cls_prob_reshape, rpn_bbox_pred, name = 'proposal_nms_layer'):
        with tf.variable_scope(name) as scope:
            print('proposal_nms_layer process')
            self.H = mc.H
            self.W = mc.W
            self.ANCHOR_PER_GRID = mc.ANCHOR_PER_GRID
            self.ANCHOR_BOX = mc.ANCHORS
            self.TOP_N_DETECTION = mc.TOP_N_DETECTION
            self.IM_H = mc.IMAGE_HEIGHT
            self.IM_W = mc.IMAGE_WIDTH
            self.NMS_THRESH = mc.NMS_THRESH
            rois, rpn_scores = tf.py_func(roi_proposal,[rpn_cls_prob_reshape, rpn_bbox_pred, self.H, self.W, self.ANCHOR_PER_GRID,
                                          self.ANCHOR_BOX, self.TOP_N_DETECTION, self.NMS_THRESH, self.IM_H, self.IM_W],
                                          [tf.float32, tf.float32])
            rois.set_shape([mc.TOP_N_DETECTION, 5])
            rpn_scores.set_shape([mc.TOP_N_DETECTION, 1])
        return rois, rpn_scores

    def residual_branch(self, inputs, layer_name, in_filters, out_filters, down_sample=False, freeze=False):
        with tf.variable_scope('res'+layer_name+'_branch2'):
            stride = 2 if down_sample else 1
            output = self.conv_bn_layer(inputs,conv_param_name='res'+layer_name+'_branch2a',
                                         bn_param_name='bn'+layer_name+'_branch2a',scale_param_name='scale'+layer_name+'_branch2a',
                                         filters=in_filters, size=1, stride=stride, freeze=freeze)
            output = self.conv_bn_layer(output,conv_param_name='res'+layer_name+'_branch2b',
                                         bn_param_name='bn'+layer_name+'_branch2b',scale_param_name='scale'+layer_name+'_branch2b',
                                         filters=in_filters, size=3, stride=1, freeze=freeze)
            output = self.conv_bn_layer(output,conv_param_name='res'+layer_name+'_branch2c',
                                         bn_param_name='bn'+layer_name+'_branch2c',scale_param_name='scale'+layer_name+'_branch2c',
                                         filters=out_filters, size=1, stride=1, freeze=freeze, relu=False)
        return output

    def conv_bn_layer(self, inputs, conv_param_name, bn_param_name, scale_param_name, filters, size, stride, padding='SAME',
                      freeze=False, relu=True, conv_with_bias=False, stddev=0.001):
        mc = self.mc
        with tf.variable_scope(conv_param_name) as scope:
            channels = inputs.get_shape()[3]
            if mc.LOAD_PRETRAINED_MODEL:
                cw = self.caffemodel_weight
                #because the weights parameters stored in caffe model is different with tensorflow, so we need transpose to tf structure
                kernel_val = np.transpose(cw[conv_param_name][0], [2,3,1,0])
                if conv_with_bias:
                    bias_val = cw[conv_param_name][1]
                mean_val   = cw[bn_param_name][0]
                var_val    = cw[bn_param_name][1]
                gamma_val  = cw[scale_param_name][0]
                beta_val   = cw[scale_param_name][1]
            else:
                kernel_val = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                if conv_with_bias:
                    bias_val = tf.constant_initializer(0.0)
                mean_val   = tf.constant_initializer(0.0)
                var_val    = tf.constant_initializer(1.0)
                gamma_val  = tf.constant_initializer(1.0)
                beta_val   = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],wd=mc.WEIGHT_DECAY, initializer=kernel_val, trainable=(not freeze))
            self.model_params += [kernel]
            if conv_with_bias:
                biases = _variable_on_device('biases', [filters], bias_val,trainable=(not freeze))
                self.model_params += [biases]
            gamma = _variable_on_device('gamma', [filters], gamma_val,trainable=(not freeze))
            beta  = _variable_on_device('beta', [filters], beta_val,trainable=(not freeze))
            mean  = _variable_on_device('mean', [filters], mean_val, trainable=False)
            var   = _variable_on_device('var', [filters], var_val, trainable=False)
            self.model_params += [gamma, beta, mean, var]

            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
            if conv_with_bias:
                conv = tf.nn.bias_add(conv, biases, name='bias_add')

            conv = tf.nn.batch_normalization(conv, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=mc.BATCH_NORM_EPSILON, name='batch_norm')

            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def pooling_layer(self, layer_name, inputs, size, stride, padding='SAME'):
        with tf.variable_scope(layer_name) as scope:
            out =  tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],padding=padding)
        return out

    def conv_layer(self, layer_name, inputs, filters, size, stride, padding='SAME',freeze=False, xavier=False, relu=True, mean = 0.0, stddev=0.001):
        mc = self.mc
        use_pretrained_param = False
        if mc.LOAD_PRETRAINED_MODEL:
            cw = self.caffemodel_weight
            if layer_name in cw:
                kernel_val = np.transpose(cw[layer_name][0], [2,3,1,0])
                bias_val = cw[layer_name][1]
                # check the shape
                if (kernel_val.shape == (size, size, inputs.get_shape().as_list()[-1], filters)) and (bias_val.shape == (filters, )):
                    use_pretrained_param = True
                else:
                    print ('Shape of the pretrained parameter of {} does not match, use randomly initialized parameter'.format(layer_name))
            else:
                print ('Cannot find {} in the pretrained model. Use randomly initialized parameters'.format(layer_name))

        if mc.DEBUG_MODE:
            print('Input tensor shape to {}: {}'.format(layer_name, inputs.get_shape()))

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]
            # re-order the caffe kernel with shape [out, in, h, w] -> tf kernel with
            # shape [h, w, in, out]
            if use_pretrained_param:
                if mc.DEBUG_MODE:
                    print ('Using pretrained model for {}'.format(layer_name))
                kernel_init = tf.constant(kernel_val , dtype=tf.float32)
                bias_init = tf.constant(bias_val, dtype=tf.float32)
            elif xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(mean = 0.0, stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = _variable_with_weight_decay('kernels', shape=[size, size, int(channels), filters],wd=mc.WEIGHT_DECAY, initializer=kernel_init, trainable=(not freeze))

            biases = _variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
            self.model_params += [kernel, biases]

            conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding,name='convolution')
            conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

        return out
