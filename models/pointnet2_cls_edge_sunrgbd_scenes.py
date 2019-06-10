"""
    PointNet++ Model for point clouds pose detection
"""

import os
import sys
import math
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import numpy as np
import tf_util
import modelnet40_dataset_orig
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from sklearn.metrics import mean_squared_error
from eul2rot import euler2rotm
from sklearn.neighbors import NearestNeighbors
from tensorflow.python.ops import array_ops


# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_NEXT_AXIS = [1, 2, 0, 1]

def placeholder_inputs_class(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl_class = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl,labels_pl_class

def placeholder_inputs_pose(batch_size, num_point):
    labels_pl_pose = tf.placeholder(tf.float32, shape=(batch_size, num_point,7))
    return labels_pl_pose

def placeholder_inputs_anchor(batch_size, num_point):
    labels_pl_anchor = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    labels_pl_anchor_theta = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return labels_pl_anchor,labels_pl_anchor_theta

def placeholder_inputs_edge(batch_size, num_point):
    labels_pl_edge = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return labels_pl_edge

def placeholder_inputs_edge_cnn(batch_size):
    labels_pl_edge = tf.placeholder(tf.float32, shape=(batch_size, 32,32,3))
    return labels_pl_edge

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)

def get_model_edge_cnn(point_cloud, img_cnn, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud#tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = point_cloud#tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    #Encoder
    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=3064, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius=1, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    '''
    # CNN layers for edge detection
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(inputs=img_cnn,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)#32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1,filters=128,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)#64
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    net_edge = tf.reshape(pool2, [batch_size, -1])
    net_edge = tf_util.fully_connected(net_edge, 2048, bn=True, is_training=is_training, scope='fc1_edge', bn_decay=bn_decay)
    net_edge = tf_util.dropout(net_edge, keep_prob=0.5, is_training=is_training, scope='dp1_edge')
    net_edge = tf_util.fully_connected(net_edge, 1024, bn=True, is_training=is_training, scope='fc2_edge', bn_decay=bn_decay)
    '''
    #Decoder
    # Feature Propagation layers for decoder
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')


    #Classification layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_class', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1_class')
    net_class = tf_util.conv1d(net, 11, 1, padding='VALID', activation_fn=None, scope='fc2_class')
    #prediction = tf.nn.relu(net_edge)#tf.round(tf.nn.sigmoid(net_edge))
    '''
    #Centroid and offset layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.conv1d(net, 7, 1, padding='VALID', activation_fn=None, scope='fc2_pose')

    #Anchor box classification
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_anchor', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1_anchor')
    net_anchor = tf_util.conv1d(net, 5, 1, padding='VALID', activation_fn=None, scope='fc2_anchor')


    #Anchor theta classification
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1_anchor_theta', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1_anchor_theta')
    net_anchor_theta = tf_util.conv1d(net, 5, 1, padding='VALID', activation_fn=None, scope='fc2_anchor_theta')
    '''
    '''
    # Fully connected layers for classification
    net_class = tf.reshape(l3_points, [batch_size, -1])
    net_class = tf.concat([net_class,net_edge],axis=-1)
    net_class = tf_util.fully_connected(net_class, 512, bn=True, is_training=is_training, scope='fc1_class', bn_decay=bn_decay)
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp1_class')
    net_class = tf_util.fully_connected(net_class, 256, bn=True, is_training=is_training, scope='fc2_class', bn_decay=bn_decay)
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp2_class')#256
    net_class = tf_util.fully_connected(net_class, 11, activation_fn=None, scope='fc3_class')
    
    # Fully connected layers for center estimation and offset
    net_pose = tf.reshape(l3_points, [batch_size, -1])
    net_pose = tf.concat([net_pose,net_edge],axis=-1)
    net_pose = tf_util.fully_connected(net_pose, 512, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 256, bn=True, is_training=is_training, scope='fc2_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp2_pose')
    net_pose = tf_util.fully_connected(net_pose, 7, activation_fn=None, scope='fc3_pose')

    # Fully connected layers for anchor box classification
    net_anchor = tf.reshape(l3_points, [batch_size, -1])
    net_anchor = tf.concat([net_anchor,net_edge],axis=-1)
    net_anchor = tf_util.fully_connected(net_anchor, 512, bn=True, is_training=is_training, scope='fc1_anchor', bn_decay=bn_decay)#512
    net_anchor = tf_util.dropout(net_anchor, keep_prob=0.5, is_training=is_training, scope='dp1_anchor')
    net_anchor = tf_util.fully_connected(net_anchor, 256, bn=True, is_training=is_training, scope='fc2_anchor', bn_decay=bn_decay)
    net_anchor = tf_util.dropout(net_anchor, keep_prob=0.5, is_training=is_training, scope='dp2_anchor')#256
    net_anchor = tf_util.fully_connected(net_anchor, 4, activation_fn=None, scope='fc3_anchor')
    
    # Fully connected layers for anchor box classification
    net_anchor_theta = tf.reshape(l3_points, [batch_size, -1])
    net_anchor_theta = tf.concat([net_anchor_theta,net_edge],axis=-1)
    net_anchor_theta = tf_util.fully_connected(net_anchor_theta, 512, bn=True, is_training=is_training, scope='fc1_anchor_theta', bn_decay=bn_decay)#512
    net_anchor_theta = tf_util.dropout(net_anchor_theta, keep_prob=0.5, is_training=is_training, scope='dp1_anchor_theta')
    net_anchor_theta = tf_util.fully_connected(net_anchor_theta, 256, bn=True, is_training=is_training, scope='fc2_anchor_theta', bn_decay=bn_decay)
    net_anchor_theta = tf_util.dropout(net_anchor_theta, keep_prob=0.5, is_training=is_training, scope='dp2_anchor_theta')#256
    net_anchor_theta = tf_util.fully_connected(net_anchor_theta, 4, activation_fn=None, scope='fc3_anchor_theta')
    '''
    return net_class,end_points#, net_pose, net_anchor,net_anchor_theta

def get_loss_pose(pred_pose, label_pose, end_points,bsize):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # L1 norm
    reg_loss = tf.norm(label_pose - pred_pose)
    loss = huber_loss(reg_loss, delta=2.0)
    tf.summary.scalar('regression loss', loss)
    tf.add_to_collection('losses_pose', loss)
    return loss

def get_loss_class(pred_class, label_class, end_points):
    '''
    cls = tf.one_hot(label_class, 10)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=cls, logits=pred_class)) 
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_class, labels=label_class)
    classify_loss = loss#tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses_class', classify_loss)
    '''
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_class, labels=label_class)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses_class', classify_loss)
    return classify_loss

def get_loss_anchor(pred_anchor, label_anchor):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_anchor, labels=label_anchor)
    anchor_loss = tf.reduce_mean(loss)
    tf.summary.scalar('anchor loss', anchor_loss)
    tf.add_to_collection('losses_anchor', anchor_loss)
    return anchor_loss

def get_loss_anchor_theta(pred_anchor_theta, label_anchor_theta):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_anchor_theta, labels=label_anchor_theta)
    anchor_loss_theta = tf.reduce_mean(loss)
    tf.summary.scalar('anchor loss_theta', anchor_loss_theta)
    tf.add_to_collection('losses_anchor_theta', anchor_loss_theta)
    return anchor_loss_theta

def get_loss_edge(pred_edge, label_edge):
    cls = tf.one_hot(label_edge, 2)
    flat_logits = tf.reshape(tensor=cls, shape=(-1, 2))
    flat_pred = tf.reshape(tensor=pred_edge, shape=(-1, 2))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=flat_logits, logits=flat_pred)) 
    edge_loss = loss#tf.reduce_mean(loss)
    tf.summary.scalar('edge loss', edge_loss)
    tf.add_to_collection('losses_edge', edge_loss)
    return edge_loss

def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    sigmoid_p = tf.nn.sigmoid(logits)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For positive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    labels = tf.one_hot(labels, 11)
    pos_p_sub = array_ops.where(labels > zeros, labels - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(labels > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    tf.summary.scalar('classify loss', tf.reduce_sum(per_entry_cross_ent))
    tf.add_to_collection('losses_class', tf.reduce_sum(per_entry_cross_ent))
    return tf.reduce_sum(per_entry_cross_ent)
    '''
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[2]
    
    model_out = tf.nn.relu(logits)#tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=2)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    tf.summary.scalar('classify loss', tf.reduce_mean(reduced_fl))
    tf.add_to_collection('losses_class', tf.reduce_mean(reduced_fl))
    
    return tf.reduce_mean(reduced_fl)
    '''


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))

