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
from pointnet_util_edge_only import pointnet_sa_module#
from sklearn.metrics import mean_squared_error
from eul2rot import euler2rotm
from sklearn.neighbors import NearestNeighbors


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
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 4))
    labels_pl_class = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl,labels_pl_class

def placeholder_inputs_pose(batch_size, num_point):
    labels_pl_pose = tf.placeholder(tf.float32, shape=(batch_size,7))
    return labels_pl_pose

def placeholder_inputs_anchor(batch_size, num_point):
    labels_pl_anchor = tf.placeholder(tf.int32, shape=(batch_size))
    return labels_pl_anchor


def get_corners(batch_size,bb_param):
	#bb_param (batchsize, w, l, h, x, y ,z) Nx6
	corners = tf.placeholder(tf.float32,shape=(batch_size,6,8,3))
	'''
	corners[:,0,0:3] = [bb_param[:,3]+bb_param[:,0]/2,bb_param[:,4]+bb_param[:,1]/2,bb_param[:,5]-bb_param[:,2]/2]
	corners[:,1,0:3] = [bb_param[:,3]+bb_param[:,0]/2,bb_param[:,4]-bb_param[:,1]/2,bb_param[:,5]-bb_param[:,2]/2]
	corners[:,2,0:3] = [bb_param[:,3]+bb_param[:,0]/2,bb_param[:,4]+bb_param[:,1]/2,bb_param[:,5]+bb_param[:,2]/2]
	corners[:,3,0:3] = [bb_param[:,3]+bb_param[:,0]/2,bb_param[:,4]-bb_param[:,1]/2,bb_param[:,5]+bb_param[:,2]/2]
	corners[:,4,0:3] = [bb_param[:,3]-bb_param[:,0]/2,bb_param[:,4]+bb_param[:,1]/2,bb_param[:,5]-bb_param[:,2]/2]
	corners[:,5,0:3] = [bb_param[:,3]-bb_param[:,0]/2,bb_param[:,4]-bb_param[:,1]/2,bb_param[:,5]-bb_param[:,2]/2]
	corners[:,6,0:3] = [bb_param[:,3]-bb_param[:,0]/2,bb_param[:,4]+bb_param[:,1]/2,bb_param[:,5]+bb_param[:,2]/2]
	corners[:,7,0:3] = [bb_param[:,3]-bb_param[:,0]/2,bb_param[:,4]-bb_param[:,1]/2,bb_param[:,5]+bb_param[:,2]/2]
	'''
	return corners

def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return tf.reduce_mean(losses)


def get_model_class(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.   npoint=512,128 nsample = 32, 64
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,npoint=512, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer1'),scope_reuse = False, use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=1, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer2'),scope_reuse = False)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope=('layer3'),scope_reuse = False)

    # Fully connected layers for classification
    net_class = tf.concat([tf.reshape(l3_points, [batch_size, -1]),l0_xyz[:,:,3]],axis=-1)#tf.reshape(l3_points, [batch_size, -1])
    net_class = tf_util.fully_connected(net_class, 512, bn=True, is_training=is_training, scope='fc1_class', bn_decay=bn_decay)
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp1_class')
    net_class = tf_util.fully_connected(net_class, 256, bn=True, is_training=is_training, scope='fc2_class', bn_decay=bn_decay)
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp2_class')#256
    net_class = tf_util.fully_connected(net_class, 40, activation_fn=None, scope='fc3_class')

    # Fully connected layers for center estimation and offset
    net_pose = tf.reshape(l3_points, [batch_size, -1])
    net_pose = tf_util.fully_connected(net_pose, 512, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 256, bn=True, is_training=is_training, scope='fc2_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp2_pose')
    net_pose = tf_util.fully_connected(net_pose, 7, activation_fn=None, scope='fc3_pose')

    # Fully connected layers for anchor box classification
    net_anchor = tf.reshape(l3_points, [batch_size, -1])
    net_anchor = tf_util.fully_connected(net_anchor, 512, bn=True, is_training=is_training, scope='fc1_anchor', bn_decay=bn_decay)#512
    net_anchor = tf_util.dropout(net_anchor, keep_prob=0.5, is_training=is_training, scope='dp1_anchor')
    net_anchor = tf_util.fully_connected(net_anchor, 256, bn=True, is_training=is_training, scope='fc2_anchor', bn_decay=bn_decay)
    net_anchor = tf_util.dropout(net_anchor, keep_prob=0.5, is_training=is_training, scope='dp2_anchor')#256
    net_anchor = tf_util.fully_connected(net_anchor, 4, activation_fn=None, scope='fc3_anchor')


    return net_class, net_pose, net_anchor, end_points

def get_model_pose(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.   npoint=512,128 nsample = 32, 64
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer11'),scope_reuse = False, use_nchw=False)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=1, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer22'),scope_reuse = False)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope=('layer33'),scope_reuse = False)

    # Fully connected layers for pose estimation
    net_pose = tf.reshape(l3_points, [batch_size, -1])
    #net_pose = tf_util.fully_connected(net_pose, 1024, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)#512
    #net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 512, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 256, bn=True, is_training=is_training, scope='fc2_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp2_pose')
    net_pose = tf_util.fully_connected(net_pose, 6, activation_fn=None, scope='fc3_pose')

    return net_pose, end_points

def get_loss_pose(pred_pose, label_pose, end_points,bsize):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # L1 norm

    reg_loss = tf.norm(label_pose - pred_pose)
    loss = huber_loss(reg_loss, delta=2.0)
    #tf.summary.scalar('center loss', center_loss)
    #stage1_center_dist = tf.norm(center_label - \
    #    end_points['stage1_center'], axis=-1)
    #stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)
    #tf.summary.scalar('stage1 center loss', stage1_center_loss)


    #loss = tf.reduce_mean(tf.square(tf.abs(label_pose-pred_pose)))
    #pred_corners = get_corners(bsize,pred_pose)
    #actual_corners = get_corners(bsize,pred_pose)
    tf.summary.scalar('regression loss', loss)
    tf.add_to_collection('losses_pose', loss)
    return loss

def get_loss_class(pred_class, label_class, end_points):
    cls = tf.one_hot(label_class, 40)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=cls, logits=pred_class)) 
    #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_class, labels=label_class)
    classify_loss = loss#tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses_class', classify_loss)
    return classify_loss

def get_loss_anchor(pred_anchor, label_anchor):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_anchor, labels=label_anchor)
    anchor_loss = tf.reduce_mean(loss)
    tf.summary.scalar('anchor loss', anchor_loss)
    tf.add_to_collection('losses_anchor', anchor_loss)
    return anchor_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,4))
        output, _ = get_model(inputs, tf.constant(True))

