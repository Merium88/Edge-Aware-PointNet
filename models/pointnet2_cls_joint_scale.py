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
from pointnet_util import pointnet_sa_module,pointnet_sa_module_edge
from sklearn.metrics import mean_squared_error
from eul2rot import euler2rotm
import modelnet_dataset_joint
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
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl_class = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl,labels_pl_class

def placeholder_inputs_pose(batch_size, num_point):
    labels_pl_pose = tf.placeholder(tf.float32, shape=(batch_size,7))
    return labels_pl_pose

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
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer1'),scope_reuse = False, use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer2'),scope_reuse = False)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope=('layer3'),scope_reuse = False)

    # Fully connected layers for classification
    net_class = tf.reshape(l3_points, [batch_size, -1])
    net_class = tf_util.fully_connected(net_class, 512, bn=True, is_training=is_training, scope='fc1_class', bn_decay=bn_decay)#512
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp1_class')
    net_class = tf_util.fully_connected(net_class, 256, bn=True, is_training=is_training, scope='fc2_class', bn_decay=bn_decay)
    net_class = tf_util.dropout(net_class, keep_prob=0.5, is_training=is_training, scope='dp2_class')#256
    net_class = tf_util.fully_connected(net_class, 10, activation_fn=None, scope='fc3_class')


    return net_class, end_points



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
    net_pose = tf_util.fully_connected(net_pose, 7, activation_fn=None, scope='fc3_pose')

    return net_pose, end_points

def get_model_pose_edge(point_cloud, is_training, bn_decay=None):
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
    edge_xyz,edge_bool,grouped = pointnet_sa_module_edge(l0_xyz, l0_points, npoint=512, radius=1, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer11'),scope_reuse = False, use_nchw=True)
    '''
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=1, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope=('layer22'),scope_reuse = False)
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope=('layer33'),scope_reuse = False)

    # Fully connected layers for classification
    net_pose = tf.reshape(l3_points, [batch_size, -1])
    #net_pose = tf_util.fully_connected(net_pose, 1024, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)#512
    #net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 512, bn=True, is_training=is_training, scope='fc1_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp1_pose')
    net_pose = tf_util.fully_connected(net_pose, 256, bn=True, is_training=is_training, scope='fc2_pose', bn_decay=bn_decay)
    net_pose = tf_util.dropout(net_pose, keep_prob=0.5, is_training=is_training, scope='dp2_pose')
    net_pose = tf_util.fully_connected(net_pose, 6, activation_fn=None, scope='fc3_pose')
    '''
    return edge_xyz,edge_bool,grouped

def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = tf.sin(ai), tf.sin(aj), tf.sin(ak)
    ci, cj, ck = tf.cos(ai), tf.cos(aj), tf.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    #M = tf.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        '''
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
        '''
        M = tf.reshape([cj*ck,sj*sc-cs,sj*cc+ss,cj*sk,sj*ss+cc,sj*cs-sc,-sj,cj*si,cj*ci], shape = (3,3))
    return M


def get_add_loss_pose(pred_pose,label_pose, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # For SLoss
    for k in range(0,pred_pose.shape[0]):
       eul_pred = modelnet_dataset_joint.denormalize(pred_pose[k,0:3])
       eul_true = modelnet_dataset_joint.denormalize(label_pose[k,0:3])
       R_pred = euler_matrix(eul_pred[0],eul_pred[1],eul_pred[2],'rzyx')
       R_true = euler_matrix(eul_true[0],eul_true[1],eul_true[2],'rzyx')
       '''        
       R = R_pred
       t_row = tf.reshape(tf.constant([0,0,0],dtype = tf.float32),[1,3])
       R = tf.concat([R,t_row],0)
       t_col = tf.constant([1],dtype = tf.float32)
       T = pred_pose[k,3:6]
       T = tf.reshape(tf.concat([T,t_col],0),[4,1])
       T = tf.concat([R,T],1)
       '''
       data =end_points[k]
       #data_row = tf.ones([1024,1],tf.float32) 
       #data = tf.concat([data,data_row],1)
       Tx_pred = tf.transpose(tf.matmul(R_pred,tf.transpose(data)))
       Tx_pred = tf.reshape(Tx_pred,shape=(1,1024,3))
       '''
       R = R_true
       t_row = tf.reshape(tf.constant([0,0,0],dtype = tf.float32),[1,3])
       R = tf.concat([R,t_row],0)
       t_col = tf.constant([1],dtype = tf.float32)
       T = label_pose[k,3:6]
       T = tf.reshape(tf.concat([T,t_col],0),[4,1])
       T = tf.concat([R,T],1)
       '''
       Tx_true = tf.transpose(tf.matmul(R_true,tf.transpose(data)))
       Tx_true = tf.reshape(Tx_true,shape=(1,1024,3))

       dist,idx = knn_point(1, Tx_true, Tx_pred)
       grouped_xyz = tf.reshape(group_point(Tx_true, idx),shape=(1,1024,3))

       ploss = dist
       #trans_loss = tf.reduce_mean(tf.abs(pred_pose[k,3:6]-label_pose[k,3:6]))
       #ploss = pose_loss + trans_loss
       correct = tf.cast((tf.sqrt(dist) < 0.1), tf.float32)
       accuracy = tf.reduce_mean(correct)
       tf.add_to_collection('Tx_pred',Tx_pred)
       tf.add_to_collection('Tx_true',Tx_true)
       tf.add_to_collection('Tx_neares',grouped_xyz)
       tf.add_to_collection('pose_acc',accuracy)
       #tf.add_to_collection('pose_loss',ploss)
            

    pred_pcd = tf.reshape(tf.get_collection('Tx_pred'),shape=(pred_pose.shape[0],1024,3))
    near_pcd = tf.reshape(tf.get_collection('Tx_true'),shape=(pred_pose.shape[0],1024,3))
    acc = tf.reshape(tf.get_collection('pose_acc'),shape=(pred_pose.shape[0],1))
    #p_loss = tf.reshape(tf.get_collection('pose_loss'),shape=(pred_pose.shape[0],1))

    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred_pcd, near_pcd)),2))/3)
    regression_loss = tf.reduce_mean(tf.abs(label_pose-pred_pose))#loss+tf.reduce_mean(tf.abs(label_pose[:,3:6]-pred_pose[:,3:6]))
    acc_pcd = tf.reduce_mean(acc)
    tf.summary.scalar('regression loss', regression_loss)
    tf.summary.scalar('regression acc', acc_pcd)
    tf.add_to_collection('losses_pose', regression_loss)
    tf.add_to_collection('accuracy_pose', acc_pcd)
    return regression_loss, acc_pcd

def get_loss_pose(pred_pose, label_pose, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    # L1 norm
    loss = tf.reduce_mean(tf.square(tf.abs(label_pose-pred_pose)))
    regression_loss = tf.reduce_mean(tf.norm(pred_pose-label_pose, ord='euclidean',axis=[-2,-1]))
#tf.reduce_mean(tf.abs(label_pose-pred_pose))  #5*loss + 2*
    tf.summary.scalar('regression loss', regression_loss)
    tf.add_to_collection('losses_pose', regression_loss)
    return regression_loss

def get_loss_class(pred_class, label_class, end_points):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_class, labels=label_class)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses_class', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
