""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tf_sampling import farthest_point_sample, gather_point

def sample(npoint, xyz):
    # Add Visualization predicted
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xyz[0,:,0],xyz[0,:,1], xyz[0,:,2], c='red', marker='.')
    #data = np.zeros((1,xyz.shape[0],3))
    #data[0,...] = xyz
    new_xyz=gather_point(xyz,farthest_point_sample(npoint,xyz))
    
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
    sess = tf.Session()
    sess.run(init)
    new_data = new_xyz.eval(session = sess)
    #new_data = new_data.transpose()
    #ax.scatter(new_data[0,:,0],new_data[0,:,1], new_data[0,:,2], c='blue', marker='.')
    #plt.show()
    sess.close() 
    return new_data

def farthest_point_sample_edge(npoint, xyz,edge_ids):
    data = tf.Variable(tf.zeros([1,xyz.shape[1], 3], tf.int32),validate_shape = False)
    data.set_shape((1,None,3)) 
    data[0,...] = xyz[0,edge_ids,:]
    new_xyz=gather_point(xyz,farthest_point_sample(npoint,xyz))
    
    return new_xyz
