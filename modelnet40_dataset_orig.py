'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
    Edited by Mariam Ahmed
'''

import os
import os.path
import json
import numpy as np
import tensorflow as tf
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
sys.path.append('/home/mariam/pointnet2/tf_ops/sampling')
import provider
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tf_grouping import query_ball_point, group_point, knn_point
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from plot_cube import plot_cube
from numpy import linalg as LA
from scipy.spatial import distance
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
import random
import math
from random import randint
from numpy import linspace, meshgrid
import matplotlib.cm as cm
import random
from PIL import Image
from scipy.misc import imsave
from scipy.spatial.distance import cdist

total_count = [0 for _ in range(40)]

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def denormalize(pc):
    #pc = pc[:,0].values.reshape(-1,1)
    #norm_output = norm_output.reshape(-1,1)
    X_std = (pc - (-1)) / (1 - (-1))
    norm_output = X_std * (3.142 - (-3.142)) + (-3.142)
    return norm_output

def hex_to_rgb(value):
    #value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)) 

def normalize_rot(pc):
    #pc = pc[:,0].values.reshape(-1,1)
    #norm_output = norm_output.reshape(-1,1)
    X_std = (pc - (-3.142)) / (3.142 - (-3.142))
    norm_output = X_std * (1 - (-1)) + (-1)
    return norm_output

def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]


def mat2quat(M):
    # Qyx refers to the contribution of the y input vector component to
    # the x output vector component.  Qyx is therefore the same as
    # M[0,1].  The notation is from the Wikipedia article.
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    # Fill only lower half of symmetric matrix
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    # Use Hermitian eigenvectors, values for speed
    vals, vecs = np.linalg.eigh(K)
    # Select largest eigenvector, reorder to w,x,y,z quaternion
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    # Prefer quaternion with positive w
    # (q * -1 corresponds to same rotation as q)
    if q[0] < 0:
        q *= -1
    return q

def rotz(z):
    mat = np.zeros((4,4),dtype = np.float32)
    mat[0,0:4] = [np.cos(z), -np.sin(z), 0, 0]
    mat[1,0:4] = [np.sin(z), np.cos(z), 0, 0]
    mat[2,0:4] = [0, 0, 1, 0]
    mat[3,0:4] = [0, 0, 0, 1] 
    return mat

def rotx(z):
    mat = np.zeros((4,4),dtype = np.float32)
    mat[0,0:4] = [1, 0, 0, 0]
    mat[1,0:4] = [0, np.cos(z), -np.sin(z), 0]
    mat[2,0:4] = [0, np.sin(z), np.cos(z), 0]
    mat[3,0:4] = [0, 0, 0, 1] 
    return mat

def find_index(datapath,name,num,pose):
    des_id = name +"_"+num+"_"+pose
    with open(datapath) as f:
      scene_id = f.readlines()
    for i in range (0,len(scene_id)):
       ind = scene_id[i].split(",")
       if(des_id == ind[0]):
	 #x, y, z, offset_w, offset_l, offset_h, offset_theta
         pose = [float(ind[1]),float(ind[2]),float(ind[3]),float(ind[4]),float(ind[5]),float(ind[6]),float(ind[7])]
         label = int(ind[9])
         break
    return (pose,label)

class ModelNetDataset():
    def __init__(self, root, batch_size = 32, npoints = 1024, split='train', normalize=True, normal_channel=False, modelnet10=True, cache_size=15000, shuffle=None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        
        self.classfile = os.path.join(self.root, 'shape_names.txt')
        self.catclss = [line.rstrip() for line in open(self.classfile)]
        self.classes1 = dict(zip(self.catclss, range(len(self.catclss))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
        shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_full_test.txt'))]
        assert(split=='train' or split=='test')
        
        if (split=='train'):
          shape_names = ['_'.join(x.split('_')[0:-2]) for x in shape_ids[split]]
        else:
           shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]#-2
        shape_pose = ['_'.join(x.split('_')[-1:]) for x in shape_ids[split]]
        shape_num = ['_'.join(x.split('_')[-2:-1]) for x in shape_ids[split]]
        shape_trans = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        if (split=='train'):
           self.datapath = [(shape_names[i],shape_num[i],shape_pose[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]
         
        else:
           self.datapath = [(shape_names[i],shape_num[i],shape_pose[i], os.path.join("/media/mariam/Seagate Backup Plus Drive/ModelNet/modelnet40_normal_resampled", shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

          
        self.cache_size = cache_size 
        self.cache = {}

        if shuffle is None:
            if split == 'train': self.shuffle = True
            else: self.shuffle = False
        else:
            self.shuffle = shuffle

        self.reset()

    def _augment_batch_data(self, batch_data):
        rotated_data = provider.rotate_point_cloud(batch_data)
        return provider.shuffle_points(rotated_data)

    def _augment_batch_data_pose(self, batch_data,batch_label):
        #if self.normal_channel:
        #    rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
        #    rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        #else:
        rotated_data,rot_label = provider.rotate_point_cloud_with_label(batch_data,denormalize(batch_label))
        rot_label[:,0:3] = normalize_rot(rot_label[:,0:3])
        return rotated_data,rot_label


    def _get_item(self, index,data='train'): #modify this
        if index in self.cache:
            point,name,bb_param,bb_label,bb_edge3 = self.cache[index]
        else:
         if data == 'train':
            #Load class
            name = self.classes1[self.datapath[index][0]]
            name = np.array([name]).astype(np.int32)
            #Load points from file
            fn = self.datapath[index]
            point = np.loadtxt(fn[3],delimiter=',').astype(np.float32)
            pose_file = "/home/mariam/pointnet2/data/ModelNet40/"+self.datapath[index][0]+"/" + self.datapath[index][0]+ "_"+self.datapath[index][1]+".txt"
            f = open(pose_file,"r")
            i = 0
            arr = []
            arrays = np.zeros((486,4), dtype=np.float32)
            for line in f:
               arr = np.array(line.split(),dtype=np.float32)
               arrays[i,0:4] = arr[0:4]
               i = i +1
            #Apply transformation
            trans = arrays[(int(self.datapath[index][2]))*3:(int(self.datapath[index][2]))*3+3,3]
            trans = np.transpose(trans) 
            rot_mat = arrays[int(self.datapath[index][2])*3:(int(self.datapath[index][2]))*3+3,0:3]
            T = np.zeros((4,4),dtype = np.float32)
            data = np.zeros((1024,4),dtype = np.float32)  
            size = point[:,0].size       
            if(size>1024):
              data[0:1024,0:3] = point[0:1024,0:3]
            else:
              less = 1024-size
              data[0:size,0:3] = point[:,0:3]
              ind = np.random.choice(point.shape[0], less, replace=True)
              data[size:1024,0:3] = point[ind[:],0:3]         
            point = data[:,0:3]
            T[0:3,0:3] = rot_mat
            T[3,0:3] = 0
            T[0:3,3] = trans
            T[3,3] = 1
            rot_inv = inv(rot_mat)
            data[:,3] = 1 
            ps_trans = np.transpose(np.matmul(inv(T),np.transpose(data)))
            quat = mat2quat(T[0:3,0:3]) 
            euler = quaternion_to_euler(quat[1],quat[2],quat[3],quat[0])
            rand_head = random.uniform(-0.175, +0.175)
            z_mat = rotz(euler[0]+rand_head)#heading change will be +/- 10 degrees
            #Rotate data around z axis
            pcd = np.transpose(np.matmul(z_mat,np.transpose(ps_trans))) 
            point = pcd[:,0:3] 
            if(name==1 or name==2):
              point[:,0] = point[:,0]*0.5
              point[:,2] = point[:,2]*0.5 
            if(name==3):
              point[:,0] = point[:,0]*0.8
              point[:,1] = point[:,1]*0.4
              point[:,1] = point[:,2]*0.4
            if(name==4):
              point[:,0] = point[:,0]*0.5
              point[:,1] = point[:,1]*0.2 
              point[:,2] = point[:,2]*0.5
            if(name==5):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.4 
            if(name==6 ):
              point[:,2] = point[:,2]*0.5
            if(name==7):
              point[:,1] = point[:,1]*0.5
              point[:,2] = point[:,2]*0.5 
            if(name==8):
              point[:,0] = point[:,0]*0.4
              point[:,2] = point[:,2]*0.6 
            if(name==10):
              point[:,0] = point[:,0]*0.6
              point[:,1] = point[:,1]*0.6
              point[:,2] = point[:,2]*0.8 
            if(name==11):
              point[:,0] = point[:,0]*0.1
              point[:,1] = point[:,1]*0.4
              point[:,2] = point[:,2]*0.8 
            if(name==12):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.6 
              point[:,2] = point[:,2]*0.6 
            if(name==13):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.1 
            if(name==14):
              point[:,0] = point[:,0]*0.4
              point[:,2] = point[:,2]*0.6
            if(name==15):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2] 
            if(name==16):
              point[:,0] = point[:,0]*0.3
              point[:,2] = point[:,2]*0.2 
            if(name==19):
              point[:,0] = point[:,0]*0.5
              point[:,1] = point[:,1]*0.6 
            if(name==20):
              point[:,0] = point[:,0]*0.8
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.4 
            if(name==21):
              point[:,0] = point[:,0]*0.6
              point[:,1] = point[:,1]*0.4 
            if(name==22):
              point[:,0] = point[:,0]*0.6
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.6 
            if(name==23):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.7 
            if(name==24):
              point[:,0] = point[:,0]*0.4
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.8 
            if(name==25):
              point[:,0] = point[:,0]*0.6
              point[:,1] = point[:,1]*0.6 
              point[:,2] = point[:,2]*0.7 
            if(name==27):
              point[:,0] = point[:,0]*0.1 
              point[:,1] = point[:,1]*0.1 
              point[:,2] = point[:,2]*0.05 
            if(name==28):
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.7 
            if(name==29):
              point[:,0] = point[:,0]*0.8 
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.4 
            if(name==30):
              point[:,0] = point[:,0]*0.6 
              point[:,1] = point[:,1]*0.6 
              point[:,2] = point[:,2]*0.6 
            if(name==31):
              point[:,0] = point[:,0]*0.6 
              point[:,1] = point[:,1]*0.2 
              point[:,2] = point[:,2]*0.6 
            if(name==32):
              point[:,0] = point[:,0]*0.4 
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.75 
            if(name==33):
              point[:,0] = point[:,0]*0.4 
              point[:,1] = point[:,1]*0.8 
              point[:,2] = point[:,2]*0.5 
            if(name==34):
              point[:,0] = point[:,0]*0.3 
              point[:,1] = point[:,1]*0.8
              point[:,2] = point[:,2]*0.4 
            if(name==35):
              point[:,0] = point[:,0]*0.4 
              point[:,1] = point[:,1]*0.8 
              point[:,2] = point[:,2]*0.7
            if(name==36):
              point[:,1] = point[:,1]*0.4 
              point[:,2] = point[:,2]*0.4  
            if(name==37):
              point[:,0] = point[:,0]*0.8 
              point[:,1] = point[:,1]*0.8 
              point[:,2] = point[:,2]*0.8 
            if(name==38):
              point[:,0] = point[:,0]*0.4 
              point[:,1] = point[:,1]*0.2 
              point[:,2] = point[:,2]*0.75 
            if(name==39):
              point[:,0] = point[:,0]*0.2 
              point[:,1] = point[:,1]*0.6 
              point[:,2] = point[:,2]*0.8             
            #Find Edge points
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(point)
            distances, indices = nbrs.kneighbors(point)
            centroids = np.mean(point[indices[:,:],0:3],axis=1)
            dist = np.sqrt(np.sum((point[:,0:3]-centroids)**2,axis=1))
            edges = np.zeros((1024,3), dtype=np.float32)
            binary_edges = np.zeros((1024,4), dtype=np.float32)
            #edges[:,0] = 1*(dist[:]>0.015)
            edges[:,0] = 1*(dist[:]>0.02)
            binary_edges[:,0:3] = point[:,:]
            binary_edges[:,3] = edges[:,0]
            ind = np.where(edges[:,0]==1)[0]
            edges[ind,0:3] = point[ind,0:3]
            #point = binary_edges
            bb_edge = binary_edges[:,3]

            #Convert Edge to 2D maps
            x = np.arange(-1.0, 1.0, 0.0625)
            y = np.arange(-1.0, 1.0, 0.0625)
            #x, y = np.meshgrid(x, y)
            p = np.zeros((1, 2))
            grds_xy = np.zeros((32,32))
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,binary_edges[:,0:2])))
                 if(d<0.1):
                   grds_xy[l,t] = 1 

            grds_yz = np.zeros((32,32))
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,binary_edges[:,1:3])))
                 if(d<0.1):
                  grds_yz[l,t] = 1 

            grds_xz = np.zeros((32,32))
            e = np.zeros((1024,2))
            e[:,0] = binary_edges[:,0]
            e[:,1] = binary_edges[:,2]
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,e)))
                 if(d<0.1):
                  grds_xz[l,t] = 1 

            bb_edge3 = np.zeros((32,32,3))
            bb_edge3[:,:,0] = grds_xy
            bb_edge3[:,:,1] = grds_yz
            bb_edge3[:,:,2] = grds_xz
            '''
            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(edges[0:1024,0], edges[0:1024,1], edges[0:1024,2],color='yellow')
            plt.show()
            
            if(total_count[int(name)]==0):
              fig = plt.figure()
              ax = fig.add_subplot(111, projection='3d')
              ax.scatter(point[:,0], point[:,1],point[:,2],color='red')
              ax.scatter(edges[0:1024,0], edges[0:1024,1],edges[0:1024,2],color='yellow')
              fig1 = plt.figure()
              ax1 = fig1.add_subplot(111)
              ax1.scatter(edges[0:1024,0], edges[0:1024,1],color='red')
              fig2 = plt.figure()
              ax2 = fig2.add_subplot(111)
              ax2.scatter(edges[0:1024,1], edges[0:1024,2],color='red')
              fig3 = plt.figure()
              ax3 = fig3.add_subplot(111)
              ax3.scatter(edges[0:1024,0], edges[0:1024,2],color='red')
              print(name)
              ax.set_xlabel('x-axis')
              ax.set_ylabel('y-axis')
              ax1.set_xlabel('x-axis')
              ax1.set_ylabel('y-axis')
              plt.show()
              total_count[int(name)] += 1
            '''
            #Load pose data and anchor label from dictionary
            bb_path = "/home/mariam/pointnet2/data/ModelNet40/gt_"+self.datapath[index][0]+"_one.txt"
            [bb_param,bb_label] = find_index(bb_path,self.datapath[index][0],self.datapath[index][1],self.datapath[index][2])
            bb_param[6] += rand_head

         else:
            #Load class
            name = self.classes1[self.datapath[index][0]]
            name = np.array([name]).astype(np.int32)
            #Load points from file
            fn = self.datapath[index]
            point = np.loadtxt(fn[3],delimiter=',').astype(np.float32)
            data = np.zeros((1024,4), dtype=np.float32)
            size = point[:,0].size       
            if(size>1024):
              ind = np.random.choice(point.shape[0], 1024, replace=False)
              data[0:1024,0:3] = point[ind[:],0:3]   
            else:
              less = 1024-size
              data[0:size,0:3] = point[:,0:3]
              ind = np.random.choice(point.shape[0], less, replace=True)
              data[size:1024,0:3] = point[ind[:],0:3]         
            point = data[:,0:3]
            if(name == 18):
              x_mat = rotx(-0.57)
              data[:,3] = 1 
              pcd = np.transpose(np.matmul(x_mat,np.transpose(data)))
              point = pcd[:,0:3]
            else:
              x_mat = rotx(1.57)
              data[:,3] = 1 
              pcd = np.transpose(np.matmul(x_mat,np.transpose(data)))
              point = pcd[:,0:3] 
            
            #Find Edge points
            nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(point)
            distances, indices = nbrs.kneighbors(point)
            centroids = np.mean(point[indices[:,:],0:3],axis=1)
            dist = np.sqrt(np.sum((point[:,0:3]-centroids)**2,axis=1))
            edges = np.zeros((len(point),3), dtype=np.float32)
            binary_edges = np.zeros((len(point),4), dtype=np.float32)
            edges[:,0] = 1*(dist[:]>0.035)
            binary_edges[:,0:3] = point[:,:]
            binary_edges[:,3] = edges[:,0]
            #point = binary_edges
            ind = np.where(edges[:,0]==1)[0]
            edges[ind,0:3] = point[ind,0:3]
            bb_edge = binary_edges[:,3]

            #Convert Edge to 2D maps
            x = np.arange(-1.0, 1.0, 0.0625)
            y = np.arange(-1.0, 1.0, 0.0625)
            #x, y = np.meshgrid(x, y)
            p = np.zeros((1, 2))
            grds_xy = np.zeros((32,32))
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,binary_edges[:,0:2])))
                 if(d<0.1):
                   grds_xy[l,t] = 1 

            grds_yz = np.zeros((32,32))
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,binary_edges[:,1:3])))
                 if(d<0.1):
                  grds_yz[l,t] = 1 

            grds_xz = np.zeros((32,32))
            e = np.zeros((1024,2))
            e[:,0] = binary_edges[:,0]
            e[:,1] = binary_edges[:,2]
            for l in range (0,32):
               for t in range (0,32):
                 p[0,0] = x[l]
                 p[0,1] = y[t] 
                 d = min(np.squeeze(cdist(p,e)))
                 if(d<0.1):
                  grds_xz[l,t] = 1 

            bb_edge3 = np.zeros((32,32,3))
            bb_edge3[:,:,0] = grds_xy
            bb_edge3[:,:,1] = grds_yz
            bb_edge3[:,:,2] = grds_xz
            '''
            imsave('edge_xy.png', grds_xy)
            imsave('edge_yz.png', grds_yz)
            imsave('edge_xz.png', grds_xz)
            
            
            if(total_count[int(name)]==0):
              fig = plt.figure()
              ax = fig.add_subplot(111, projection='3d')
              ax.scatter(point[:,0], point[:,1],point[:,2],color='red')
              ax.scatter(edges[0:1024,0], edges[0:1024,1],edges[0:1024,2],color='yellow')
              fig1 = plt.figure()
              ax1 = fig1.add_subplot(111)
              ax1.scatter(edges[0:1024,0], edges[0:1024,1],color='red')
              fig2 = plt.figure()
              ax2 = fig2.add_subplot(111)
              ax2.scatter(edges[0:1024,1], edges[0:1024,2],color='red')
              fig3 = plt.figure()
              ax3 = fig3.add_subplot(111)
              ax3.scatter(edges[0:1024,0], edges[0:1024,2],color='red')
              print(name)
              ax.set_xlabel('x-axis')
              ax.set_ylabel('y-axis')
              ax1.set_xlabel('x-axis')
              ax1.set_ylabel('y-axis')
              plt.show()
              total_count[int(name)] += 1
            '''
            #Padding the additional pose parameters
            bb_param = [0,0,0,0,0,0,0]
            bb_label = 0


         if len(self.cache) < self.cache_size:
                self.cache[index] = (point,name,bb_param,bb_label,bb_edge3)
        return point, name,bb_param,bb_label,bb_edge3
        
    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 6

    def reset(self):
        self.idxs = np.arange(0, len(self.datapath))
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.num_batches = (len(self.datapath)+self.batch_size-1) // self.batch_size
        self.batch_idx = 0

    def has_next_batch(self):
        return self.batch_idx < self.num_batches

    def next_batch(self, augment=False,data='train'):#modify this 
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, len(self.datapath))
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.npoints, 3))
        batch_edge = np.zeros((bsize, 32,32,3))
        batch_pose = np.zeros((bsize,7), dtype=np.float32)
        batch_center = np.zeros((bsize,3), dtype=np.float32)
        batch_id = np.empty([bsize,2],dtype="<U12")
        batch_class = np.zeros((bsize), dtype=np.int32)
        batch_anchor = np.zeros((bsize), dtype=np.int32)
        for i in range(bsize):
            ps,name,bb,anchor,edge = self._get_item(self.idxs[i+start_idx],data)#cls,cen,ids,
            batch_data[i] = ps
            batch_pose[i] = bb
            #batch_center[i] = cen
            #batch_id[i] = ids
            batch_class[i] = name
            batch_anchor[i] = anchor
            batch_edge[i,...] = edge
        self.batch_idx += 1
        if augment: batch_data,batch_pose = self._augment_batch_data_pose(batch_data,batch_pose)
        return batch_data,batch_class,batch_pose,batch_anchor,batch_edge
    
if __name__ == '__main__':
    d = ModelNetDataset(root = '../data/modelnet10_normal_resampled', split='test')
    print(d.shuffle)
    print(len(d))
    import time
    tic = time.time()
    for i in range(10):
        ps, cls = d[i]
    print(time.time() - tic)
    print(ps.shape, type(ps), cls)

    print(d.has_next_batch())
    ps_batch, cls_batch, cen_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
