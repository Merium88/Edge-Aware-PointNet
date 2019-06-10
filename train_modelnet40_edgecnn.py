'''
   Read partial view txt files of 3D models, remove files<1050 points
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_grouping import query_ball_point, group_point, knn_point
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from plot_cube import plot_cube
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from plot_cube import plot_cube
from eul2rot import euler2rotm
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
import provider
import tf_util
import modelnet40_dataset_orig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from sklearn.metrics import confusion_matrix
epsilon = sys.float_info.epsilon

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_edge', help='Model name [default: pointnet2_cls_edge]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=5, help='Epoch to run [default: 251]')
parser.add_argument('--model_path', default='log/model_orig_edge_cnn_rgbd.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
FLAGS = parser.parse_args()

NUM_CLASSES = 40
NUM_POSE = 7
NUM_ANCHOR = 4
NUM_CHANNELS = 3

EPOCH_CNT = 0
prec = np.zeros((30,2), dtype=np.float32)
rec = np.zeros((30,2), dtype=np.float32)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
MODEL_PATH = FLAGS.model_path
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


# Shapenet official train/test split

DATA_PATH = os.path.join('/home/mariam/pointnet2/data/ModelNet40/')
FLAGS.normal = False
TRAIN_DATASET = modelnet40_dataset_rgbd.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
TEST_DATASET = modelnet40_dataset_rgbd.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def nearest_neighbors(values, all_values, nbr_neighbors=1):
    nn = NearestNeighbors(nbr_neighbors, metric='euclidean', algorithm='brute').fit(all_values)
    dists, idxs = nn.kneighbors(values)
    return dists

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            
            pointclouds_pl, labels_pl_class = MODEL.placeholder_inputs_class(BATCH_SIZE, NUM_POINT)
            labels_pl_pose = MODEL.placeholder_inputs_pose(BATCH_SIZE, NUM_POINT)
            labels_pl_anchor = MODEL.placeholder_inputs_anchor(BATCH_SIZE, NUM_POINT)
            labels_pl_edge = MODEL.placeholder_inputs_edge_cnn(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            #tf.summary.scalar('bn_decay', bn_decay)
            
            # Get model and loss 
            pred_class, pred_pose, pred_anchor, end_points = MODEL.get_model_edge_cnn(pointclouds_pl, labels_pl_edge, is_training_pl, bn_decay=bn_decay)
            MODEL.get_loss_class(pred_class, labels_pl_class, end_points)
            losses_class = tf.get_collection('losses_class')
            total_loss_class = tf.add_n(losses_class, name='total_loss_class')
            tf.summary.scalar('total_loss_class', total_loss_class)
            for l in losses_class + [total_loss_class]:
                tf.summary.scalar(l.op.name, l)
            correct_class = tf.equal(tf.argmax(pred_class, 1), tf.to_int64(labels_pl_class))
            accuracy_class = tf.reduce_sum(tf.cast(correct_class, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy_class', accuracy_class)
           

            MODEL.get_loss_pose(pred_pose,labels_pl_pose, pointclouds_pl,BATCH_SIZE)
            losses_pose = tf.get_collection('losses_pose')
            total_loss_pose = tf.add_n(losses_pose, name='losses_pose')
            tf.summary.scalar('total_loss_pose', total_loss_pose)
            for l in losses_pose + [total_loss_pose]:
                tf.summary.scalar(l.op.name, l)
            accuracy = tf.reduce_sum(tf.cast(total_loss_pose, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy_pose', accuracy)

            MODEL.get_loss_anchor(pred_anchor, labels_pl_anchor)
            losses_class_anchor = tf.get_collection('losses_anchor')
            total_loss_anchor = tf.add_n(losses_class_anchor, name='total_loss_anchor')
            tf.summary.scalar('total_loss_anchor', total_loss_anchor)
            for l in losses_class_anchor + [total_loss_anchor]:
                tf.summary.scalar(l.op.name, l)
            correct_anchor = tf.equal(tf.argmax(pred_anchor, 1), tf.to_int64(labels_pl_anchor))
            accuracy_anchor = tf.reduce_sum(tf.cast(correct_anchor, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy_anchor', accuracy_anchor)
            
            print ("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            total_loss = 1.5*(total_loss_class) + (total_loss_pose + total_loss_anchor)
            train_op = optimizer.minimize(total_loss, global_step=batch)
           
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore trained graph
        #saver.restore(sess, MODEL_PATH)

        ops = {'pointclouds_pl': pointclouds_pl,
               'is_training_pl': is_training_pl,
               'labels_pl_class': labels_pl_class,
               'labels_pl_pose': labels_pl_pose,
               'labels_pl_anchor': labels_pl_anchor,
               'labels_pl_edge': labels_pl_edge,
               'pred_pose': pred_pose, 
               'pred_class': pred_class,
               'pred_anchor': pred_anchor,
               'loss_class': total_loss_class,
               'loss_pose': total_loss_pose,
               'loss_anchor': total_loss_anchor,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            #Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_orig_edge_cnn_sunrgbd.ckpt"))#rgbd_nonscaled/sum (for edge cnn on ModelNet40 orig)
                log_string("Model saved in file: %s" % save_path)
                    
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,NUM_CHANNELS))#TRAIN_DATASET.num_channel()
    cur_batch_label_class = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_pose = np.zeros((BATCH_SIZE,7), dtype=np.float32)
    cur_batch_label_anchor = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_edge = np.zeros((BATCH_SIZE,32,32,3))

    total_correct = 0
    total_correct_anchor = 0
    total_correct_edge = 0
    total_seen = 0
    total_seen_anchor = 0
    loss_sum_pose = 0
    loss_sum_class = 0
    loss_sum_anchor = 0
    loss_sum_edge = 0
    batch_idx = 0
    batch_class = 0
    batch_pose = 0
    bsize = 32
    correct_pose = 0
    total_pose = 0
    total_pose_add = 0
    total_seen_pose = 0
    total_pose_L1 = 0
    total_diff_pose = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data,batch_label_class,batch_label_pose,batch_label_anchor,batch_label_edge,batch_label_id = TRAIN_DATASET.next_batch(augment=False,data='train')
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data[:,:,0:NUM_CHANNELS]
        cur_batch_label_class[0:bsize] = batch_label_class
        cur_batch_label_pose[0:bsize,:] = batch_label_pose
        cur_batch_label_anchor[0:bsize] = batch_label_anchor
        cur_batch_label_edge[0:bsize,...] = batch_label_edge[:,:]
        #Run session for classification
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl_class']: cur_batch_label_class,
                     ops['labels_pl_pose']: cur_batch_label_pose,
                     ops['labels_pl_anchor']: cur_batch_label_anchor,
                     ops['labels_pl_edge']: cur_batch_label_edge,
                     ops['is_training_pl']: is_training,}
        

        summary_class, step_class, _, loss_val_class, pred_val_class,loss_val_pose,pred_val_pose,loss_val_anchor,pred_val_anchor = sess.run([ops['merged'], ops['step'],ops['train_op'], ops['loss_class'], ops['pred_class'], ops['loss_pose'],ops['pred_pose'], ops['loss_anchor'],ops['pred_anchor']], feed_dict=feed_dict)
        train_writer.add_summary(summary_class, step_class)
        scores = softmax(pred_val_class)
        pred_val_class = np.argmax(pred_val_class, 1)
        correct = np.sum(pred_val_class[0:bsize] == batch_label_class[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum_class += loss_val_class
            
        loss_sum_pose += loss_val_pose
        Groundtruth_pose = np.zeros((bsize,7), dtype=np.float32)
        Groundtruth_pose[:,0:3] = 0.01
        Groundtruth_pose[:,3:7] = 0.01
        diff = np.abs(batch_label_pose[0:bsize,:]-pred_val_pose[0:bsize,:])
        diff1 = np.mean(diff,axis=0)
        pose = (diff<Groundtruth_pose).astype(float)
        correct_pose = np.sum((np.sum(pose,1)/7)) 
        total_pose += correct_pose
        total_pose_L1 += correct_pose 
        total_seen_pose += bsize
        total_diff_pose += diff1

        scores_anchor = softmax(pred_val_anchor)
        pred_val_anchor = np.argmax(pred_val_anchor, 1)
        correct_anchor = np.sum(pred_val_anchor[0:bsize] == batch_label_anchor[0:bsize])
        total_correct_anchor += correct_anchor
        total_seen_anchor += bsize
        loss_sum_anchor += loss_val_anchor
        
        if ((batch_idx+1)%100 == 0) :
          log_string(' ---- batch: %03d ----' % (batch_idx+1))
          log_string('mean loss classification: %f' % (loss_sum_class / 500))
          log_string('accuracy classification: %f' % (total_correct / float(total_seen)))
          #log_string('execution time for 8 batch size: %f' %(float(duration)))
          log_string('accuracy regression L1: %f' %(float(total_pose_L1)/total_seen_pose))
          log_string('accuracy regression ADD: %f' %(float(total_pose)/500))
          log_string('loss for each parameter: %f %f %f \n %f %f %f \n %f' %(float(total_diff_pose[0]/500),float(total_diff_pose[1]/500),float(total_diff_pose[2]/500),float(total_diff_pose[3]/500),float(total_diff_pose[4]/500),float(total_diff_pose[5]/500),float(total_diff_pose[6]/500)))
          log_string('mean loss: %f' % (loss_sum_pose / 500))
          
          log_string('mean loss anchor classification: %f' % (loss_sum_anchor / 500))
          log_string('accuracy anchor classification: %f' % (total_correct_anchor / float(total_seen_anchor)))

          total_correct_anchor = 0
          
          total_seen_anchor = 0
          loss_sum_anchor = 0
          total_correct_edge = 0
          total_correct = 0
          total_seen = 0
          loss_sum_class = 0
          total_pose = 0
          loss_sum_pose = 0
          total_seen_pose = 0
          total_pose_L1 = 0
          total_diff_pose = 0

        batch_idx += 1
    
    TRAIN_DATASET.reset()

def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,NUM_CHANNELS))
    cur_batch_label_class = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_pose = np.zeros((BATCH_SIZE,7), dtype=np.float32)
    cur_batch_label_anchor = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_edge = np.zeros((BATCH_SIZE,32,32,3), dtype=np.int32)

    total_correct = 0
    total_correct_anchor = 0
    total_correct_edge = 0
    total_anchor = 0
    total_edge = 0
    total_seen = 0
    total_seen_anchor = 0
    loss_sum_pose = 0
    loss_sum_class = 0
    loss_sum_anchor = 0
    loss_sum_edge = 0
    batch_idx = 0
    batch_class = 0
    batch_pose = 0
    bsize = 32
    correct_pose = 0
    total_pose = 0
    total_pose_add = 0
    total_seen_pose = 0
    total_pose_L1 = 0
    total_diff_pose = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    total_seen_anchor = [0 for _ in range(NUM_ANCHOR)]
    total_correct_anchor = [0 for _ in range(NUM_ANCHOR)]
    total_seen_pose = 0
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    results_class = "/home/mariam/pointnet2/data/sunrgbd/prediction_class_edge_sunrgbd.txt"#rgbd_4views
    results_bbox = "/home/mariam/pointnet2/data/sunrgbd/prediction_bbox_edge_sunrgbd.txt"#rgbd_4views
    f_class = open(results_class,"a")
    f_bbox = open(results_bbox,"a")
    
    while TEST_DATASET.has_next_batch():

        batch_data,batch_label_class,batch_label_pose,batch_label_anchor,batch_label_edge,batch_label_id  = TEST_DATASET.next_batch(augment=False,data='test')

        # for the last batch in the epoch, the bsize:end are from last batch
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data[:,:,0:NUM_CHANNELS]
        cur_batch_label_class[0:bsize] = batch_label_class
        cur_batch_label_pose[0:bsize,:] = batch_label_pose
        cur_batch_label_anchor[0:bsize] = batch_label_anchor
        cur_batch_label_edge[0:bsize,...] = batch_label_edge[:,:,:]
        #Run session for classification
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl_class']: cur_batch_label_class,
                     ops['labels_pl_pose']: cur_batch_label_pose,
                     ops['labels_pl_anchor']: cur_batch_label_anchor,
                     ops['labels_pl_edge']: cur_batch_label_edge,
                     ops['is_training_pl']: is_training,}
        #start_time = time.time()
        summary_class, step_class, _, loss_val_class, pred_val_class,loss_val_pose,pred_val_pose,loss_val_anchor,pred_val_anchor = sess.run([ops['merged'], ops['step'],ops['train_op'], ops['loss_class'], ops['pred_class'], ops['loss_pose'],ops['pred_pose'], ops['loss_anchor'],ops['pred_anchor']], feed_dict=feed_dict)
        #duration = time.time() - start_time
        #print(duration)
        test_writer.add_summary(summary_class, step_class)
        scores = softmax(pred_val_class)
        pred_val_class = np.argmax(pred_val_class, 1)
        correct = np.sum(pred_val_class[0:bsize] == batch_label_class[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum_class += loss_val_class

        loss_sum_pose += loss_val_pose
        Groundtruth_pose = np.zeros((bsize,7), dtype=np.float32)
        Groundtruth_pose[:,0:3] = 0.01
        Groundtruth_pose[:,3:7] = 0.01
        diff = np.abs(batch_label_pose[0:bsize,:]-pred_val_pose[0:bsize,:])
        diff1 = np.mean(diff,axis=0)
        pose = (diff<Groundtruth_pose).astype(float)
        correct_pose = np.sum((np.sum(pose,1)/7)) 
        total_pose += correct_pose
        total_pose_L1 += correct_pose 
        total_seen_pose += bsize
        total_diff_pose += diff1

        scores_anchor = softmax(pred_val_anchor)
        pred_val_anchor = np.argmax(pred_val_anchor, 1)
        correct_anchor = np.sum(pred_val_anchor[0:bsize] == batch_label_anchor[0:bsize])
        total_anchor += correct_anchor
        loss_sum_anchor += loss_val_anchor

        for i in range(0, bsize):
            l = batch_label_class[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val_class[i] == l)
           
    
        if ((EPOCH_CNT+1)%1 == 0):
          
          for l in range (bsize):
             f_class.write("%i,%i,%i,%i,%i,%i,%i\n" %(EPOCH_CNT,batch_label_id[l,0],batch_label_id[l,1],pred_val_class[l], batch_label_class[l],pred_val_anchor[l], batch_label_anchor[l]))
             f_bbox.write("%i,%i,%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" %(EPOCH_CNT,batch_label_id[l,0],batch_label_id[l,1],pred_val_pose[l,0],pred_val_pose[l,1],pred_val_pose[l,2],pred_val_pose[l,3],pred_val_pose[l,4],pred_val_pose[l,5],pred_val_pose[l,6],batch_label_pose[l,0],batch_label_pose[l,1],batch_label_pose[l,2],batch_label_pose[l,3],batch_label_pose[l,4],batch_label_pose[l,5],batch_label_pose[l,6]))
              
        batch_idx += 1   
    f_class.close()
    f_bbox.close()
    
    log_string('batch id: %f' %(float(batch_idx+1)))
    log_string('loss for each parameter: %f %f %f \n %f %f %f \n %f' %(float(total_diff_pose[0]/float(batch_idx+1)),float(total_diff_pose[1]/float(batch_idx+1)),float(total_diff_pose[2]/float(batch_idx+1)),float(total_diff_pose[3]/float(batch_idx+1)),float(total_diff_pose[4]/float(batch_idx+1)),float(total_diff_pose[5]/float(batch_idx+1)),float(total_diff_pose[6]/float(batch_idx+1))))

    log_string('eval mean loss class: %f' % (loss_sum_class / float(batch_idx+1)))
    log_string('eval accuracy class: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    #log_string('eval mean loss anchor: %f' % (loss_sum_anchor / float(batch_idx+1)))
    #log_string('eval accuracy anchor: %f'% (total_anchor / float(total_seen)))
    #log_string('eval avg anchor acc: %f' % (np.mean(np.array(total_correct_anchor)/np.array(total_seen_anchor,dtype=np.float))))
   
    for k in range(0,NUM_CLASSES):
      if(total_seen_class[k]>0):
        log_string('class : %f '% (k))
        log_string('class accuracy: %f ' % (np.array(total_correct_class[k]/total_seen_class[k],dtype=np.float))) 
    #for k in range(0,NUM_ANCHOR):
    #  log_string('anchor class : %f '% (k))
    #  log_string('anchor accuracy: %f ' % (np.array(total_correct_anchor[k]/total_seen_anchor[k],dtype=np.float))) 
    print(EPOCH_CNT)
    EPOCH_CNT += 1

    TEST_DATASET.reset()

    #return total_correct/float(total_seen)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
