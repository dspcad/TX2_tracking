#!/usr/bin/python

import numpy as np
import csv
import tensorflow as tf
import time
from PIL import Image
from skimage import io
from skimage import transform
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#import matplotlib.pyplot as plt

def limitWithinOne(val):
  if val > 1:
    return 1

  if val < 0:
    return 0

  return val

def clipWidth(val):
  if val > 640:
    return 640

  if val < 0:
    return 0

  return val

def clipHeight(val):
  if val > 360:
    return 360

  if val < 0:
    return 0

  return val


def checkIOU(label_BBox, pred_BBox):
  IOU = np.zeros(label_BBox.shape[0])

  for i in range(label_BBox.shape[0]):
    ###############################
    #  check validity of pred box #
    #   (xmin, xmax, ymin, ymax)  #
    ###############################
 
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(label_BBox[i][0], pred_BBox[i][0])
    yA = max(label_BBox[i][2], pred_BBox[i][2])
    xB = min(label_BBox[i][1], pred_BBox[i][1])
    yB = min(label_BBox[i][3], pred_BBox[i][3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (label_BBox[i][1] - label_BBox[i][0] + 1) * (label_BBox[i][3] - label_BBox[i][2] + 1)
    boxBArea = (pred_BBox[i][1]  - pred_BBox[i][0] + 1) * (pred_BBox[i][3] - pred_BBox[i][2] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    IOU[i] = interArea / float(boxAArea + boxBArea - interArea)

  return IOU


def checkIntersectionGrid(x, y, BBox):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  xmin = BBox[0] 
  xmax = BBox[1]
  ymin = BBox[2]
  ymax = BBox[3]

  cell_xmin = x*80
  cell_xmax = cell_xmin + 80
  cell_ymin = y*60
  cell_ymax = cell_ymin + 80

  #########################################
  #     BBox intersects the grid cells    #
  #########################################
  target_x = cell_xmin
  target_y = cell_ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = cell_xmax
  target_y = cell_ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = cell_xmin
  target_y = cell_ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = cell_xmax
  target_y = cell_ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1
 
 
  #########################################
  #     BBox is within in a grid cell     #
  #########################################
  if xmin >= cell_xmin and ymin >= cell_ymin and xmax <= cell_xmax and ymax <= cell_ymax:
    return 1


  return 0

def expandLabel(Y_, BBox_, batch_size):
  #print "Y_ shape: ",  Y_.shape
  #print "BBox_ shape: ",  BBox_.shape
  Y_labels_with_grids = np.zeros((batch_size, G))
  #target_classes = np.argmax(Y_, axis=1)

  #print "target_classes shape: ",  target_classes.shape
  #print "Y_labels_with_grids shape: ",  Y_labels_with_grids.shape

  for idx in range(0, batch_size):
    for i in range(0, 8):
      for j in range(0, 6):
        #print "idx: ", idx
        #print "i: ", i
        #print "j: ", j
        #print target_classes
        Y_labels_with_grids[idx][i*6+j] = checkIntersectionGrid(i,j, BBox_[idx])
   

  return Y_labels_with_grids

if __name__ == '__main__':
  print '===== Start loading the labels of DAC Tracking datasets ====='
  infile = open("label.txt", "r")
  lines = infile.readlines()

  lines = map(lambda s: s.strip(), lines)
  label_dict = {}
  for l in lines:
    elements = l.split()
    label_dict[elements[0]] = elements[1]




  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128
  EPOCH = 676

  K = 98 # number of classes
  G = 512 # number of grid cells
  P = 4  # four parameters of the bounding boxes
  lamda = 0.001

  NUM_FILTER_1 = 32
  NUM_FILTER_2 = 32
  NUM_FILTER_3 = 64
  NUM_FILTER_4 = 64
  NUM_FILTER_5 = 128
  NUM_FILTER_6 = 128

  NUM_NEURON_1 = 1024
  NUM_NEURON_2 = 1024

  LEARNING_RATE = float(sys.argv[1])

  print 'Settings: '
  print '    Learning Rate: : ', LEARNING_RATE
 

  # Dropout probability
  lr          = tf.placeholder(tf.float32)
  is_training = tf.placeholder(tf.bool)



  # initialize parameters randomly
  X      = tf.placeholder(tf.float32, shape=[None, 360,640,3])
  Y_     = tf.placeholder(tf.float32, shape=[None,K])
  #Y_GRID = tf.placeholder(tf.float32, shape=[None,G])
  Y_BBOX = tf.placeholder(tf.float32, shape=[None,P])


  W1  = tf.get_variable("W1", shape=[6,10,3,NUM_FILTER_1], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W2  = tf.get_variable("W2", shape=[3,3,NUM_FILTER_1,NUM_FILTER_2], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W3  = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W4  = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W5  = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W6  = tf.get_variable("W6", shape=[3,3,NUM_FILTER_5,NUM_FILTER_6], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))

  W9  = tf.get_variable("W9", shape=[23*27*NUM_FILTER_6,NUM_NEURON_1], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W10 = tf.get_variable("W10", shape=[NUM_NEURON_1,NUM_NEURON_2], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K*G], 
                        initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))



#  b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
#  b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
#  b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
#  b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
#  b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
#  b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')
  b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
  b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')
  #b11 = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b11')
  b11 = tf.Variable(tf.constant(0.1, shape=[K*G], dtype=tf.float32), trainable=True, name='b11')

  matrix_w = np.zeros((K*G,K))
  for i in range(0,K):
    for j in range(0,G):
      matrix_w[i*G+j][i] = 0.1

  label_pred_transform_W = tf.constant(matrix_w, shape=matrix_w.shape, dtype=tf.float32)
  tf.stop_gradient(label_pred_transform_W)
  #label_pred_transform_W = tf.get_variable("W_class", shape=[K*G,K], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_class = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b_class')


  matrix_wb = np.zeros((K*G,K*P))
  #matrix_w = np.full((K*G,K), 0.001)
  for i in range(0,K):
    for j in range(0,G):
      matrix_wb[i*G+j][i*P] = 0.1
      matrix_wb[i*G+j][i*P+1] = 0.1
      matrix_wb[i*G+j][i*P+2] = 0.1
      matrix_wb[i*G+j][i*P+3] = 0.1

  W_bbox_1  = tf.constant(matrix_wb, shape=matrix_wb.shape, dtype=tf.float32)
  tf.stop_gradient(W_bbox_1)
  #W_bbox_1 = tf.get_variable("W_bbox_1", shape=[K*G,K*P], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_bbox_1 = tf.Variable(tf.constant(0.1, shape=[K*P], dtype=tf.float32), trainable=True, name='b_bbox_1')


  #W_bbox_1 = tf.get_variable("W_bbox_1", shape=[K*G,K*P], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  #b_bbox_1 = tf.Variable(tf.constant(0.1, shape=[K*P], dtype=tf.float32), trainable=True, name='b_bbox_1')

  W_bbox_2 = tf.get_variable("W_bbox_2", shape=[K*P,P], initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(lamda))
  b_bbox_2 = tf.Variable(tf.constant(0.1, shape=[P], dtype=tf.float32), trainable=True, name='b_bbox_2')





  #===== architecture =====#
  conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,2,3,1], padding='VALID'))
  conv2 = tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')
  norm1 = tf.nn.relu(tf.layers.batch_normalization(conv2, training=is_training, renorm=True))
  pool1 = tf.nn.max_pool(norm1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
  conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME'))
  conv4 = tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')
  norm2 = tf.nn.relu(tf.layers.batch_normalization(conv4, training=is_training, renorm=True))
  pool2 = tf.nn.max_pool(norm2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
  conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME'))
  conv6 = tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')
  norm3 = tf.nn.relu(tf.layers.batch_normalization(conv6, training=is_training, renorm=True))
  pool3 = tf.nn.max_pool(norm3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')





#  conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,2,3,1], padding='VALID')+b1)
#  conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')+b2)
#  pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
#
#  conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME')+b3)
#  conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
#  pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#
#  conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME')+b5)
#  conv6 = tf.nn.relu(tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')+b6)
#  pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



  print "conv1: ", conv1.get_shape()
  print "conv2: ", conv2.get_shape()
  print "pool1: ", pool1.get_shape()

  print "conv3: ", conv3.get_shape()
  print "conv4: ", conv4.get_shape()
  print "pool2: ", pool2.get_shape()

  print "conv5: ", conv5.get_shape()
  print "conv6: ", conv6.get_shape()
  print "pool3: ", pool3.get_shape()


  YY = tf.reshape(pool3, shape=[-1,23*27*NUM_FILTER_6])
  
  #fc1 = tf.matmul(YY,W9)+b9
  fc1 = tf.nn.relu(tf.matmul(YY,W9)+b9)
  #fc1_drop = tf.nn.dropout(fc1, keep_prob)
  
  fc2 = tf.matmul(fc1,W10)+b10
  fc2_norm = tf.nn.relu(tf.layers.batch_normalization(fc2, training=is_training, renorm=True))
  #fc2_drop = tf.nn.dropout(fc2, keep_prob)
  
  Y = tf.matmul(fc2_norm,W11)+b11
  
  Y_class  = tf.matmul(Y,label_pred_transform_W)+b_class
  Y_bbox_1 = tf.matmul(Y,W_bbox_1)+b_bbox_1
  Y_bbox   = tf.matmul(Y_bbox_1,W_bbox_2)+b_bbox_2


  #total_preds  = tf.concat([Y_soft,Y_bbox],-1)
  #total_labels = tf.concat([Y_,Y_BBOX],-1)

  #mse_loss = tf.losses.mean_squared_error(labels=Y_GRID, predictions=Y, weights=1e-1)
  #mse_loss = tf.losses.mean_squared_error(labels=Y_GRID, predictions=Y, weights=1e-1*Y_GRID)
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
  #mse_loss = tf.losses.mean_squared_error(labels=Y_, predictions=Y_soft)

  #mse_weight = np.full(K+P,0.5)
  #for i in range(K, K+P):
  #  mse_weight = 0.01

  

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    mse_loss = tf.losses.mean_squared_error(labels=Y_BBOX, predictions=Y_bbox, weights=1e-3)
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=Y_, logits=Y_class, weights=1e-2))
    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_class))
    reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    total_loss = mse_loss + cross_entropy + reg_loss
    #total_loss = mse_loss + reg_loss
    #total_loss = 1e-4*mse_loss + 1e-3*cross_entropy
    #mse_loss = tf.losses.mean_squared_error(labels=total_labels, predictions=total_preds)

    #global_step = tf.Variable(0, trainable=False)
    #lr = tf.train.exponential_decay(LEARNING_RATE, global_step,
    #                                250000, 0.1, staircase=True)
    #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_nesterov=True).minimize(total_loss)
    train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(total_loss)
    #train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy, global_step=global_step)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


  correct_prediction = tf.equal(tf.argmax(Y_class, 1), tf.argmax(Y_, 1))
  #correct_prediction = tf.equal(tf.argmax(Y_class, 1), tf.argmax(Y_, 1))
  correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver(var_list=tf.global_variables())

  #train_data_path = []
  #for name in label_dict.keys():
  #  print "/home/hhwu/tracking/data_training/train_%s.tfrecords" % name
  #  train_data_path.append("/home/hhwu/tracking/data_training/train_%s.tfrecords" % name)
  #  #train_data_path.append("/mnt/ramdisk/tf_data/train_%d.tfrecords" % i)

  #train_data_path = []
  #for i in range(1,8):
  #  train_data_path.append("/home/hhwu/tracking/crop_data/train_boat%d.tfrecords" % i)
  #train_data_path = "/home/hhwu/tracking/tf_data/train/train_person3.tfrecords"
  train_data_path = "/home/hhwu/tracking/data_training/train_all.tfrecords"
  valid_data_path = "/home/hhwu/tracking/data_training/valid_all.tfrecords"
  #train_data_path = "/home/hhwu/tracking/crop_data/train_boat8.tfrecords"


  mean_img = np.load("/home/hhwu/tracking/data_training/mean.npy")
  #print mean_img
  #channel_mean = np.mean(mean_img, axis=(0,1))
  #print channel_mean

  with tf.Session() as sess:
    ################################
    #        Training Data         #
    ################################
    train_feature = {'train/image': tf.FixedLenFeature([], tf.string),
                     'train/xmin' : tf.FixedLenFeature([], tf.int64),
                     'train/xmax' : tf.FixedLenFeature([], tf.int64),
                     'train/ymin' : tf.FixedLenFeature([], tf.int64),
                     'train/ymax' : tf.FixedLenFeature([], tf.int64),
                     'train/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    #train_filename_queue = tf.train.string_input_producer(train_data_path, shuffle=True)
    train_filename_queue = tf.train.string_input_producer([train_data_path], shuffle=True, capacity=20*mini_batch)
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    train_reader = tf.TFRecordReader()
    _, train_serialized_example = train_reader.read(train_filename_queue)

        # Decode the record read by the reader
    train_features = tf.parse_single_example(train_serialized_example, features=train_feature)
    # Convert the image data from string back to the numbers
    #train_image = tf.decode_raw(train_features['train/image'], tf.uint8)
    train_image = tf.cast(tf.decode_raw(train_features['train/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    train_label_idx = tf.cast(train_features['train/label'], tf.int32)
    train_label = tf.one_hot(train_label_idx, K)

    train_label_xmin = tf.cast(train_features['train/xmin'], tf.float32)
    train_label_xmax = tf.cast(train_features['train/xmax'], tf.float32)
    train_label_ymin = tf.cast(train_features['train/ymin'], tf.float32)
    train_label_ymax = tf.cast(train_features['train/ymax'], tf.float32)



    # Reshape image data into the original shape
    train_image = tf.reshape(train_image, [360, 640, 3])
    #train_image = tf.subtract(train_image,channel_mean)
    train_image = tf.subtract(train_image,mean_img)
    # train_image = tf.image.per_image_standardization(train_image)

    #sel = np.random.uniform(0,1)
    #if(sel <= 0.5):
    #  train_image = tf.image.flip_left_right(train_image)
    #  train_label_box_coor = tf.stack([639-train_label_xmax, 639-train_label_xmin, train_label_ymin, train_label_ymax])
    #else:
    #  train_label_box_coor = tf.stack([train_label_xmin, train_label_xmax, train_label_ymin, train_label_ymax])

    train_label_box_coor = tf.stack([train_label_xmin, train_label_xmax, train_label_ymin, train_label_ymax])



    #train_image = tf.image.resize_images(train_image, [640, 640])
    #train_image = tf.image.random_flip_left_right(train_image)

    train_label_box_coor = tf.stack([train_label_xmin, train_label_xmax, train_label_ymin, train_label_ymax])

    #print "TFRecord: hhwu !"
    train_images, train_labels, tr_box_coors = tf.train.batch([train_image, train_label, train_label_box_coor], 
                                                 batch_size=mini_batch, capacity=20*mini_batch, num_threads=16)
    #train_images, train_labels, tr_box_coors = tf.train.shuffle_batch([train_image, train_label, train_label_box_coor], 
    #                                             batch_size=mini_batch, capacity=20*mini_batch, num_threads=16, min_after_dequeue=1000)
    #train_images, train_labels = tf.train.batch([train_image, train_label], 
    #                                             batch_size=mini_batch, capacity=20*mini_batch, num_threads=16)


    ################################
    #       Validation Data        #
    ################################
    valid_feature = {'valid/image': tf.FixedLenFeature([], tf.string),
                     'valid/xmin' : tf.FixedLenFeature([], tf.int64),
                     'valid/xmax' : tf.FixedLenFeature([], tf.int64),
                     'valid/ymin' : tf.FixedLenFeature([], tf.int64),
                     'valid/ymax' : tf.FixedLenFeature([], tf.int64),
                     'valid/label': tf.FixedLenFeature([], tf.int64)}
    # Create a list of filenames and pass it to a queue
    valid_filename_queue = tf.train.string_input_producer([valid_data_path])
    #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    # Define a reader and read the next record
    valid_reader = tf.TFRecordReader()
    _, valid_serialized_example = valid_reader.read(valid_filename_queue)

        # Decode the record read by the reader
    valid_features = tf.parse_single_example(valid_serialized_example, features=valid_feature)
    # Convert the image data from string back to the numbers
    valid_image = tf.cast(tf.decode_raw(valid_features['valid/image'], tf.uint8), tf.float32)
    
    # Cast label data into int32
    valid_label_idx = tf.cast(valid_features['valid/label'], tf.int32)
    valid_label = tf.one_hot(valid_label_idx, K)
    valid_label_xmin = tf.cast(valid_features['valid/xmin'], tf.float32)
    valid_label_xmax = tf.cast(valid_features['valid/xmax'], tf.float32)
    valid_label_ymin = tf.cast(valid_features['valid/ymin'], tf.float32)
    valid_label_ymax = tf.cast(valid_features['valid/ymax'], tf.float32)


    # Reshape image data into the original shape
    valid_image = tf.reshape(valid_image, [360, 640, 3])
    #valid_image = tf.subtract(valid_image,channel_mean)
    valid_image = tf.subtract(valid_image,mean_img)
    # valid_image = tf.image.per_image_standardization(valid_image)

    #valid_image = tf.image.resize_images(valid_image, [640, 640])

    valid_label_box_coor = tf.stack([valid_label_xmin, valid_label_xmax, valid_label_ymin, valid_label_ymax])
    

    valid_images, valid_labels, vl_box_coors = tf.train.batch([valid_image, valid_label, valid_label_box_coor], 
                                                 batch_size=100, capacity=1000, num_threads=16)




    #sess.run(tf.global_variables_initializer())
    # Initialize all global and local variables
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())


    ###############################
    # Restore variables from disk #
    ###############################
    #model_name = "model_small_59.30_119000"
    #saver.restore(sess, "./checkpoint/%s.ckpt" % model_name)
    #print "Model %s restored." % model_name


    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #x, y = batchRead(class_name, mean_img, pool)


    highest_IOU = 0
    epoch_num = 0
    for itr in xrange(70000):
      #x, y = batchRead(image_name, class_dict, mean_img, pool)

      #print y
      #asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7, asyn_train_y = setAsynBatchRead(class_name, pool, mean_img)
      #start_time = time.time()

      #x, y = sess.run([train_image, train_label])
      x, y, box_coord = sess.run([train_images, train_labels, tr_box_coors])
      #Y_labels_with_grid = expandLabel(y, box_coord, mini_batch)

      #box_coord[:,0] = box_coord[:,0]
      #box_coord[:,1] = box_coord[:,1]
      #box_coord[:,2] = box_coord[:,2]
      #box_coord[:,3] = box_coord[:,3]

      #print "box_coord[0]: ", box_coord[:,0]
      #print "box_coord[1]: ", box_coord[:,1]
      #print "box_coord[2]: ", box_coord[:,2]
      #print "box_coord[3]: ", box_coord[:,3]
 
      #print "Y_labels_with_grid: ", Y_labels_with_grid

      #for i in range(0, mini_batch):
      #  io.imsave("%s_%d.%s" % ("test_img", i, 'jpeg'), x[i])
      
      #train_step.run(feed_dict={X: x, Y_: lump_y, keep_prob: DROPOUT_PROB})
      sess.run([train_step,update_ops],feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: True})
      #train_step.run(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, keep_prob: DROPOUT_PROB, is_training: True})
      #elapsed_time = time.time() - start_time
      #print "Time for training: %f" % elapsed_time
      if itr % 20 == 0:
        pred_bbox = Y_bbox.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False})
        #print "pred_bbox: ", pred_bbox
        #tt = np.mean(checkIOU(box_coord, pred_bbox))
        #print tt
        #print tt.shape
        print "Iter %d:  learning rate: %f  cross entropy: %f  mse: %f  reg: %f  accuracy: %f  mean IOU: %f" % (itr,
                                                                LEARNING_RATE,
                                                                #lr.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord}),
                                                                cross_entropy.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False}),
                                                                mse_loss.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False}),
                                                                #reg_loss,
                                                                reg_loss.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False}),
                                                                accuracy.eval(feed_dict={X: x, Y_: y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False}),
                                                                np.mean(checkIOU(box_coord, pred_bbox)))

      #if itr % 20 == 0 and itr != 0:
      if itr % EPOCH == 0 and itr != 0:
        epoch_num = epoch_num + 1
        print "Epoch ", epoch_num

        valid_accuracy = 0.0
        valid_IOU = 0.0
        for i in range(0,100):
          test_x, test_y, box_coord = sess.run([valid_images, valid_labels, vl_box_coors])
          #Y_labels_with_grid = expandLabel(test_y, box_coord, 100)
          #box_coord[:,0] = box_coord[:,0]
          #box_coord[:,1] = box_coord[:,1]
          #box_coord[:,2] = box_coord[:,2]
          #box_coord[:,3] = box_coord[:,3]

          pred_bbox = Y_bbox.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False})

   
          valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord, lr: LEARNING_RATE, is_training: False})
          valid_IOU += np.mean(checkIOU(box_coord, pred_bbox))
        print "Validation Accuracy: %f (%.1f/10000)" %  (valid_accuracy/10000, valid_accuracy)
        print "Validation Mean IOU: %f (%.1f/100)" %  (valid_IOU/100, valid_IOU)


        f_train = open("history_train.txt","a") 
        f_train.write("Validation Accuracy: %f (%.1f/10000) " %  (valid_accuracy/10000, valid_accuracy))
        f_train.write("Validation Mean IOU: %f (%.1f/100)\n" %  (valid_IOU/100, valid_IOU)) 
        f_train.close() 

        #valid_result.write("Validation Accuracy: %f" % (valid_accuracy/20000))
        #valid_result.write("\n")
        if valid_IOU > highest_IOU:
          highest_IOU = valid_IOU
          model_name = "./checkpoint/model_small_%.2f_%d.ckpt" % (valid_IOU, itr)
          save_path = saver.save(sess, model_name)
          print("Model saved in file: %s" % save_path)

       

      if epoch_num == 20:
        LEARNING_RATE = 1e-4


      if epoch_num == 60:
        LEARNING_RATE = 1e-6
      #  model_name = "./checkpoint/model_small_%d.ckpt" % itr
      #  save_path = saver.save(sess, model_name)
      #  #save_path = saver.save(sess, "./checkpoint/model.ckpt")
      #  print("Model saved in file: %s" % save_path)




    #model_name = "./checkpoint/model_small_trained.ckpt"
    #save_path = saver.save(sess, model_name)
    ##save_path = saver.save(sess, "./checkpoint/model.ckpt")
    #print("Model saved in file: %s" % save_path)


    coord.request_stop()
    coord.join(threads)
    sess.close()



