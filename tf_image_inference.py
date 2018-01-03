#!/usr/bin/python

import numpy as np
import os
import csv
import tensorflow as tf
import time
from PIL import Image
from skimage import io
from skimage import transform
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#import matplotlib.pyplot as plt


def drawBBox(img, pred_coordinates, ground_truth_coordinates):
  xmin = int(clipWidth(pred_coordinates[0])) - 1
  xmax = int(clipWidth(pred_coordinates[1])) - 1
  ymin = int(clipHeight(pred_coordinates[2])) - 1
  ymax = int(clipHeight(pred_coordinates[3])) - 1

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 255
    img[ymin][xmin+i][1] = 0
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 255
    img[ymin+i][xmin][1] = 0
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 255
    img[ymin+i][xmax][1] = 0
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 255
    img[ymax][xmin+i][1] = 0
    img[ymax][xmin+i][2] = 0

  xmin = int(ground_truth_coordinates[0]) - 1
  xmax = int(ground_truth_coordinates[1]) - 1
  ymin = int(ground_truth_coordinates[2]) - 1
  ymax = int(ground_truth_coordinates[3]) - 1

  #print "image shape: ", img.shape
  #print "coordinate: ", coordinates

  for i in range(0,xmax-xmin):
    img[ymin][xmin+i][0] = 0
    img[ymin][xmin+i][1] = 128
    img[ymin][xmin+i][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmin][0] = 0
    img[ymin+i][xmin][1] = 128
    img[ymin+i][xmin][2] = 0

  for i in range(0,ymax-ymin):
    img[ymin+i][xmax][0] = 0
    img[ymin+i][xmax][1] = 128
    img[ymin+i][xmax][2] = 0

  for i in range(0,xmax-xmin):
    img[ymax][xmin+i][0] = 0
    img[ymax][xmin+i][1] = 128
    img[ymax][xmin+i][2] = 0


  return img


def limitWithinOne(val):
  if val > 1:
    return 1

  if val < 0:
    return 0

  return val

def clipWidth(val):
  if val > 640:
    return 640

  if val < 1:
    return 1

  return val

def clipHeight(val):
  if val > 360:
    return 360

  if val < 1:
    return 1

  return val


def checkIOU(label_BBox, pred_BBox):
  #print "label_BBox shape: ", label_BBox.shape
  #print "pred_BBox shape: ", pred_BBox.shape

  IOU = np.zeros(label_BBox.shape[0])

  for i in range(label_BBox.shape[0]):
    ###############################
    #  check validity of pred box #
    ###############################
    if pred_BBox[i][0] >= pred_BBox[i][1] or pred_BBox[i][2] >= pred_BBox[i][3]:
      IOU[i] = 0
    else:
      if checkIntersection(label_BBox[i], pred_BBox[i]) == 1:

        #xmin_A = limitWithinOne(pred_BBox[i][0])
        #xmax_A = limitWithinOne(pred_BBox[i][1])
        #ymin_A = limitWithinOne(pred_BBox[i][2])
        #ymax_A = limitWithinOne(pred_BBox[i][3])

        xmin_A = clipWidth(pred_BBox[i][0])
        xmax_A = clipWidth(pred_BBox[i][1])
        ymin_A = clipHeight(pred_BBox[i][2])
        ymax_A = clipHeight(pred_BBox[i][3])


        xmin_B = label_BBox[i][0]
        xmax_B = label_BBox[i][1]
        ymin_B = label_BBox[i][2]
        ymax_B = label_BBox[i][3]

        #print "pred BBox[0]: ", xmin_A
        #print "pred BBox[1]: ", xmax_A
        #print "pred BBox[2]: ", ymin_A
        #print "pred BBox[3]: ", ymax_A
        #print "width A: ", (xmax_A-xmin_A)
        #print "height A: ", (ymax_A-ymin_A)

        #print "label BBox[0]: ", xmin_B
        #print "label BBox[1]: ", xmax_B
        #print "label BBox[2]: ", ymin_B
        #print "label BBox[3]: ", ymax_B
        #print "width B: ", (xmax_B-xmin_B)
        #print "height B: ", (ymax_B-ymin_B)


        xmin_intersection = np.maximum(xmin_A, xmin_B)
        xmax_intersection = np.minimum(xmax_A, xmax_B)
        ymin_intersection = np.maximum(ymin_A, ymin_B)
        ymax_intersection = np.minimum(ymax_A, ymax_B)

        intersection_area = (xmax_intersection-xmin_intersection)*(ymax_intersection-ymin_intersection)
        area_two_boxes = (xmax_A-xmin_A)*(ymax_A-ymin_A) + (xmax_B-xmin_B)*(ymax_B-ymin_B)
        IOU[i] = intersection_area/(area_two_boxes-intersection_area) 

        #print "intersection_area: ", intersection_area
        #print "area_two_boxes: ", area_two_boxes
        #print "IOU[%d]: %f" % (i, IOU[i])
      else:
        IOU[i] = 0
      

  #print "xmin_union: ", xmin_union
  #print "ymin_union: ", ymin_union
  #print "xmax_union: ", xmax_union
  #print "ymax_union: ", ymax_union


    
  #print IOU
  return IOU

def checkIntersection(BBoxA, BBoxB):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################

  xmin = BBoxB[0] 
  xmax = BBoxB[1]
  ymin = BBoxB[2]
  ymax = BBoxB[3]


  #########################################
  #     BBox intersects the grid cells    #
  #########################################
  target_x = BBoxA[0] #xmin
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[0] #xmin
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[3] #ymax
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1

  target_x = BBoxA[1] #xmax
  target_y = BBoxA[2] #ymin
  if target_x >= xmin and target_x <= xmax and target_y >= ymin and target_y <= ymax:
    return 1
 
 
  #########################################
  #       BBoxB is within in BBoxA        #
  #########################################
  if xmin >= BBoxA[0] and ymin >= BBoxA[2] and xmax <= BBoxA[1] and ymax <= BBoxA[3]:
    return 1


  return 0

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
  look_up_label_dict = {}
  for l in lines:
    elements = l.split()
    look_up_label_dict[int(elements[1])] = elements[0]


  #########################################
  #  Configuration of CNN architecture    #
  #########################################
  mini_batch = 128

  K = 98 # number of classes
  G = 144 # number of grid cells
  P = 4  # four parameters of the bounding boxes
  NUM_FILTER_1 = 32
  NUM_FILTER_2 = 32
  NUM_FILTER_3 = 64 
  NUM_FILTER_4 = 64 
  NUM_FILTER_5 = 128
  NUM_FILTER_6 = 128

  NUM_NEURON_1 = 1024
  NUM_NEURON_2 = 1024

  DROPOUT_PROB = 1.0

  LEARNING_RATE = 1e-3
 

  # Dropout probability
  #keep_prob     = tf.placeholder(tf.float32)


  # initialize parameters randomly
  X      = tf.placeholder(tf.float32, shape=[None, 360,640,3])
  Y_     = tf.placeholder(tf.float32, shape=[None,K])
  #Y_GRID = tf.placeholder(tf.float32, shape=[None,G])
  Y_BBOX = tf.placeholder(tf.float32, shape=[None,P])


  W1  = tf.get_variable("W1", shape=[6,10,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())
  W2  = tf.get_variable("W2", shape=[3,3,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())
  W3  = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())
  W4  = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())
  W5  = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())
  W6  = tf.get_variable("W6", shape=[3,3,NUM_FILTER_5,NUM_FILTER_6], initializer=tf.contrib.layers.xavier_initializer())

  W9  = tf.get_variable("W9", shape=[23*27*NUM_FILTER_6,NUM_NEURON_1], initializer=tf.contrib.layers.xavier_initializer())
  W10 = tf.get_variable("W10", shape=[NUM_NEURON_1,NUM_NEURON_2], initializer=tf.contrib.layers.xavier_initializer())
  #W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer())
  W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K*G], initializer=tf.contrib.layers.xavier_initializer())



  b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
  b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
  b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
  b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
  b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
  b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')
  b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
  b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')
  #b11 = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b11')
  b11 = tf.Variable(tf.constant(0.1, shape=[K*G], dtype=tf.float32), trainable=True, name='b11')

  matrix_w = np.zeros((K*G,K))
  for i in range(0,K):
    for j in range(0,G):
      matrix_w[i*G+j][i] = 1

  label_pred_transform_W = tf.constant(matrix_w, shape=matrix_w.shape, dtype=tf.float32)


  W_bbox = tf.get_variable("W_bbox", shape=[K*G,P], initializer=tf.contrib.layers.xavier_initializer())
  b_bbox = tf.Variable(tf.constant(0.1, shape=[P], dtype=tf.float32), trainable=True, name='b_bbox')

  #===== architecture =====#
  conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,2,3,1], padding='VALID')+b1)
  conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


  conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME')+b3)
  conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
  pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME')+b5)
  conv6 = tf.nn.relu(tf.nn.conv2d(conv5, W6, strides=[1,1,1,1], padding='SAME')+b6)
  pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



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

  fc1 = tf.nn.relu(tf.matmul(YY,W9)+b9)
  #fc1_drop = tf.nn.dropout(fc1, keep_prob)

  fc2 = tf.nn.relu(tf.matmul(fc1,W10)+b10)
  #fc2_drop = tf.nn.dropout(fc2, keep_prob)

  Y = tf.matmul(fc2,W11)+b11
  Y_class = tf.matmul(Y,label_pred_transform_W)
  #Y_soft = tf.nn.softmax(Y_class)

  Y_bbox = tf.matmul(tf.nn.relu(Y),W_bbox)+b_bbox

  #total_preds  = tf.concat([Y_soft,Y_bbox],-1)
  #total_labels = tf.concat([Y_,Y_BBOX],-1)

  #mse_loss = tf.losses.mean_squared_error(labels=Y_GRID, predictions=Y, weights=1e-1)
  #mse_loss = tf.losses.mean_squared_error(labels=Y_GRID, predictions=Y, weights=1e-1*Y_GRID)
  #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
  #mse_loss = tf.losses.mean_squared_error(labels=Y_, predictions=Y_soft)

#  mse_weight = np.full(K+P,0.5)
#  for i in range(K, K+P):
#    mse_weight = 0.005
#
#  #mse_loss = tf.losses.mean_squared_error(labels=total_labels, predictions=total_preds, weights=mse_weight)
#  mse_loss = tf.losses.mean_squared_error(labels=Y_BBOX, predictions=Y_bbox)
#  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y_class))
#  total_loss = 1e-4*mse_loss+1e-3*cross_entropy
#  #mse_loss = tf.losses.mean_squared_error(labels=total_labels, predictions=total_preds)
#
#  global_step = tf.Variable(0, trainable=False)
#  lr = tf.train.exponential_decay(LEARNING_RATE, global_step,
#                                  100000, 0.1, staircase=True)
#  #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_nesterov=True).minimize(total_loss)
#  train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, global_step=global_step)
#  #train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy, global_step=global_step)
#  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


  correct_prediction = tf.equal(tf.argmax(Y_class, 1), tf.argmax(Y_, 1))
  #correct_prediction = tf.equal(tf.argmax(Y_class, 1), tf.argmax(Y_, 1))
  correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()

  #train_data_path = []
  #for name in label_dict.keys():
  #  print "/home/hhwu/tracking/data_training/train_%s.tfrecords" % name
  #  train_data_path.append("/home/hhwu/tracking/data_training/train_%s.tfrecords" % name)
  #  #train_data_path.append("/mnt/ramdisk/tf_data/train_%d.tfrecords" % i)

  #train_data_path = []
  #for i in range(1,8):
  #  train_data_path.append("/home/hhwu/tracking/crop_data/train_boat%d.tfrecords" % i)
  #train_data_path = "/home/hhwu/tracking/tf_data/train/train_person3.tfrecords"
  valid_data_path = "/home/hhwu/tracking/data_training/valid_all.tfrecords"
  #train_data_path = "/home/hhwu/tracking/crop_data/train_boat8.tfrecords"

  with tf.Session() as sess:
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

    #valid_image = tf.image.resize_images(valid_image, [640, 640])

    valid_label_box_coor = tf.stack([valid_label_xmin, valid_label_xmax, valid_label_ymin, valid_label_ymax])
    

    valid_images, valid_labels, vl_box_coors = tf.train.batch([valid_image, valid_label, valid_label_box_coor], 
                                                 batch_size=100, capacity=2000, num_threads=16)




    #sess.run(tf.global_variables_initializer())
    # Initialize all global and local variables
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    saver.restore(sess, "./checkpoint/model_small_trained.ckpt")
    print "Model %s restored." % ("model_small_trained")


    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #x, y = batchRead(class_name, mean_img, pool)



    print "Inference Starts..."
    valid_accuracy = 0.0
    valid_IOU = 0.0
    time_start=time.time()
    for i in range(0,100):
      test_x, test_y, box_coord = sess.run([valid_images, valid_labels, vl_box_coors])
      Y_labels_with_grid = expandLabel(test_y, box_coord, 100)

      pred_bbox = Y_bbox.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord})

   
      valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, Y_BBOX: box_coord})
      valid_IOU += np.mean(checkIOU(box_coord, pred_bbox))

      #for j in range(0, 100):
      #  bbox_image = drawBBox(test_x[j],pred_bbox[j], box_coord[j])
      #  #print "Image: ", bbox_image
      #  #print np.argmax(test_y[j])
      #  #print look_up_label_dict[np.argmax(test_y[j])]
      #  io.imsave("%s_%d_%s.%s" % ("./val_images/test_img", 100*i+j, look_up_label_dict[np.argmax(test_y[j])], 'jpg'), test_x[j]/256.0)

    time_end = time.time()
    resultRunTime = time_end-time_start
    print "Spent time: ", resultRunTime  
    print "FPS: ", 10000/resultRunTime  
    print "Validation Accuracy: %f (%.1f/10000)" %  (valid_accuracy/10000, valid_accuracy)
    print "Validation Mean IOU: %f (%.1f/100)" %  (valid_IOU/100, valid_IOU)



    coord.request_stop()
    coord.join(threads)
    sess.close()



