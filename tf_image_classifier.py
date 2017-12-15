#!/usr/bin/python

import numpy as np
import os
import csv
import tensorflow as tf
import time
from PIL import Image
from skimage import io
from skimage import transform

#import matplotlib.pyplot as plt

def cropImg(target_img):
  ###############################
  #       shape[0]: height      #
  #       shape[1]: width       #
  ###############################
  #mean_pixel = [123.182570556, 116.282672124, 103.462011796]


  #floating_img[:,:,0] = target_img[:,:,0] - mean_pixel[0]
  #floating_img[:,:,1] = target_img[:,:,1] - mean_pixel[1]
  #floating_img[:,:,2] = target_img[:,:,2] - mean_pixel[2]

  print target_img
  #floating_img = target_img - mean_img
  floating_img = np.empty(target_img.shape, dtype=np.float32)
  floating_img = target_img
  target_img = transform.resize(floating_img, (640,640,3))



  #floating_img = tf.image.random_flip_left_right(floating_img)
  #target_img = tf.random_crop(floating_img, [224, 224, 3])

  #target_img = tf.image.crop_to_bounding_box(floating_img, 14, 14, 227, 227)
  #target_img = target_img - mean_img

  #reflection   = np.random.randint(0,2)
  #if reflection == 0:
  #  floating_img = np.fliplr(floating_img)


  ################################
  ##      Data Augementation     #
  ################################
  #height_shift = np.random.randint(0,256-224)
  #width_shift  = np.random.randint(0,256-224)

  #height_shift = 16
  #width_shift  = 16
  ##target_img = floating_img[height_shift:height_shift+224, width_shift:width_shift+224,:]
  #target_img = floating_img[height_shift:height_shift+224, width_shift:width_shift+224,:]


  print "resize image into 640x640"
  return target_img



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

  K = 98 # number of classes
  NUM_FILTER_1 = 16
  NUM_FILTER_2 = 16
  NUM_FILTER_3 = 32
  NUM_FILTER_4 = 32
  NUM_FILTER_5 = 64
  NUM_FILTER_6 = 64
  NUM_FILTER_7 = 128
  NUM_FILTER_8 = 128

  NUM_NEURON_1 = 1024
  NUM_NEURON_2 = 1024

  DROPOUT_PROB = 0.50

  LEARNING_RATE = 1e-3
 

  # Dropout probability
  keep_prob     = tf.placeholder(tf.float32)


  # initialize parameters randomly
  X  = tf.placeholder(tf.float32, shape=[None, 320,320,3])
  Y_ = tf.placeholder(tf.float32, shape=[None,K])


  W1  = tf.get_variable("W1", shape=[3,3,3,NUM_FILTER_1], initializer=tf.contrib.layers.xavier_initializer())
  W2  = tf.get_variable("W2", shape=[3,3,NUM_FILTER_1,NUM_FILTER_2], initializer=tf.contrib.layers.xavier_initializer())
  W3  = tf.get_variable("W3", shape=[3,3,NUM_FILTER_2,NUM_FILTER_3], initializer=tf.contrib.layers.xavier_initializer())
  W4  = tf.get_variable("W4", shape=[3,3,NUM_FILTER_3,NUM_FILTER_4], initializer=tf.contrib.layers.xavier_initializer())
  W5  = tf.get_variable("W5", shape=[3,3,NUM_FILTER_4,NUM_FILTER_5], initializer=tf.contrib.layers.xavier_initializer())
  W6  = tf.get_variable("W6", shape=[3,3,NUM_FILTER_5,NUM_FILTER_6], initializer=tf.contrib.layers.xavier_initializer())
  W7  = tf.get_variable("W7", shape=[3,3,NUM_FILTER_6,NUM_FILTER_7], initializer=tf.contrib.layers.xavier_initializer())
  W8  = tf.get_variable("W8", shape=[3,3,NUM_FILTER_7,NUM_FILTER_8], initializer=tf.contrib.layers.xavier_initializer())

  W9  = tf.get_variable("W9", shape=[5*5*NUM_FILTER_8,NUM_NEURON_1], initializer=tf.contrib.layers.xavier_initializer())
  W10 = tf.get_variable("W10", shape=[NUM_NEURON_1,NUM_NEURON_2], initializer=tf.contrib.layers.xavier_initializer())
  W11 = tf.get_variable("W11", shape=[NUM_NEURON_2,K], initializer=tf.contrib.layers.xavier_initializer())



  b1  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_1], dtype=tf.float32), trainable=True, name='b1')
  b2  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_2], dtype=tf.float32), trainable=True, name='b2')
  b3  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_3], dtype=tf.float32), trainable=True, name='b3')
  b4  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_4], dtype=tf.float32), trainable=True, name='b4')
  b5  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_5], dtype=tf.float32), trainable=True, name='b5')
  b6  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_6], dtype=tf.float32), trainable=True, name='b6')
  b7  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_7], dtype=tf.float32), trainable=True, name='b7')
  b8  = tf.Variable(tf.constant(0.1, shape=[NUM_FILTER_8], dtype=tf.float32), trainable=True, name='b8')
  b9  = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_1], dtype=tf.float32), trainable=True, name='b9')
  b10 = tf.Variable(tf.constant(0.1, shape=[NUM_NEURON_2], dtype=tf.float32), trainable=True, name='b10')
  b11 = tf.Variable(tf.constant(0.1, shape=[K], dtype=tf.float32), trainable=True, name='b11')


  #===== architecture =====#
  conv1 = tf.nn.relu(tf.nn.conv2d(X,     W1, strides=[1,1,1,1], padding='SAME')+b1)
  conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1,1,1,1], padding='SAME')+b2)
  pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  conv3 = tf.nn.relu(tf.nn.conv2d(pool1, W3, strides=[1,1,1,1], padding='SAME')+b3)
  conv4 = tf.nn.relu(tf.nn.conv2d(conv3, W4, strides=[1,1,1,1], padding='SAME')+b4)
  pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  conv5 = tf.nn.relu(tf.nn.conv2d(pool2, W5, strides=[1,1,1,1], padding='SAME')+b5)
  pool3 = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  conv6 = tf.nn.relu(tf.nn.conv2d(pool3, W6, strides=[1,1,1,1], padding='SAME')+b6)
  pool4 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  conv7 = tf.nn.relu(tf.nn.conv2d(pool4, W7, strides=[1,1,1,1], padding='SAME')+b7)
  pool5 = tf.nn.max_pool(conv7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  conv8 = tf.nn.relu(tf.nn.conv2d(pool5, W8, strides=[1,1,1,1], padding='SAME')+b8)
  pool6 = tf.nn.max_pool(conv8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

  YY = tf.reshape(pool6, shape=[-1,5*5*NUM_FILTER_8])

  fc1 = tf.nn.relu(tf.matmul(YY,W9)+b9)
  fc1_drop = tf.nn.dropout(fc1, keep_prob)

  fc2 = tf.nn.relu(tf.matmul(fc1_drop,W10)+b10)
  fc2_drop = tf.nn.dropout(fc2, keep_prob)

  Y = tf.matmul(fc2_drop,W11)+b11



  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))

  global_step = tf.Variable(0, trainable=False)
  lr = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                  100000, 0.1, staircase=True)
  #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9, use_nesterov=True).minimize(total_loss)
  train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(cross_entropy, global_step=global_step)
  #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)


  correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
  correct_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
    train_filename_queue = tf.train.string_input_producer([train_data_path])
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
    # Reshape image data into the original shape
    train_image = tf.reshape(train_image, [360, 640, 3])

    train_image = tf.image.resize_images(train_image, [320, 320])

    #print "TFRecord: hhwu !"
    train_images, train_labels = tf.train.batch([train_image, train_label], 
                                                 batch_size=mini_batch, capacity=20*mini_batch, num_threads=16)


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
    # Reshape image data into the original shape
    valid_image = tf.reshape(valid_image, [360, 640, 3])

    valid_image = tf.image.resize_images(valid_image, [320, 320])
    
    valid_images, valid_labels = tf.train.batch([valid_image, valid_label], 
                                                 batch_size=100, capacity=2000, num_threads=16)




    #sess.run(tf.global_variables_initializer())
    # Initialize all global and local variables
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    #saver.restore(sess, "./checkpoint/model_200000.ckpt")
    #print("Model restored.")


    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #x, y = batchRead(class_name, mean_img, pool)



    image_iterator = 0
    data = []
    label = []
    for itr in xrange(1000):
      #x, y = batchRead(image_name, class_dict, mean_img, pool)

      #print y
      #asyn_0, asyn_1, asyn_2, asyn_3, asyn_4, asyn_5, asyn_6, asyn_7, asyn_train_y = setAsynBatchRead(class_name, pool, mean_img)
      #start_time = time.time()

      #x, y, image_iterator, data, label = batchSerialRead(image_iterator, data, label)
      x, y = sess.run([train_images, train_labels])
      #for i in range(0, mini_batch):
      #  io.imsave("%s_%d.%s" % ("test_img", i, 'jpeg'), x[i])

      train_step.run(feed_dict={X: x, Y_: y, keep_prob: DROPOUT_PROB})
      #elapsed_time = time.time() - start_time
      #print "Time for training: %f" % elapsed_time
      if itr % 20 == 0:
        print "Iter %d:  learning rate: %f  dropout: %.1f cross entropy: %f  accuracy: %f" % (itr,
                                                                #LEARNING_RATE,
                                                                lr.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}),
                                                                DROPOUT_PROB,
                                                                cross_entropy.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}),
                                                                accuracy.eval(feed_dict={X: x, Y_: y, keep_prob: 1.0}))

      if itr % 100 == 0:
        valid_accuracy = 0.0
        for i in range(0,200):
          test_x, test_y = sess.run([valid_images, valid_labels])
          valid_accuracy += correct_sum.eval(feed_dict={X: test_x, Y_: test_y, keep_prob: 1.0})
        print "Validation Accuracy: %f (%.1f/20000)" %  (valid_accuracy/20000, valid_accuracy)
        #valid_result.write("Validation Accuracy: %f" % (valid_accuracy/20000))
        #valid_result.write("\n")

       

      if itr % 10000 == 0 and itr != 0:
        model_name = "./checkpoint/model_%d.ckpt" % itr
        save_path = saver.save(sess, model_name)
        #save_path = saver.save(sess, "./checkpoint/model.ckpt")
        print("Model saved in file: %s" % save_path)




    coord.request_stop()
    coord.join(threads)
    sess.close()



