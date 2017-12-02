#!/usr/bin/python

import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import sys
import csv
import os
from skimage import io
from multiprocessing.pool import ThreadPool


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def xmlParser(f_path):
  tree = ET.parse(f_path)
  xmin = -1
  xmax = -1
  ymin = -1
  ymax = -1
  name = 'none'

  for elem in tree.iter():
    tag = elem.tag
    if tag == 'xmin':
      xmin = int(elem.text)
    elif tag == 'xmax':
      xmax = int(elem.text)
    elif tag == 'ymin':
      ymin = int(elem.text)
    elif tag == 'ymax':
      ymax = int(elem.text)
    elif tag == 'name':
      name = elem.text


  return name, xmin, xmax, ymin, ymax;


if __name__ == '__main__':
  datapath = '/home/hhwu/tensorflow_work/TX2_tracking/data_training/bird1'
  print "Path: ", datapath


  file_list = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    print "dirpath: ", dirpath
    print "dirnames: ", dirnames
    print "The number of files: %d" % len(filenames)
    
    file_list = filenames

  image_list = []
  xml_list = []

  for f in file_list:
    if f.endswith(".xml"):
      xml_list.append(f)
    elif f.endswith(".jpg"):
      image_list.append(f)

  image_list = sorted(image_list)
  xml_list = sorted(xml_list)
  assert len(image_list) == len(xml_list)

  for xml_elem in xml_list:
    f_path = "%s/%s" % (datapath,xml_elem)
    print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(f_path)

    print "label: ", label
    print "xmin: ", xmin
    print "xmax: ", xmax
    print "ymin: ", ymin
    print "ymax: ", ymax

#    feature = {'train/label': _int64_feature(label_list[j]),
#               'train/image': _bytes_feature(tf.compat.as_bytes(data_list[j].tostring()))}
#  image_counter = 0
#  data_dict = {}
#  for i in xrange(0,len(idx)):
#    if i % file_size == 0:
#      if i == 0:
#        data_list  = []
#        label_list = []
#      else:
#        output_name = "train_%d.tfrecords" % image_counter
#        writer = tf.python_io.TFRecordWriter(output_name)
#
#        for j in xrange(0, len(label_list)):
#          feature = {'train/label': _int64_feature(label_list[j]),
#                     'train/image': _bytes_feature(tf.compat.as_bytes(data_list[j].tostring()))}
#          # Create an example protocol buffer
#          example = tf.train.Example(features=tf.train.Features(feature=feature))
#    
#          # Serialize to string and write on the file
#          writer.write(example.SerializeToString())
#
#        print "File %s is written." % output_name
#        data_list = []
#        label_list = []
#        image_counter += 1
#        print i
#    
#
#    absfile = os.path.join(dirpath, image_name[idx[i]]) 
#    target_img = io.imread(absfile)
#    data_list.append(target_img)
#    label_list.append(int(class_dict[image_name[idx[i]].split("_")[0]]))
#    #print class_name[image_name[idx[i]].split('_')[0]]
#    #print image_name[idx[i]].split('_')[0]
#
#  #output_name = "train_%d.bin" % image_counter
#  #ouf = open(output_name, 'w')
#  #cPickle.dump(data_dict, ouf, 1)
#  #print "File %s is written." % output_name
#
#  output_name = "train_%d.tfrecords" % image_counter
#  writer = tf.python_io.TFRecordWriter(output_name)
#  data_list
#  label_list
#
#  for j in xrange(0, len(label_list)):
#    feature = {'train/label': _int64_feature(label_list[j]),
#               'train/image': _bytes_feature(tf.compat.as_bytes(data_list[j].tostring()))}
#    # Create an example protocol buffer
#    example = tf.train.Example(features=tf.train.Features(feature=feature))
#  
#    # Serialize to string and write on the file
#    writer.write(example.SerializeToString())
#
#  print "File %s is written." % output_name
 



