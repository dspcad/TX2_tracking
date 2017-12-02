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
  name = 0


  infile = open("label.txt", "r")
  lines = infile.readlines()

  lines = map(lambda s: s.strip(), lines)
  label_dict = {}
  for l in lines:
    elements = l.split()
    label_dict[elements[0]] = elements[1]


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
      name = label_dict[elem.text]


  return name, xmin, xmax, ymin, ymax;


if __name__ == '__main__':
  #datapath = '/home/hhwu/tensorflow_work/TX2_tracking/data_training/bird1'
  datapath = sys.argv[1]
  print "Path: ", datapath


  file_list = []
  for dirpath, dirnames, filenames in os.walk(datapath):
    #print "dirpath: ", dirpath
    #print "dirnames: ", dirnames
    #print "The number of files: %d" % len(filenames)
    
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




  #for xml_elem in xml_list:

  order_idx = np.random.randint(0,len(xml_list),len(xml_list))
  train_idx = order_idx[:int(0.8*len(xml_list))]
  valid_idx = order_idx[int(0.8*len(xml_list)):]


  output_name = "train_1.tfrecords"
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in train_idx:
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
    target_img = io.imread(jpg_f_path)

    #print "label: ", label
    #print "xmin: ", xmin
    #print "xmax: ", xmax
    #print "ymin: ", ymin
    #print "ymax: ", ymax

    feature = {'train/label': _int64_feature(int(label)),
               'train/xmin' : _int64_feature(int(xmin)),
               'train/xmax' : _int64_feature(int(xmax)),
               'train/ymin' : _int64_feature(int(ymin)),
               'train/ymax' : _int64_feature(int(ymax)),
               'train/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  print "File %s is written." % output_name

  output_name = "valid_1.tfrecords"
  writer = tf.python_io.TFRecordWriter(output_name)
  for i in valid_idx:
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
    target_img = io.imread(jpg_f_path)

    #print "label: ", label
    #print "xmin: ", xmin
    #print "xmax: ", xmax
    #print "ymin: ", ymin
    #print "ymax: ", ymax

    feature = {'train/label': _int64_feature(int(label)),
               'train/xmin' : _int64_feature(int(xmin)),
               'train/xmax' : _int64_feature(int(xmax)),
               'train/ymin' : _int64_feature(int(ymin)),
               'train/ymax' : _int64_feature(int(ymax)),
               'train/image': _bytes_feature(tf.compat.as_bytes(target_img.tostring()))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

  print "File %s is written." % output_name



