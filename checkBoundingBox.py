#!/usr/bin/python

#import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import sys
import csv
import os
from skimage import io
from multiprocessing.pool import ThreadPool



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

  height = []
  width  = []
  area = []
  for f in xml_list:
    xml_f_path = "%s/%s" % (datapath, f)
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
    bbox = (xmax-xmin)*(ymax-ymin)


    #print "============================="
    #print "label: ", label
    #print "xmin: ", xmin
    #print "xmax: ", xmax
    #print "ymin: ", ymin
    #print "ymax: ", ymax
    #print "area: ", bbox

    if label != 0:
      height.append(ymax-ymin)
      width.append(xmax-xmin)
      area.append(bbox)

  print "End of file %s." % sys.argv[1]

  print "mean of height: %f      std: %f" % (np.mean(height), np.std(height))
  print "mean of width : %f      std: %f" % (np.mean(width),  np.std(height))
  print "mean of area  : %f      std: %f" % (np.mean(area),   np.std(area))
