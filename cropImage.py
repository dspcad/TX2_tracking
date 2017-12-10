#!/usr/bin/python

#import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import sys
import csv
import os
from skimage import io

import matplotlib.patches as mpatches


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
  for i in range(0, len(xml_list)):
    xml_f_path = "%s/%s" % (datapath, xml_list[i])
    jpg_f_path = "%s/%s" % (datapath, image_list[i])
    #print "Path: ", f_path
 
    label, xmin, xmax, ymin, ymax = xmlParser(xml_f_path)
    target_img = io.imread(jpg_f_path)
    #target_img = Image.open(jpg_f_path).convert("RGBA")

    pos_x = 0
    pos_y = 0

    if xmin - (224 -(xmax-xmin))/2 > 0 and xmax + (224 -(xmax-xmin))/2 <= 640:
      pos_x = xmin - (224 -(xmax-xmin))/2
    elif xmin - (224 -(xmax-xmin))/2 > 0 and xmax + (224 -(xmax-xmin))/2 > 640:
      pos_x =  640 - 224

    if ymin - (224 -(ymax-ymin))/2 > 0 and ymax + (224 -(ymax-ymin))/2 <= 360:
      pos_y = ymin - (224 -(ymax-ymin))/2
    elif ymin - (224 -(ymax-ymin))/2 > 0 and ymax + (224 -(ymax-ymin))/2 > 360:
      pos_y =  360 - 224

    ###############################
    #       shape[0]: height      #
    #       shape[1]: width       #
    ###############################
    target_img[ymin:ymax,xmin:xmax,:0] = 0
    target_img[ymin:ymax,xmin:xmax,:1] = 0
    target_img[ymin:ymax,xmin:xmax,:2] = 0
    target_img = target_img[pos_y:pos_y+224, pos_x:pos_x+224, :]
    io.imsave("%s%d%s" % ("test_img_", i, '.jpeg'), target_img)
    

  print "End of file %s." % sys.argv[1]

