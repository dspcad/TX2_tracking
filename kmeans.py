#!/usr/bin/python

import numpy as np
import sys


def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = np.random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
   assert False, "Shouldn't get here"


def overlap(width_x, height_x, width_y, height_y):
  return min((width_x,width_y))*min((height_x,height_y))

def IOU(box_w, box_h, centroid_w, centroid_h):
  return float(overlap(box_w, box_h, centroid_w, centroid_h))/(box_w*box_h+centroid_w*centroid_h-overlap(box_w, box_h, centroid_w, centroid_h))


def minimum_distnce(elem, centroids):
  distance_to_centroids = []

  for c in centroids:
    distance_to_centroids.append(1-IOU(elem[0], elem[1], c[0], c[1]))

  return min(distance_to_centroids)


def compute_distance(centroids_idx, data_list):
  distance = []
  centroids = [data_list[i] for i in centroids_idx]

  print "centroids: ", centroids
  for elem in data_list:
    distance.append(minimum_distnce(elem, centroids))

  return distance


def clustering(centroids, data_list):

  group = []
  for elem in data_list:

    distance_to_centroids = []
    for c in centroids:
      distance_to_centroids.append(1-IOU(elem[0], elem[1], c[0], c[1]))

    group.append(np.argmin(distance_to_centroids))

  return group


if __name__ == '__main__':
  K = int(sys.argv[1])
  print "K: ", K

  input_file = open('input_kmeans.txt','r')
  content = input_file.readlines()
  content = [x.strip() for x in content] 

  data_list = []
  for x in content:
    elem = x.split()
    data_list.append((int(elem[1]), int(elem[3])))


  data_idx = np.random.randint(0, len(data_list), len(data_list))

  print data_idx
  centroids_idx = [data_idx[0]]
  print centroids_idx[0]
  print data_list[centroids_idx[0]]

  c_w = data_list[centroids_idx[0]][0]
  c_h = data_list[centroids_idx[0]][1]

  org_data_idx = range(0, len(data_list))

  for i in range(0,K-1):
    distance = compute_distance(centroids_idx, data_list)  
    data_idx_and_distance = zip(org_data_idx, distance)
    #print data_idx_and_distance
    centroids_idx.append(weighted_choice(data_idx_and_distance))

  print "===== selected centroids ====="
  centroids = [data_list[x] for x in centroids_idx]
  print centroids



  for counter in range(0, int(sys.argv[2])):
    group_list = clustering(centroids, data_list)
    #print group_list

    for i in range(0,K):
      sum_w = 0.0
      sum_h = 0.0
      num   = 0
      for idx in xrange(0, len(data_list)):
        if group_list[idx] == i:
          sum_w += data_list[idx][0]
          sum_h += data_list[idx][1]
          num += 1

      print "%dth  centroid: %d" % (i+1, num)
      centroids[i] = (sum_w/num, sum_h/num)

    print centroids


  for i in range(0,K):
    area = 0.0
    for idx in xrange(0, len(data_list)):
      centroid = centroids[group_list[idx]]
      elem = data_list[idx]
      area += IOU(elem[0], elem[1], centroid[0], centroid[1])

  print "Average of IOU: %.2f" % (area/len(data_list))

  print centroids


  #for elem in data_list:
  #  #distsnce.append(1-IOU(elem[0],elem[1],data_list[centroids_idx[0]][0], data_list[centroids_idx[0]][1]))
  #  print elem
  #  print 1-IOU(elem[0],elem[1], c_w, c_h)
  

 
  #print overlap(1.2,4.5,3.7,2.9)
  #print IOU(1.2,4.5,3.7,2.9)

  #prob = [('a',1.0),('b',2.0),('c',3.0),('d',4.0)]

  #for i in range(0,10):
  #  print weighted_choice(prob)
