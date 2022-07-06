import sys
import os
from datetime import datetime
import time
import random
import cv2
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat
from math import floor

import matplotlib.pyplot as plt
from glob import glob

import json
from pathlib import Path
import pickle

from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
 
def gaussian_filter_density(img,points):
    '''
    This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.
    points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
    img_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.
    return:
    density: the density-map we want. Same shape as input image but only has one channel.
    example:
    points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
    img_shape: (768,1024) 768 is row and 1024 is column.
    '''
    img_shape=[img.shape[0],img.shape[1]]
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    #print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 3:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = 15 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density

path = '/mnt/home/zpengac/USERDIR/count/SmartCity'
images = path + '/images'
anno = path + '/ground_truth'

for fn in tqdm(os.listdir(images)):
    img = cv2.imread(os.path.join(images, fn))
    fn_gt = os.path.join(anno, fn.split('.')[0]+'.mat')
    label = loadmat(fn_gt)
    annPoints = label['loc']
    im_density = gaussian_filter_density(img, annPoints)
    np.save(os.path.join(path, 'density_maps', fn.split('.')[0]+'.npy'), im_density)