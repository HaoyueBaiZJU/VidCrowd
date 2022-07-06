import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import pandas as pd
from glob import glob

import json
from pathlib import Path

import scipy
import scipy.io as io
import pickle
from scipy.ndimage.filters import gaussian_filter

from os.path import isfile
from joblib import Parallel, delayed
import psutil

from parser import args

SEED = args.seed

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


path = '/mnt/home/hheat/USERDIR/counting-bench/data'
train_images = path + '/images'
test_images = path + '/test_images/images'
anno = path + '/annotation'
density_maps = path + '/dmaps'
sm_train_images = path + '/sz_392_train_images'
sm_test_images = path + '/sz_392_test_images'
sm_dmaps = path + '/sz_392_dmaps'


def gaussian_filter_density(img_shape,points):
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
    #print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
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
            sigma = 5 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density


def expand_path(p):
    fn = p.split('/')[-1].split('.')[0]
    if isfile(sm_train_images + '/sm_' + fn + '.jpg'): 
        return sm_train_images + '/sm_' + fn + '.jpg'
    elif isfile(sm_test_images + '/sm_' + fn + '.jpg'):
        return sm_test_images + '/sm_' + fn + '.jpg'
    return p

def open_image(p):
    image = cv2.imread(p)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.
    return image

def get_density_map(p,test=False):
    image_file_p = expand_path(p)
    if image_file_p != p:
        dmap_p = image_file_p.split('/')[-1].split('.')[0]
        dmap_p = sm_dmaps + '/' + dmap_p + '.npy'
        if isfile(dmap_p):
            return
        mat = io.loadmat(p)
        points = mat['annotation'].astype(int)
        ############## RESCALE ################
        points[:,0] = points[:,0] / 2720 * 680
        points[:,1] = points[:,1] / 1530 * 392
        #######################################
        image_shape = cv2.imread(image_file_p).shape[:2]
        if test: 
            density_map = None
        else:
            density_map = gaussian_filter_density(image_shape,points)
            dmap_p = image_file_p.split('/')[-1].split('.')[0]
            dmap_p = sm_dmaps + '/' + dmap_p + '.npy'
            np.save(dmap_p,density_map)
    else:
        density_map = None
        return p
    
    
fns = []
fns.append(Parallel(n_jobs=psutil.cpu_count(),verbose=10)(
    (delayed(get_density_map)(fp) for fp in glob(anno+'/*.mat'))
))

print(fns)