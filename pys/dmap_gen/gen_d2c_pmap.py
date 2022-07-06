import os
from datetime import datetime
import time
import random
import cv2
import numpy as np

from tqdm import tqdm
from scipy.io import loadmat
from math import floor
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter
 
def gaussian_filter_prob(img, pts):
    img_shape=[img.shape[0],img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    #print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        filter = gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[int(pt[1])][int(pt[0])]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    #print('done.')
    return density

def gaussian_filter_prob_fixed(img, pts):
    img_shape=[img.shape[0],img.shape[1]]
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(pts)
    if gt_count == 0:
        return density

    #print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        sigma =15
        filter = gaussian_filter(pt2d, sigma, mode='constant')
        peak = filter[int(pt[1])][int(pt[0])]
        density_new = filter / float(peak)
        density = np.maximum(density_new, density)
    #print('done.')
    return density

path = '/mnt/home/hheat/USERDIR/counting-bench/data'
train_images = path + '/images'
test_images = path + '/test_images/images'
anno = path + '/annotation'
density_maps = path + '/dmaps'

for images in [train_images, test_images]:
    for fn in tqdm(os.listdir(images)):
        img = cv2.imread(os.path.join(images, fn))
        fn_gt = os.path.join(anno, fn[:6]+'.mat')
        label = loadmat(fn_gt)
        annPoints = label['annotation']
        im_density = gaussian_filter_prob_fixed(img, annPoints)
        np.save(os.path.join(path, 'pmap_d2c_fixed', fn[:6]+'.npy'), im_density)