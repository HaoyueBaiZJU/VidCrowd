import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import json
from pathlib import Path
import pickle

from tqdm import tqdm

from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstring

import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter

path = '/mnt/home/zpengac/USERDIR/count/UCF_CC_50'
train_images = path + '/images'
#train_images_TIR = path + '/Train/TIR'
anno = path + '/ground_truth'
dmap_folder = path + '/dmaps'


# Copy from MCNN paper,modified locally
import scipy
import scipy.io as io
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
            sigma = 15 #np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #print ('done.')
    return density


class VIS_PROC:
    def __init__(self,images_path,anno_path,output_path,kernel_func,test=False):
        self.images_path,self.anno_path,self.output_path = images_path,anno_path,output_path
        self.kernel_func = kernel_func
        self.test = test
        #self.output = []
        
        
    def run_proc(self):
        for anno_file in tqdm(sorted(glob(self.anno_path+'/*.xml'),key=lambda x: int(x.split('/')[-1].split('.')[0][:-1])),total=len(glob(self.anno_path+'/*.xml'))):
            with open(anno_file,'r') as f:
                data = f.read()
            try:
                test_data = bf.data(fromstring(data))
                img_shape = test_data['annotation']['size']['height']['$'],test_data['annotation']['size']['width']['$']
                img = np.zeros(img_shape)
                points = self._get_annotation(test_data)
                d_map = self.kernel_func(img,points)
                fname = anno_file.split('/')[-1].split('.')[0][:-1] + 'Dmap.pkl'
                out_filename = self.output_path + '/' + fname
                with open(out_filename,'wb') as f:
                    pickle.dump(d_map,f)
            except:
                print(anno_file)
                e = sys.exc_info()[0]
                print(e)
            if self.test:
                raise ValueError('Test Stop...')
#             outp_file = self.output_path + '/gt_dmap.pkl'
#             with open(outp_file,'wb') as f:
#                 pickle.dump(self.output,f)
        
    
    def _get_annotation(self,xml_datas):
        points = []
        for data in xml_datas['annotation']['object']:
            if data['name']['$'] == 'person' or data['name']['$'] == 'person]':
                if 'point' in data:
                    x = data['point']['x']['$']
                    y = data['point']['y']['$']
                else:
                    x = data['bndbox']['xmin']['$']
                    y = data['bndbox']['ymin']['$']
                points.append((x,y))
            else: print(data['name']['$'] +' not a person...')
        return points

#     def plot_data(img,points=None,fig_size=(18,12)):
#         fig, ax = plt.subplots(1, 1, figsize=fig_size)
#         if points:
#             for point in points:
#     #             x,y = datas['points']
#     #             point =(int(x),int(y))
#                 cv2.circle(image, point, radius=0,color=(0, 1, 0), thickness=5)
#         ax.imshow(img)
        