import cv2
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import psutil

path = '/mnt/home/hheat/USERDIR/counting-bench/data'
train_images = path + '/images'
test_images = path + '/test_images/images'
anno = path + '/annotation'
density_maps = path + '/dmaps'
sm_train_images = path + '/sz_392_train_images'
sm_test_images = path + '/sz_392_test_images'
sm_dmaps = path + '/sz_392_dmaps'

def bench_resize_image(path,des=sm_test_images,size=(680,382),interpolation=cv2.INTER_NEAREST):
    img = cv2.imread(path)
    new_img = cv2.resize(img,size,interpolation)
    outp = des + '/sm_' + path.split('/')[-1]
    cv2.imwrite(outp,new_img)
    
    
Parallel(n_jobs=psutil.cpu_count(),verbose=10)(
    (delayed(bench_resize_image)(fp) for fp in glob(test_images+'/*.jpg'))
)