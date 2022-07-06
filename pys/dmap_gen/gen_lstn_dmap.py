import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

path = '/mnt/home/hheat/USERDIR/counting-bench/data'
train_images = path + '/sm_train_images'
test_images = path + '/sm_test_images'
anno = path + '/annotation'

for images in [train_images, test_images]:
    for fn in tqdm(os.listdir(images)):
        if 'checkpoint' in fn:
            continue
        img = plt.imread(os.path.join(images, fn))
        fn_gt = os.path.join(anno, fn[3:9]+'.mat')
        label = io.loadmat(fn_gt)
        annPoints = label['annotation']
        annPoints[:,0] = annPoints[:,0] / 2720 * 1360
        annPoints[:,1] = annPoints[:,1] / 1530 * 784
        im_density = np.zeros((img.shape[0],img.shape[1]))
        for i in range(0,len(annPoints)):
            if int(annPoints[i][1])<img.shape[0] and int(annPoints[i][0])<img.shape[1]:
                im_density[int(annPoints[i][1]),int(annPoints[i][0])]=1
        im_density = gaussian_filter(im_density, 15)
        np.save(os.path.join(path, 'sm_dmap_lstn', 'sm_' + fn[3:9] + '.npy'), im_density)