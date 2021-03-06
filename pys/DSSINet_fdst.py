import torch
import torch.nn as nn
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
from glob import glob


from tqdm import tqdm

from utils import visualize, plot_data
from scipy.io import loadmat

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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

path = '/mnt/home/hbaiaa/USERDIR/data/FDST'
train_images = path + '/train/img'
test_images = path + '/test/img'
#anno = path + '/annotation'
#images = path + '/rf_image_vehicle'
#density_maps = path + '/rf_GT_vehicle'

LOG_PARA = args.log_para

def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.Resize(360,640,interpolation=2),
            #A.RandomSizedCrop(min_max_height=(409, 512), height=409, width=512, p=1.0),
            #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=1.0),
        ],
        #additional_targets={'image': 'image','image1': 'image'}
        #keypoint_params = A.KeypointParams(format='xy')
)

def get_train_image_only_transforms():
    return A.Compose(
        [
            #A.Resize(360,640),
            #A.OneOf([
            #    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
            #                         val_shift_limit=0.2, p=0.9),
            #    A.RandomBrightnessContrast(brightness_limit=0.2, 
            #                               contrast_limit=0.2, p=0.9),
            #],p=0.9),
            #A.Blur(blur_limit=3,p=0.2),
            A.Normalize(mean=mean,std=std,p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ],
        additional_targets={'image': 'image'}
    )

def get_valid_trainsforms():
    return A.Compose(
        [
            #A.Resize(360,640,interpolation=2),
            A.Normalize(mean=mean,std=std,p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ]
    )

# def get_valid_image_only_transforms():
#     return A.Compose(
#         [
#             A.Resize(360,640),
#         ],
#         additional_targets={'image': 'image'}
#     )

mean = torch.tensor([0.4939, 0.4794, 0.4583])
std = torch.tensor([0.2177, 0.2134, 0.2144])

def denormalize(img):
    img = img * std[...,None,None] + mean[...,None,None]
    img = img.permute(1,2,0).cpu().numpy()
    return img

class Counting_Dataset(Dataset):
    def __init__(self,path,image_fnames,dmap_folder,gt_folder=None,transforms=None,mosaic=False,downsample=4):
        '''
            path: root path 
            image_fnames: path of images
            dmap_folder: density map folder, eg: /dmap
            gt_folder: gt folder, currently set to visdrone xml format, modify _get_gt_data() if needed
            transforms: iteratable, can be tuple / list ... etc
            mosaic: mix up image and density map to form a new image, set to false by default
            downsample: resize dmap
        '''
        super().__init__()
        self.path = path
        self.image_fnames = image_fnames
        self.dmap_folder = path + dmap_folder
        self.transforms = transforms
        self.mosaic = mosaic
        self.downsample = downsample
        self.gt_folder = gt_folder # test purpose
        
    def __len__(self):
        return len(self.image_fnames)
    
    def __getitem__(self,idx):
        image_id = self.image_fnames[idx]
        
        if self.mosaic and random.randint(0,1) < 0.5:
            image, density_map, gt_points = self._load_mosaic_image_and_density_map(idx)
        else:
            image, density_map, gt_points = self._load_image_and_density_map(idx)
        
        h,w = image.shape[0]//self.downsample, image.shape[1]//self.downsample
        image = cv2.resize(image,(w, h))
        density_map = cv2.resize(density_map,(w//(self.downsample*2),h//(self.downsample*2)))#,interpolation=cv2.INTER_NEAREST)
        
        # Warning: doesn't work for cutout, uncommet transform and make fix code to enable cutout
        # Reason: cutout doesn't apply to mask, so mask must be image. check 01a bottom for code
        if self.transforms:
            for tfms in self.transforms:
                aug = tfms(**{
                    'image': image,
                    'mask': density_map,
                    #'keypoints': gt_points
                })
                #image, density_map, gt_points = aug['image'], aug['mask'], aug['keypoints']
                image, density_map = aug['image'], aug['mask'] # issue with previous keypoints (albumentation?)
        
        
        return image, density_map, image_id, gt_points
        
    
    def _get_dmap_name(self,fn):
        mask_name = fn.split('/')[-1].split('.')[0]
        mask_path = self.dmap_folder + '/' + mask_name + '.npy'
        return mask_path
    
    def _load_image_and_density_map(self,idx):
        image_fname = self.image_fnames[idx]
        dmap_fname = self._get_dmap_name(image_fname)
        image = cv2.imread(image_fname)
        d_map = np.load(dmap_fname,allow_pickle=True)
        d_map = d_map.squeeze()
        pad_h = 0
        pad_w = 0
        if image.shape[0] < 1080:
            pad_h = (1080 - image.shape[0]) // 2
        if image.shape[1] < 1920:
            pad_w = (1920 - image.shape[1]) // 2
        image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        d_map = cv2.copyMakeBorder(d_map, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.
        
        #sanity check gt
        _, points = self._get_gt_data(idx)
        # end sanity check
        
        return image, d_map, points
    
    def _load_mosaic_image_and_density_map(self,idx):
        image_1, dmap_1, points_1 = self._load_image_and_density_map(idx)
        while True:
            idx_2 = random.randint(0,len(self.image_fnames)-1)
            if idx != idx_2:
                break
        image_2, dmap_2, points_2 = self._load_image_and_density_map(idx_2)
        
        imsize = min(*image_1.shape[:2])
        xc,yc = [int(random.uniform(imsize*0.4,imsize*0.6)) for _ in range(2)]
        h,w = image_1.shape[0], image_1.shape[1]

        pos = random.randint(0,1)
        if pos == 0: #top left
            x1a,y1a,x2a,y2a = 0,0,xc,yc # img_1
            x1b,y1b,x2b,y2b = w-xc,h-yc,w,h # img_2
        elif pos == 1: # top right
            x1a,y1a,x2a,y2a = w-xc,0,w,yc
            x1b,y1b,x2b,y2b = 0,h-yc,xc,h
        elif pos == 2: # bottom left
            x1a,y1a,x2a,y2a = 0,h-yc,xc,h
            x1b,y1b,x2b,y2b = w-xc,0,w,yc
        elif pos == 3: # bottom right
            x1a,y1a,x2a,y2a = w-xc,h-yc,w,h
            x1b,y1b,x2b,y2b = 0,0,xc,yc
        
        new_image = image_1.copy()
        new_dmap = dmap_1.copy()
        new_image[y1a:y2a,x1a:x2a] = image_2[y1b:y2b,x1b:x2b]
        new_dmap[y1a:y2a,x1a:x2a] = dmap_2[y1b:y2b,x1b:x2b]
        
        #TODO: sanity check to see generate gt
        
        new_gt_points = self._get_mixed_gt_points(points_1,points_2,(x1a,y1a,x2a,y2a),(x1b,y1b,x2b,y2b),(h,w))
        
        return new_image, new_dmap, new_gt_points
    
    '''
    The follow section blocks are for sanity check 
    to compare dmap.sum() with gt points
    remove if needed
    '''
    def _get_mixed_gt_points(self,points_1,points_2,img_1_loc, img_2_loc,img_shape):
#         fn_1, points_1 = self._get_gt_data(idx_1)
#         fn_2, points_2 = self._get_gt_data(idx_2)
        x1a,y1a,x2a,y2a = img_1_loc
        x1b,y1b,x2b,y2b = img_2_loc
        h,w = img_shape
        
        result_boxes = []
        result_boxes.append(points_2)
        result_boxes = np.concatenate(result_boxes,0)
        padw = x1a-x1b
        pady = y1a-y1b

        result_boxes[:,0] += padw
        result_boxes[:,1] += pady

        np.clip(result_boxes[:,0],0,w,out=result_boxes[:,0])
        np.clip(result_boxes[:,1],0,h,out=result_boxes[:,1])
        result_boxes = result_boxes.astype(np.int32)

        result_boxes = result_boxes[np.where(result_boxes[:,0] * result_boxes[:,1] > 0)]
        result_boxes = result_boxes[np.where(result_boxes[:,0] < w)]
        result_boxes = result_boxes[np.where(result_boxes[:,1] < h)]
        
        boxes = []
        for (x,y) in points_1:
            if x >= x1a and x <= x2a and y >= y1a and y <= y2a:
                continue
            else:
                boxes.append((x,y))
        if len(boxes) == 0:
            return result_boxes
        return np.concatenate((boxes, result_boxes),axis=0)
    
    def _get_gt_data(self,idx):
        if not self.gt_folder:
            return (None,0)
        fn = self.image_fnames[idx]
        anno_path = self.path + self.gt_folder + '/' + fn.split('/')[-1].split('.')[0] + '.mat'
        test_data = loadmat(anno_path)
        points = test_data['annotation'].astype(int)
        return fn, points

# ADD LOG_PARA to density map

class Crop_Dataset(Counting_Dataset):
    def __init__(self,path,image_fnames,dmap_folder,gt_folder=None,transforms=None,mosaic=False,downsample=4,crop_size=512,method='train'):
        super().__init__(path,image_fnames,dmap_folder,gt_folder,transforms,mosaic,downsample)
        self.crop_size = crop_size
        if method not in ['train','valid']:
            raise Exception('Not Implement')
        self.method = method
    
    def __getitem__(self,idx):
        fn = self.image_fnames[idx]
        
        image,density_map,gt_points = self._load_image_and_density_map(idx)
        h,w = image.shape[0], image.shape[1]
        #image = cv2.resize(image,(w, h))
        
        
        if self.method == 'train':
            #h,w = image.shape[:2]
            i,j = self._random_crop(h,w,self.crop_size,self.crop_size)
            image = image[i:i+self.crop_size,j:j+self.crop_size]
            density_map = density_map[i:i+self.crop_size,j:j+self.crop_size]
            
            gt_points = gt_points - [j,i]
            mask = (gt_points[:,0] >=0 ) * (gt_points[:,0] <= self.crop_size) * (gt_points[:,1]>=0) * (gt_points[:,1]<=self.crop_size)
            gt_points = gt_points[mask]
            density_map = cv2.resize(density_map,(self.crop_size//self.downsample,self.crop_size//self.downsample))
            
        else:
            image = image[4:-4,:,:]
            density_map = cv2.resize(density_map,(w//self.downsample,h//self.downsample))#,interpolation=cv2.INTER_NEAREST)
            density_map = density_map[4:-4,:]
            #density_map = density_map[1:-1,:]
        
        if self.transforms:
            for tfms in self.transforms:
                aug = tfms(**{
                    'image': image,
                    'mask': density_map,
                    #'keypoints': gt_points
                })
                #image, density_map, gt_points = aug['image'], aug['mask'], aug['keypoints']
                image, density_map = aug['image'], aug['mask'] # issue with previous keypoints (albumentation?)
        return image, density_map*LOG_PARA, fn, gt_points
    
    def _random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j

train_fp = glob(train_images + '/*.jpg')
test_fp = glob(test_images + '/*.jpg')

train_dataset = Crop_Dataset(path=path,
                             image_fnames=train_fp,dmap_folder='/train/ground_truth',
                             gt_folder='/train/annotation',
                             transforms=[get_train_transforms(),get_train_image_only_transforms()],
                             downsample=args.downsample,
                             crop_size=args.crop_size
                                )

valid_dataset = Crop_Dataset(path=path,
                             image_fnames=test_fp,dmap_folder='/test/ground_truth',
                             gt_folder='/test/annotation',
                             transforms=[get_valid_trainsforms()],
                             method='valid',
                             downsample=args.downsample
                             #crop_size=args.crop_size
                                )

class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = 0.0002

    folder = 'DSSINet-9.1-fdst-512'
    downsample = args.downsample

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=args.max_lr,
        #total_steps = len(train_dataset) // 4 * n_epochs, # gradient accumulation
        epochs=n_epochs,
        steps_per_epoch=int(len(train_dataset) / batch_size),
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy, 
        final_div_factor=args.final_div_factor
    )
    
#     SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
#     scheduler_params = dict(
#         mode='min',
#         factor=0.5,
#         patience=1,
#         verbose=False, 
#         threshold=0.0001,
#         threshold_mode='abs',
#         cooldown=0, 
#         min_lr=1e-8,
#         eps=1e-08
#     )

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from collections import OrderedDict

def same_padding_length(input_length, filter_size, stride, dilation=1):
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    output_length = (input_length + stride - 1) // stride
    pad_length = max(0, (output_length - 1) * stride + dilated_filter_size - input_length)
    return pad_length

def compute_same_padding2d(input_shape, kernel_size, strides, dilation):
    space = input_shape[2:]
    assert len(space) == 2, "{}".format(space)
    new_space = []
    new_input = []
    for i in range(len(space)):
        pad_length = same_padding_length(
            space[i],
            kernel_size[i],
            stride=strides[i],
            dilation=dilation[i])
        new_space.append(pad_length)
        new_input.append(pad_length % 2)
    return tuple(new_space), tuple(new_input)

class Conv2d_dilated(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, dilation=1, bn=False, bias=True, groups=1):
        super(Conv2d_dilated, self).__init__()
        self.conv = _Conv2d_dilated(in_channels, out_channels, kernel_size, 'zeros', stride, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=True) 
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        elif NL == 'tanh':
            self.relu = nn.Tanh()
        elif NL == 'lrelu':
            self.relu = nn.LeakyReLU(inplace=True)
        elif NL == 'sigmoid':
            self.relu = nn.Sigmoid()
        else:
            self.relu = None
    
    @torch.cuda.amp.autocast()
    def forward(self, x, dilation=None):
        x = self.conv(x, dilation)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class _Conv2d_dilated(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, stride=1, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        super(_Conv2d_dilated, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding_mode=padding_mode, 
            stride=stride, groups=groups, dilation=dilation, bias=bias, padding=_pair(0), output_padding=_pair(0), transposed=False)
    
    @torch.cuda.amp.autocast()
    def forward(self, input, dilation=None):
        input_shape = list(input.size())
        dilation_rate = self.dilation if dilation is None else _pair(dilation)
        padding, pad_input = compute_same_padding2d(input_shape, kernel_size=self.kernel_size, strides=self.stride, dilation=dilation_rate)

        if pad_input[0] == 1 or pad_input[1] == 1:
            input = F.pad(input, [0, int(pad_input[0]), 0, int(pad_input[1])])
        return F.conv2d(input, self.weight, self.bias, self.stride,
                       (padding[0] // 2, padding[1] // 2), dilation_rate, self.groups)
        #https://github.com/pytorch/pytorch/issues/3867
        
class SequentialEndpoints(nn.Module):

    def __init__(self, layers, endpoints=None):
        super(SequentialEndpoints, self).__init__()
        assert isinstance(layers, OrderedDict)
        for key, module in layers.items():
            self.add_module(key, module)
        if endpoints is not None:
            self.Endpoints = namedtuple('Endpoints', endpoints.values(), verbose=True)
            self.endpoints = endpoints


    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def sub_forward(self, startpoint, endpoint):
        def forward(input):
            flag = False
            for key, module in self._modules.items():
                if startpoint == endpoint:
                    output = input
                    if key == startpoint:
                        output = module(output)
                        return output
                elif flag or key == startpoint:
                    if key == startpoint:
                        output = input
                    flag = True
                    output = module(output)
                    if key == endpoint:
                        return output
            return output
        return forward
    
    @torch.cuda.amp.autocast()
    def forward(self, input, require_endpoints=False):
        if require_endpoints:
            endpoints = self.Endpoints([None] * len(self.endpoints.keys()))
        for key, module in self._modules.items():
            input = module(input)
            if require_endpoints and key in self.endpoints.keys():
                setattr(endpoints, self.endpoints[key], input)
        if require_endpoints:
            return input, endpoints
        else:
            return input

import math

#model_urls = {
#    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
#}


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, output_stride=8, base_dilated_rate=1, NL='relu', bias=True):
    layers = []
    in_channels = 3
    layers = OrderedDict()
    idx = 0
    curr_stride = 1
    dilated_rate = base_dilated_rate
    for v in cfg:
        name, ks, padding = str(idx), (3, 3), (1, 1)
        if type(v) is tuple:
            if len(v) == 2:
                v, ks = v
            elif len(v) == 3:
                name, v, ks = v
            elif len(v) == 4:
                name, v, ks, padding = v
        if v == 'M':
            if curr_stride >= output_stride:
                dilated_rate = 2
                curr_stride *= 2
            else:
                layers[name] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                curr_stride *= 2
            idx += 1
        elif v == 'None':
            idx += 1
        else:
            # conv2d = _Conv2d_dilated(in_channels, v, dilation=dilated_rate, kernel_size=ks, bias=bias)
            conv2d = nn.Conv2d(in_channels, v, dilation=dilated_rate, kernel_size=ks, padding=padding, bias=bias)
            dilated_rate = base_dilated_rate
            layers[name] = conv2d
            idx += 1
            if batch_norm:
                layers[str(idx)] = nn.BatchNorm2d(v)
                idx += 1
            if NL == 'relu' :
                relu = nn.ReLU(inplace=True)
            if NL == 'nrelu' :
                relu = nn.ReLU(inplace=False)
            elif NL == 'prelu':
                relu = nn.PReLU()
            layers['relu'+str(idx)] = relu
            idx += 1
            in_channels = v
    print("\n".join(["{}: {}-{}".format(i, k, v) for i, (k,v) in enumerate(layers.items())]))
    return SequentialEndpoints(layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
    'G': [64, 64, 'M', 128, 128, 'M', 256, 256, 256],
    'H': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'None', 512, 512, 512, 'None', 512, 512, 512],
    'I': [24, 22, 'M', 41, 51, 'M', 108, 89, 111, 'M', 184, 276, 228],
}

def vgg16(struct='F', **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg[struct], **kwargs))

    return model

import torch.nn.functional as F

## CRFFeatureRF
class MessagePassing(nn.Module):
    def __init__(self, branch_n, input_ncs, bn=False):
        super(MessagePassing, self).__init__()
        self.branch_n = branch_n
        self.iters = 2
        for i in range(branch_n):
            for j in range(branch_n):
                if i == j:
                    continue
                setattr(self, "w_0_{}_{}_0".format(j, i),                         nn.Sequential(
                                Conv2d_dilated(input_ncs[j],  input_ncs[i], 1, dilation=1, same_padding=True, NL=None, bn=bn),
                            )
                        )
        self.relu = nn.ReLU(inplace=False)
        self.prelu = nn.PReLU()
    
    @torch.cuda.amp.autocast()
    def forward(self, input):
        hidden_state = input
        side_state = []

        for _ in range(self.iters):
            hidden_state_new = []
            for i in range(self.branch_n):

                unary = hidden_state[i]
                binary = None
                for j in range(self.branch_n):
                    if i == j:
                        continue
                    if binary is None:
                        binary = getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])
                    else:
                        binary = binary + getattr(self, 'w_0_{}_{}_0'.format(j, i))(hidden_state[j])

                binary = self.prelu(binary)
                hidden_state_new += [self.relu(unary + binary)]
            hidden_state = hidden_state_new

        return hidden_state

class CRFVGG(nn.Module):
    def __init__(self, output_stride=8, bn=False):
        super(CRFVGG, self).__init__()

        self.output_stride = output_stride

        self.pyramid = [2, 0.5]

        self.front_end = vgg16(struct='F', NL="prelu", output_stride=self.output_stride)


        self.passing1 = MessagePassing( branch_n=2, 
                                        input_ncs=[128, 64],
                                        )
        self.passing2 = MessagePassing( branch_n=3, 
                                        input_ncs=[256, 128, 64],
                                        )
        self.passing3 = MessagePassing( branch_n=3, 
                                        input_ncs=[512, 256, 128]
                                        )
        self.passing4 = MessagePassing( branch_n=2, 
                                        input_ncs=[512, 256]
                                        )


        self.decoder1 = nn.Sequential(
                Conv2d_dilated(512, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder2 = nn.Sequential(
                Conv2d_dilated(768, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder3 = nn.Sequential(
                Conv2d_dilated(896, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder4 = nn.Sequential(
                Conv2d_dilated(448, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.decoder5 = nn.Sequential(
                Conv2d_dilated(192, 128,   1, dilation=1, same_padding=True, NL='relu', bn=bn),
                Conv2d_dilated(128,   1,   3, dilation=1, same_padding=True, NL=None, bn=bn),
            )

        self.passing_weight1 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight2 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight3 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.passing_weight4 = Conv2d_dilated(1,  1, 3, same_padding=True, NL=None, bn=bn)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()


    @torch.cuda.amp.autocast()
    def forward(self, im_data, return_feature=False):
        conv1_2 = ['0', 'relu3']
        conv1_2_na = ['0', '2']
        conv2_2 = ['4', 'relu8']
        conv2_2_na = ['4', '7']
        conv3_3 = ['9', 'relu15']
        conv3_3_na = ['9', '14']
        # layer 16 is the max pooling layer
        conv4_3 = ['16', 'relu22']
        conv4_3_na = ['16', '21']
        # droping the last pooling layer, 17 would become dilated with rate 2
        # conv4_3 = ['17', 'relu22']

        batch_size, C, H, W = im_data.shape

        with torch.no_grad():
            im_scale1 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[0]), int(W * self.pyramid[0])), align_corners=False, mode="bilinear")
            im_scale2 = im_data
            im_scale3 = nn.functional.upsample(im_data, size=(int(H * self.pyramid[1]), int(W * self.pyramid[1])), align_corners=False, mode="bilinear")


        mp_scale1_feature_conv2_na = self.front_end.features.sub_forward(conv1_2[0], conv2_2_na[1])(im_scale1)
        mp_scale2_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale2)

        mp_scale1_feature_conv2, mp_scale2_feature_conv1                         = self.passing1([mp_scale1_feature_conv2_na, mp_scale2_feature_conv1_na])


        aggregation4 = torch.cat([mp_scale1_feature_conv2, mp_scale2_feature_conv1], dim=1)

        mp_scale1_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale1_feature_conv2)
        mp_scale2_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale2_feature_conv1)
        mp_scale3_feature_conv1_na = self.front_end.features.sub_forward(*conv1_2_na)(im_scale3)


        mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1                         = self.passing2([mp_scale1_feature_conv3_na, mp_scale2_feature_conv2_na, mp_scale3_feature_conv1_na])
        aggregation3 = torch.cat([mp_scale1_feature_conv3, mp_scale2_feature_conv2, mp_scale3_feature_conv1], dim=1)


        mp_scale1_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale1_feature_conv3)
        mp_scale2_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale2_feature_conv2)
        mp_scale3_feature_conv2_na = self.front_end.features.sub_forward(*conv2_2_na)(mp_scale3_feature_conv1)

        mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2                         = self.passing3([mp_scale1_feature_conv4_na, mp_scale2_feature_conv3_na, mp_scale3_feature_conv2_na])
        aggregation2 = torch.cat([mp_scale1_feature_conv4, mp_scale2_feature_conv3, mp_scale3_feature_conv2], dim=1)

        mp_scale2_feature_conv4_na = self.front_end.features.sub_forward(*conv4_3_na)(mp_scale2_feature_conv3)
        mp_scale3_feature_conv3_na = self.front_end.features.sub_forward(*conv3_3_na)(mp_scale3_feature_conv2)

        mp_scale2_feature_conv4, mp_scale3_feature_conv3                         = self.passing4([mp_scale2_feature_conv4_na, mp_scale3_feature_conv3_na])
        aggregation1 = torch.cat([mp_scale2_feature_conv4, mp_scale3_feature_conv3], dim=1)

        mp_scale3_feature_conv4 = self.front_end.features.sub_forward(*conv4_3)(mp_scale3_feature_conv3)

        dens1 = self.decoder1(mp_scale3_feature_conv4)
        dens2 = self.decoder2(aggregation1)
        dens3 = self.decoder3(aggregation2)
        dens4 = self.decoder4(aggregation3)
        dens5 = self.decoder5(aggregation4)
        #print(dens1.shape, dens2.shape, dens3.shape, dens4.shape, dens5.shape)
        dens1 = self.prelu(dens1)
        dens2 = self.prelu(dens2 + self.passing_weight1(nn.functional.upsample(dens1, scale_factor=2, align_corners=False, mode="bilinear")))
        dens3 = self.prelu(dens3 + self.passing_weight2(nn.functional.upsample(dens2, scale_factor=2, align_corners=False, mode="bilinear")))
        dens4 = self.prelu(dens4 + self.passing_weight3(nn.functional.upsample(dens3, scale_factor=2, align_corners=False, mode="bilinear")))
        dens5 = self.relu(dens5 + self.passing_weight4(nn.functional.upsample(dens4, scale_factor=2, align_corners=False, mode="bilinear")))

        return dens5

def MSELoss_MCNN(preds,targs):
    return nn.MSELoss()(preds,targs)

def MAELoss_MCNN(preds,targs,upsample):
    return nn.L1Loss()((preds/LOG_PARA).sum(dim=[-1,-2])*upsample*upsample, (targs/LOG_PARA).sum(dim=[-1,-2])*upsample*upsample)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def do_valid(epoch):
    if epoch < 50:
        return epoch % 10 == 9
    elif epoch < 80:
        return epoch % 5 == 4
    else:
        return True

import warnings
warnings.filterwarnings("ignore")

#opt_level ='O1' # apex

class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/mnt/home/zpengac/USERDIR/count/drone_benchmark/cps/fdst/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        
        #self.model, self.optimizer = amp.initialize(self.model,self.optimizer,opt_level=opt_level) # apex
        self.scaler = torch.cuda.amp.GradScaler()
        
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.criterion = MSELoss_MCNN
        self.metric = MAELoss_MCNN
        self.log(f'Fitter prepared. Device is {self.device}')
        
        # self.iters_to_accumulate = 4 # gradient accumulation

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, mae_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mae_loss: {mae_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            
            if do_valid(e):
                t = time.time()
                summary_loss, mae_loss = self.validation(validation_loader)

                self.log(f'[RESULT]: Val. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
                self.log(f'[RESULT]: Val. Epoch: {self.epoch}, mae_loss: {mae_loss.avg:.8f}, time: {(time.time() - t):.5f}')
                if summary_loss.avg < self.best_summary_loss:
                    self.best_summary_loss = summary_loss.avg
                    self.model.eval()
                    self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                    for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
                        os.remove(path)

                if self.config.validation_scheduler:
                    self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        mae_loss = AverageMeter()
        t = time.time()
        for step, (images, density_maps, fns, gt_pts) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'mse_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = images.shape[0]
                images = images.cuda().float()
                density_maps = density_maps.cuda().float()
                

                #preds = self.model(images)
                with torch.cuda.amp.autocast(): #native fp16
                    preds = self.model(images)
                    loss = self.criterion(preds,density_maps)
                    metric_loss = self.metric(preds,density_maps,self.config.downsample)
                mae_loss.update(metric_loss.detach().item(),batch_size)
                summary_loss.update(loss.detach().item(), batch_size)
                
            #if step == 20:
            #    break

        return summary_loss, mae_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        mae_loss = AverageMeter()
        t = time.time()
        for step, (images, density_maps, fns, gt_pts) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'mse_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            images = images.cuda().float()
            batch_size = images.shape[0]
            density_maps = density_maps.cuda().float()
            
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(): #native fp16
                preds = self.model(images)
                loss = self.criterion(preds,density_maps)
                metric_loss = self.metric(preds.detach(),density_maps.detach(),self.config.downsample)
            self.scaler.scale(loss).backward()
            
            # loss = loss / self.iters_to_accumulate # gradient accumulation
            
#             with amp.scale_loss(loss,self.optimizer) as scaled_loss: # apex
#                 scaled_loss.backward()
            #loss.backward()

            
            mae_loss.update(metric_loss.detach().item(),batch_size)
            summary_loss.update(loss.detach().item(), batch_size)
            
            #self.optimizer.step()
            self.scaler.step(self.optimizer) # native fp16
            
            if self.config.step_scheduler:
                self.scheduler.step()
            
            self.scaler.update() #native fp16
                
                
#             if (step+1) % self.iters_to_accumulate == 0: # gradient accumulation

#                 self.optimizer.step()
#                 self.optimizer.zero_grad()

#                 if self.config.step_scheduler:
#                     self.scheduler.step()
                    
            #if step == 20:
            #    break

        return summary_loss, mae_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
            #'amp': amp.state_dict() # apex
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')

def dist_init(host_addr, rank, local_rank, world_size, port=23456):
    host_addr_full = 'tcp://' + host_addr + ':' + str(port)
    torch.distributed.init_process_group("gloo", init_method=host_addr_full,
                                         rank=rank, world_size=world_size)
    assert torch.distributed.is_initialized()
    
def get_ip(iplist):
    ip = iplist.split('[')[0] + iplist.split('[')[1].split('-')[0]
    return ip

rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
world_size = int(os.environ['SLURM_NTASKS'])
iplist = os.environ['SLURM_JOB_NODELIST']
#ip = get_ip(iplist)
print(iplist, rank, local_rank, world_size)

dist_init(iplist, rank, local_rank, world_size, 23456)

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
val_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)

def collate_fn(batch):
    imgs, dmaps, fns, gt_points = zip(*batch)
    imgs = torch.stack(imgs)
    dmaps = torch.stack(dmaps).unsqueeze(1)
    return imgs,dmaps,fns,gt_points

def run_training():
    device = torch.device('cuda:0')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        #sampler=RandomSampler(train_dataset),
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size//2,
        num_workers=TrainGlobalConfig.num_workers//2,
        shuffle=False,
        #sampler=SequentialSampler(valid_dataset),
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)

net = CRFVGG().cuda()
net = DistributedDataParallel(net, find_unused_parameters=True)

if args.mode == 'train':
    run_training()
else:
    # Comment all the code below when training
    val_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=1,
            num_workers=1,
            shuffle=False,
            sampler=SequentialSampler(valid_dataset),
            pin_memory=True,
            collate_fn=collate_fn,
    )

    test_net = CRFVGG().cuda()
    test_net = nn.DataParallel(test_net)

    checkpoint = torch.load(f'/mnt/home/dty80/USERDIR/count/drone_benchmark/DENet-8.26-smartcity-448/best-checkpoint-007epoch.bin')
    test_net.load_state_dict(checkpoint['model_state_dict'])
    test_net.eval()

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as pnsr

    pre_count = []
    gt_count = []
    gt_points = []
    avg_ssim = AverageMeter()
    avg_pnsr = AverageMeter()
    for imgs, dmaps, fns, points in tqdm(val_loader):
        with torch.no_grad():
            imgs = imgs.cuda().float()
            preds = test_net(imgs) / LOG_PARA
        dmaps = dmaps / LOG_PARA
        
        for pred, dmap in zip(preds, dmaps):
            pred_array = pred.detach().cpu().numpy().squeeze()
            dmap_array = dmap.detach().cpu().numpy().squeeze()
            avg_ssim.update(ssim(dmap_array, pred_array, data_range=dmap_array.max()-dmap_array.min()))
            avg_pnsr.update(pnsr(dmap_array, pred_array, data_range=dmap_array.max()-dmap_array.min()))
        
        pre_count.extend(preds.sum(dim=[-1,-2]).detach().cpu().numpy())
        
        gt_count.extend(dmaps.sum(dim=[-1,-2]).detach().cpu().numpy())
        
        #gt_p = []
        #for p in points:
        #    gt_p.append(len(p))
        #gt_points.extend(gt_p)
        
    mae = mean_absolute_error(pre_count,gt_count)
    mse = np.sqrt(mean_squared_error(pre_count,gt_count))
    nae = mae * len(pre_count) / np.sum(gt_count)

    def count_parameters_in_MB(model):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

    print(f'#Paras: {count_parameters_in_MB(test_net)}')
    print(f'MAE: {mae}, MSE: {mse}, NAE: {nae}')
    print(f'SSIM: {avg_ssim.avg}, PNSR: {avg_pnsr.avg}')