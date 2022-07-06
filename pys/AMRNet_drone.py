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

path = '/mnt/home/zpengac/USERDIR/count/drone_dataset'
#train_images = path + '/images'
#test_images = path + '/test_images/images'
#anno = path + '/annotation'
images = path + '/rf_image_vehicle'
density_maps = path + '/rf_GT_vehicle'

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
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                           contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.Blur(blur_limit=3,p=0.2),
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

#mean = torch.tensor([0.4939, 0.4794, 0.4583])
#std = torch.tensor([0.2177, 0.2134, 0.2144])
mean = torch.tensor([0.38868062, 0.38568735, 0.39457315])
std = torch.tensor([0.221865, 0.23096273, 0.2210397])

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
        if image.shape[0] < 448:
            pad_h = (448 - image.shape[0]) // 2
        if image.shape[1] < 448:
            pad_w = (448 - image.shape[1]) // 2
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
        density_map = density_map.squeeze()
        h,w = image.shape[0], image.shape[1]
        #image = cv2.resize(image,(w, h))
        
        if self.method == 'train':
            #h,w = image.shape[:2]
            i,j = self._random_crop(h,w,self.crop_size,self.crop_size)
            image = image[i:i+self.crop_size,j:j+self.crop_size]
            density_map = density_map[i:i+self.crop_size,j:j+self.crop_size]
            #print(density_map.shape)
            #gt_points = gt_points - [j,i]
            #mask = (gt_points[:,0] >=0 ) * (gt_points[:,0] <= self.crop_size) * (gt_points[:,1]>=0) * (gt_points[:,1]<=self.crop_size)
            #gt_points = gt_points[mask]
            density_map = cv2.resize(density_map,(self.crop_size//self.downsample,self.crop_size//self.downsample))
            
        else:
            density_map = cv2.resize(density_map,(w//self.downsample,h//self.downsample))#,interpolation=cv2.INTER_NEAREST)
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

fp = glob(images + '/*.jpg')

split = int(len(fp) * 0.8)
fp[0:split][:10]

train_dataset = Crop_Dataset(path=path,
                             image_fnames=fp[:split],dmap_folder='/rf_GT_vehicle',
                             #gt_folder='/annotation',
                             transforms=[get_train_transforms(),get_train_image_only_transforms()],
                             downsample=args.downsample,
                             crop_size=args.crop_size
                                )

valid_dataset = Crop_Dataset(path=path,
                             image_fnames=fp[split:],dmap_folder='/rf_GT_vehicle',
                             #gt_folder='/annotation',
                             transforms=[get_valid_trainsforms()],
                             method='valid',
                             downsample=args.downsample
                             #crop_size=args.crop_size
                                )

img, dmap, fn, pt = train_dataset[0]

class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.n_epochs 
    lr = 0.0002

    folder = 'AMRNet-drone_vehicle-9.29-448'
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

import torch.nn.functional as F
from torchvision import models

class VGG16_LCM(nn.Module):
    def __init__(self, load_weights=True):
        super(VGG16_LCM, self).__init__()

        self.layer5 = self.VGG_make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                                            512, 512, 512, 'M', 512, 512, 512, 'M'])

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.AvgPool2d(2, 2),
        )

        if load_weights:
            mod = models.vgg16(pretrained=False)
            pretrain_path = './vgg16-397923af.pth'
            mod.load_state_dict(torch.load(pretrain_path))
            print("loaded pretrain model: " + pretrain_path)

            self._initialize_weights()
            self.layer5.load_state_dict(mod.features[0:31].state_dict())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    @torch.cuda.amp.autocast()
    def forward(self, x):
        x = self.layer5(x)
        x = self.reg_layer(x)

        return x

    def VGG_make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=1):
        d_rate = dilation
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

def VGG_make_layers(cfg, in_channels=3, batch_norm=False, dilation=1):
        d_rate = dilation
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
vgg = VGG_make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                                            512, 512, 512, 'M', 512, 512, 512, 'M'])

from collections import OrderedDict
class VGG16_LCM_REG(nn.Module):
    def __init__(self, load_weights=False, stage_num=[3,3,3], count_range=100, lambda_i=1., lambda_k=1.):
        super(VGG16_LCM_REG, self).__init__()

        # cfg
        self.stage_num = stage_num
        self.lambda_i = lambda_i
        self.lambda_k = lambda_k
        self.count_range = count_range
        self.multi_fuse = True
        self.soft_interval = True

        self.layer3 = self.VGG_make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512])
        self.layer4 = self.VGG_make_layers(['M', 512, 512, 512], in_channels=512)
        self.layer5 = self.VGG_make_layers(['M', 512, 512, 512], in_channels=512)

        if self.multi_fuse:
            self.fuse_layer5 = DC_layer(level=0)
            self.fuse_layer4 = DC_layer(level=1)
            self.fuse_layer3 = DC_layer(level=2)

        self.count_layer5 = Count_layer(pool=2)
        self.count_layer4 = Count_layer(pool=4)
        self.count_layer3 = Count_layer(pool=8)
        
        if self.soft_interval:
            self.layer5_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )
            self.layer4_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )
            self.layer3_k = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1),
                nn.Tanh(),
            )
        
            self.layer5_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[0], kernel_size=1),
                nn.Sigmoid(),
            )
            self.layer4_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[1], kernel_size=1),
                nn.Sigmoid(),
            )
            self.layer3_i = nn.Sequential(
                nn.Conv2d(512, self.stage_num[2], kernel_size=1),
                nn.Sigmoid(),
            )

        self.layer5_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[0], kernel_size=1),
            nn.ReLU(),
        )
        self.layer4_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[1], kernel_size=1),
            nn.ReLU(),
        )
        self.layer3_p = nn.Sequential(
            nn.Conv2d(512, self.stage_num[2], kernel_size=1),
            nn.ReLU(),
        )

        if load_weights:
            #self._initialize_weights()
            
            mod = models.vgg16(pretrained=False)
            pretrain_path = './vgg16-397923af.pth'
            mod.load_state_dict(torch.load(pretrain_path))

            new_state_dict = OrderedDict()
            for key, params in mod.features[0:23].state_dict().items():
                new_state_dict[key] = params
            self.layer3.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            for key, params in mod.features[23:30].state_dict().items():
                key = str(int(key[:2]) - 23) + key[2:]
                new_state_dict[key] = params
            self.layer4.load_state_dict(new_state_dict)

            new_state_dict = OrderedDict()
            for key, params in mod.features[23:30].state_dict().items():
                key = str(int(key[:2]) - 23) + key[2:]
                new_state_dict[key] = params
            self.layer5.load_state_dict(new_state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        if self.multi_fuse:
            x5 = self.fuse_layer5(x5)
            x4 = self.fuse_layer4(x4)
            x3 = self.fuse_layer3(x3)

        x5_= self.count_layer5(x5)
        p5 = self.layer5_p(x5_)
        if self.soft_interval:
            k5 = self.layer5_k(x5_)
            i5 = self.layer5_i(x5_)

        x4_ = self.count_layer4(x4)
        p4 = self.layer4_p(x4_)
        if self.soft_interval:
            k4 = self.layer4_k(x4_)
            i4 = self.layer4_i(x4_)

        x3_ = self.count_layer3(x3)
        p3 = self.layer3_p(x3_)
        if self.soft_interval:
            k3 = self.layer3_k(x3_)
            i3 = self.layer3_i(x3_)

        stage1_regress = p5[:, 0, :, :] * 0
        stage2_regress = p4[:, 0, :, :] * 0
        stage3_regress = p3[:, 0, :, :] * 0

        for index in range(self.stage_num[0]):
            if self.soft_interval:
                stage1_regress = stage1_regress + (float(index) + self.lambda_i * i5[:, index, :, :]) * p5[:, index, :, :]
            else:
                stage1_regress = stage1_regress + float(index) * p5[:, index, :, :]
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        if self.soft_interval:
            stage1_regress = stage1_regress / ( float(self.stage_num[0]) * (1. + self.lambda_k * k5) )
        else:
            stage1_regress = stage1_regress / float(self.stage_num[0])


        for index in range(self.stage_num[1]):
            if self.soft_interval:
                stage2_regress = stage2_regress + (float(index) + self.lambda_i * i4[:, index, :, :]) * p4[:, index, :, :]
            else:
                stage2_regress = stage2_regress + float(index) * p4[:, index, :, :]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        if self.soft_interval:
            stage2_regress = stage2_regress / ( (float(self.stage_num[0]) * (1. + self.lambda_k * k5)) *
                                                (float(self.stage_num[1]) * (1. + self.lambda_k * k4)) )
        else:
            stage2_regress = stage2_regress / float( self.stage_num[0] * self.stage_num[1] )


        for index in range(self.stage_num[2]):
            if self.soft_interval:
                stage3_regress = stage3_regress + (float(index) + self.lambda_i * i3[:, index, :, :]) * p3[:, index, :, :]
            else:
                stage3_regress = stage3_regress + float(index) * p3[:, index, :, :]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        if self.soft_interval:
            stage3_regress = stage3_regress / ( (float(self.stage_num[0]) * (1. + self.lambda_k * k5)) *
                                                (float(self.stage_num[1]) * (1. + self.lambda_k * k4)) *
                                                (float(self.stage_num[2]) * (1. + self.lambda_k * k3)) )
        else:
            stage3_regress = stage3_regress / float( self.stage_num[0] * self.stage_num[1] * self.stage_num[2] )

        # regress_count = stage1_regress * self.count_range
        # regress_count = (stage1_regress + stage2_regress) * self.count_range
        regress_count = (stage1_regress + stage2_regress + stage3_regress) * self.count_range

        return regress_count

    def VGG_make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=1):
        d_rate = dilation
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class Count_layer(nn.Module):
    def __init__(self, inplanes=512, pool=2):
        super(Count_layer, self).__init__()
        self.avgpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((pool, pool), stride=pool),
        )
        self.maxpool_layer = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((pool, pool), stride=pool),
        )
        self.conv1x1= nn.Sequential(
            nn.Conv2d(inplanes*2, inplanes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_avg = self.avgpool_layer(x)
        x_max = self.maxpool_layer(x)

        x = torch.cat([x_avg, x_max], dim=1)
        x = self.conv1x1(x)
        return x


class DC_layer(nn.Module):
    def __init__(self, level, fuse=False):
        super(DC_layer, self).__init__()
        self.level = level
        self.conv1x1_d1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d3 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_d4 = nn.Conv2d(512, 512, kernel_size=1)

        self.conv_d1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2)
        self.conv_d3 = nn.Conv2d(512, 512, kernel_size=3, padding=3, dilation=3)
        self.conv_d4 = nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4)
        
        self.fuse = fuse
        if self.fuse:
            self.fuse = nn.Conv2d(512*2, 512, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1x1_d1(x)
        x2 = self.conv1x1_d2(x)
        x3 = self.conv1x1_d3(x)
        x4 = self.conv1x1_d4(x)

        x1 = self.conv_d1(x1)
        x2 = self.conv_d2(x2)
        x3 = self.conv_d3(x3)
        x4 = self.conv_d4(x4)

        # x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = self.relu(self.fuse(x))
        x = Maxout(x1, x2, x3, x4)
        return x

def Maxout(x1, x2, x3, x4):
    mask_1 = torch.ge(x1, x2)
    mask_1 = mask_1.float()
    x = mask_1 * x1 + (1-mask_1) * x2

    mask_2 = torch.ge(x, x3)
    mask_2 = mask_2.float()
    x = mask_2 * x + (1-mask_2) * x3

    mask_3 = torch.ge(x, x4)
    mask_3 = mask_3.float()
    x = mask_3 * x + (1-mask_3) * x4
    return x

def MSELoss_MCNN(preds,targs):
    return nn.MSELoss()(preds,targs)

def MAELoss_MCNN(preds,targs,upsample):
    return nn.L1Loss()((preds).sum(dim=[-1,-2])*upsample*upsample, (targs/LOG_PARA).sum(dim=[-1,-2])*upsample*upsample)

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

import warnings
warnings.filterwarnings("ignore")

#opt_level ='O1' # apex

class Fitter:
    
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'/mnt/home/zpengac/USERDIR/count/drone_benchmark/{config.folder}'
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
                    # Construct the local counting map proposed in the paper
                    kernel6 = 64
                    filter6 = torch.ones(1, 1, kernel6, kernel6, requires_grad=False)
                    density_maps = F.conv2d(density_maps, filter6.cuda(), stride=kernel6)
                    loss = self.criterion(preds,density_maps/LOG_PARA)
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
                # Construct the local counting map proposed in the paper
                kernel6 = 64
                filter6 = torch.ones(1, 1, kernel6, kernel6, requires_grad=False)
                density_maps = F.conv2d(density_maps, filter6.cuda(), stride=kernel6).cuda()
                loss = self.criterion(preds,density_maps/LOG_PARA)
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
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1, #TrainGlobalConfig.batch_size,
        num_workers=1, #TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    #fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)

net = VGG16_LCM_REG().cuda()
net = nn.DataParallel(net)

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

    test_net = VGG16_LCM_REG().cuda()
    test_net = nn.DataParallel(test_net)

    checkpoint = torch.load(f'/mnt/home/zpengac/USERDIR/count/drone_benchmark/cps/smartcity/AMRNet-9.9-smartcity-6/best-checkpoint-031epoch.bin')
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
    kernel6 = 64
    for step, (imgs, dmaps, fns, points) in enumerate(val_loader):
        #with torch.cuda.amp.autocast():
        with torch.no_grad():
            imgs = imgs.cuda().float()
            preds = test_net(imgs)
        dmaps = dmaps / LOG_PARA

        filter6 = torch.ones(1, 1, kernel6, kernel6, requires_grad=False).float()
        dmaps = F.conv2d(dmaps.float(), filter6, stride=kernel6)
        
        for pred, dmap in zip(preds, dmaps):
            pred_array = pred.detach().cpu().numpy().squeeze()
            dmap_array = dmap.detach().cpu().numpy().squeeze()
            avg_ssim.update(ssim(dmap_array, pred_array, data_range=dmap_array.max()-dmap_array.min(), win_size=5))
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