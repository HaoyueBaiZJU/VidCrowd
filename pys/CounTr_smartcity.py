import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int,default=-1)
parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
opt = parser.parse_args()
local_rank = opt.local_rank
print("local rank {}".format(local_rank))
assert torch.cuda.device_count() > opt.local_rank
torch.cuda.set_device(opt.local_rank)
device = torch.device('cuda', opt.local_rank)
dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
opt.world_size = dist.get_world_size()
print("world size {}".format(opt.world_size))
print("get rank {}".format(dist.get_rank()))

from counTR import CounTR

import torch.nn as nn
import torch.nn.functional as F
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


# import torch.multiprocessing as mp
# from torch.nn import DataParallel

from glob import glob
from scipy.io import loadmat


import timm


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


path = '/mnt/home/zpengac/USERDIR/count/SmartCity'
images = path + '/images'
anno = path + '/ground_truth'
density_maps = path + '/density_maps'

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
#             A.OneOf([
#                 A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
#                                      val_shift_limit=0.2, p=0.9),
#                 A.RandomBrightnessContrast(brightness_limit=0.2, 
#                                            contrast_limit=0.2, p=0.9),
#             ],p=0.9),
#             A.Blur(blur_limit=3,p=0.2),
            A.Normalize(p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ],
        additional_targets={'image': 'image'}
    )

def get_valid_trainsforms():
    return A.Compose(
        [
            #A.Resize(360,640,interpolation=2),
            A.Normalize(p=1.0,max_pixel_value=1.0),
            ToTensorV2(p=1.0),
        ]
    )

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
        if image.shape[0] < 1080:
            padding = (1080 - image.shape[0]) // 2
            image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            d_map = cv2.copyMakeBorder(d_map, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
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
        points = test_data['loc'].astype(int)
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
                             image_fnames=fp[:split],dmap_folder='/density_maps',
                             gt_folder='/ground_truth',
                             transforms=[get_train_transforms(),get_train_image_only_transforms()],
                             downsample=args.downsample,
                             crop_size=args.crop_size
                                )

valid_dataset = Crop_Dataset(path=path,
                             image_fnames=fp[split:],dmap_folder='/density_maps',
                             gt_folder='/ground_truth',
                             transforms=[get_valid_trainsforms()],
                             method='valid',
                             downsample=args.downsample,
                             crop_size=args.crop_size
                                )

word_size = dist.get_world_size()
#if opt.distribute and opt.local_rank != -1:
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas = word_size,rank = opt.local_rank)
valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,num_replicas = word_size,rank = opt.local_rank)
#else:
#    train_sampler = None

backbone = nn.Sequential(*list(timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0).children())[:-2])
act_cls = nn.ReLU
idx = [1,3,21,23]


def get_pad(im_sz, crop_sz=448,**kwargs):
    h,w = im_sz
    h_mul = h // crop_sz + 1
    w_mul = w // crop_sz + 1
    pad_h = (crop_sz * h_mul - h) % crop_sz
    pad_w = (crop_sz * w_mul - w) % crop_sz
    assert pad_h%2 == 0
    assert pad_w%2 == 0
    return pad_h//2, pad_w//2


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
        if opt.distribute and opt.local_rank != -1:
            self.model.to(device)
            self.model = DDP(self.model, device_ids=[opt.local_rank], output_device=opt.local_rank)
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
            train_sampler.set_epoch(e)
            if self.config.verbose and dist.get_rank()== 0:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, mae_loss = self.train_one_epoch(train_loader)
            
            if dist.get_rank()== 0:
                self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
                self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mae_loss: {mae_loss.avg:.8f}, time: {(time.time() - t):.5f}')
                self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, mae_loss = self.validation(validation_loader)
            
            if dist.get_rank()== 0:
                self.log(f'[RESULT]: Val. Epoch: {self.epoch}, mse_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
                self.log(f'[RESULT]: Val. Epoch: {self.epoch}, mae_loss: {mae_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                if dist.get_rank()== 0:
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
                if step % self.config.verbose_step == 0 and dist.get_rank()== 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'mse_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                batch_size = images.shape[0]
                #images = images.to(self.device).float()
                images = images.cuda().float()
                #density_maps = density_maps.to(self.device).float()
                density_maps = density_maps.cuda().float()
                
                if pad_mode == True:
                    _, _, h, w = images.shape
                    pad_h, pad_w = get_pad((h, w),crop_size)
                    images = F.pad(images,(pad_w,pad_w,pad_h-1,pad_h-1))
                    density_maps = F.pad(density_maps,(pad_w,pad_w,pad_h,pad_h))
                    bs, ch, h, w = images.shape
                    m,n = int(h/crop_size), int(w/crop_size)
                    loss,metric_loss = 0, 0
                    
                
                with torch.cuda.amp.autocast(): #native fp16
                    for i in range(m):
                        for j in range(n):
                            img_patches = images[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                            dmaps_patches = density_maps[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                            preds = self.model(img_patches)
                            loss += self.criterion(preds,dmaps_patches)
                            metric_loss += self.metric(preds,dmaps_patches,self.config.downsample)
                mae_loss.update(metric_loss.detach().item(),batch_size)
                summary_loss.update(loss.detach().item(), batch_size)
            
#             if step == 20:
#                 break

        return summary_loss, mae_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        mae_loss = AverageMeter()
        t = time.time()
        for step, (images, density_maps, fns, gt_pts) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0 and dist.get_rank()== 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'mse_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            #images = images.to(self.device).float()
            images = images.cuda().float()
            batch_size = images.shape[0]
            #density_maps = density_maps.to(self.device).float()
            density_maps = density_maps.cuda().float()
            
            
            self.optimizer.zero_grad()
            
            
#             with torch.cuda.amp.autocast(): #native fp16
#                 preds = self.model(images)
#                 loss = self.criterion(preds,density_maps)
#                 metric_loss = self.metric(preds.detach(),density_maps.detach(),self.config.downsample)

            if pad_mode == True:
                #images = F.pad(images,(pad_w,pad_w,pad_h-1,pad_h-1))
                #density_maps = F.pad(density_maps,(pad_w,pad_w,pad_h,pad_h))
                #bs, ch, h, w = images.shape
                images = F.pad(images,(t_pad_w,t_pad_w,t_pad_h,t_pad_h))
                density_maps = F.pad(density_maps, (t_pad_w,t_pad_w,t_pad_h,t_pad_h))
                m,n = int(448/crop_size), int(448/crop_size)
                loss,metric_loss = 0, 0
            with torch.cuda.amp.autocast(): #native fp16
                for i in range(m):
                    for j in range(n):
                        img_patches = images[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                        dmaps_patches = density_maps[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                        preds = self.model(img_patches)
                        loss += self.criterion(preds,dmaps_patches)
                        metric_loss += self.metric(preds,dmaps_patches,self.config.downsample)
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
            
            
pad_mode = True
crop_size = 224
#pad_h, pad_w = get_pad((1530,2720),crop_size)
t_pad_h,t_pad_w = get_pad((448,448),crop_size)


class TrainGlobalConfig:
    num_workers =4
    batch_size = args.batch_size
    n_epochs = args.n_epochs 
    lr = 0.0002

    folder = 'counTR-9.9-smartcity'
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
        steps_per_epoch=int(len(train_dataset) / (batch_size*torch.cuda.device_count())),
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
net = CounTR(backbone,act_cls=act_cls,
             imsize=224,layer_idx=idx,
             self_attention=False).cuda()

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
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1,
        num_workers=1,
        shuffle=False,
        sampler=valid_sampler,#SequentialSampler(valid_dataset),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
#     fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)

if args.mode == 'train':
    run_training()
else:
    # Comment all the code below when training
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=1,
        num_workers=1,
        shuffle=False,
        #sampler=SequentialSampler(valid_dataset),
        sampler=valid_sampler,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_net = net

    checkpoint = torch.load(f'/mnt/home/zpengac/USERDIR/count/drone_benchmark/counTR-9.1-smartcity-5/best-checkpoint-016epoch.bin')
    test_net.load_state_dict(checkpoint['model_state_dict'])
    test_net.eval()

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from tqdm import tqdm
    #from skimage.metrics import structural_similarity as ssim
    #from skimage.metrics import peak_signal_noise_ratio as pnsr

    pre_count = []
    gt_count = []
    gt_points = []
    #avg_ssim = AverageMeter()
    #avg_pnsr = AverageMeter()
    for images, density_maps, fns, points in tqdm(val_loader):
        batch_size = images.shape[0]
        #images = images.to(self.device).float()
        images = images.cuda().float()
        #density_maps = density_maps.to(self.device).float()
        density_maps = density_maps.cuda().float()

        _, _, h, w = images.shape
        pad_h, pad_w = get_pad((h, w),crop_size)
        images = F.pad(images,(pad_w,pad_w,pad_h,pad_h))
        density_maps = F.pad(density_maps,(pad_w,pad_w,pad_h,pad_h))
        bs, ch, h, w = images.shape
        m,n = int(h/crop_size), int(w/crop_size)
        predc, dmapc = 0, 0

        with torch.cuda.amp.autocast(): #native fp16
            for i in range(m):
                for j in range(n):
                    img_patches = images[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                    dmaps_patches = density_maps[:,:,crop_size*i:crop_size*(i+1),crop_size*j:crop_size*(j+1)]
                    preds = test_net(img_patches)
                    predc += preds.sum(dim=[-1,-2]).detach().cpu().numpy() / LOG_PARA
                    dmapc += dmaps_patches.sum(dim=[-1,-2]).detach().cpu().numpy() / LOG_PARA
                    #print(preds.shape, dmaps_patches.shape)
        
        pre_count.extend(predc)
        
        gt_count.extend(dmapc)
        
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
    #print(f'SSIM: {avg_ssim.avg}, PNSR: {avg_pnsr.avg}')
