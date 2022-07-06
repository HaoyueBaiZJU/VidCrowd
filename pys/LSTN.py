

from fastai.layers import TimeDistributed
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
#from albumentations.pytorch.transforms import ToTensorV2
#from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob


from tqdm import tqdm

from torchvision import transforms, models
import torch.nn.functional as F
import copy

from utils import visualize, plot_data
from loss.lstn_loss import lstn_loss
from scipy.io import loadmat


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
#density_maps = path + '/dmaps'
sm_train_images = path + '/sm_train_images'
sm_test_images = path + '/sm_test_images'
sm_dmaps = path + '/sm_dmaps'

LOG_PARA = args.log_para

def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
)

def get_train_image_only_transforms():
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=3,p=0.2),
        ],
        additional_targets={'image': 'image'}
    )

# def get_valid_trainsforms():
#     return A.Compose(
#         [
#             #A.Resize(360,640,interpolation=2),
#             A.Normalize(mean=mean,std=std,p=1.0,max_pixel_value=1.0),
#             ToTensorV2(p=1.0),
#         ]
#     )

mean = torch.tensor([0.4939, 0.4794, 0.4583])
std = torch.tensor([0.2177, 0.2134, 0.2144])

def denormalize(img):
    img = img * std[...,None,None] + mean[...,None,None]
    img = img.permute(1,2,0).cpu().numpy()
    return img

class Video_Counting_Dataset(Dataset):
    def __init__(self,path,image_fnames,dmap_folder,
                 seq_len=5,gt_folder=None,suffix='jpg',
                 tfms=None,mosaic=False,
                 crop_size=384,method='train',
                 sample=0,segment=5,num_sample=5,downsample=1):
        super().__init__()
        
        self.path = path
        self.image_fnames = image_fnames
        
        # TODO:
        # Be able to get sequences
        self.image_fnames = sorted(self.image_fnames,key=self._split_fn)
        
        self.crop_size = crop_size
        if method not in ['train','valid']:
            raise Exception('Not Implement')
        self.method = method
        self.LOG_PARA = LOG_PARA
        
        self.dmap_folder = path + dmap_folder
        self.seq_len = seq_len
        self.transforms = tfms
        self.mosaic = mosaic
        self.gt_folder = path + gt_folder # test purpose
        self.sample = sample # 0 is consective, 1 is TSN
        self.segment = segment
        self.num_sample = num_sample
        self.downsample = downsample
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.item_tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,std=self.std),
        ])
        
    def __len__(self):
        return len(self.image_fnames)
    
    def _split_fn(self,f):
        f = f.split('/')[-1].split('.')[0]
        return int(f[3:5]),int(f[-3:])
    
    def __getitem__(self, idx):
        '''
            Get a sequence of frames
            Return: 
                frames, shape: seq, h,w,c
                dmaps, shape: seq, h,w
                gt_points, seq_len of each frame
        '''
        if self.sample: # TSN sampling
            frames,dmaps,fns,gt_points = self._tsn_sampling(idx)
        else:
            frames,dmaps,fns,gt_points = self._consective_sampling(idx)
            
        h,w = frames.shape[-2], frames.shape[-1]
        
        if self.method == 'train':
            i,j = self._random_crop(h,w,self.crop_size,self.crop_size)
            frames = frames[:,:,i:i+self.crop_size,j:j+self.crop_size]
            dmaps = dmaps[:,i:i+self.crop_size,j:j+self.crop_size]
            
#             import pdb
#             pdb.set_trace()
            #for idx in range(len(gt_points)):
            #    gt_points[idx] = [p_l - [j,i] for p_l in gt_points[idx]]
            #    mask = [(p[0]>=0) * (p[0]<self.crop_size) * (p[1]>=0) * (p[1]<self.crop_size) for p in gt_points[idx]]
            #    gt_points[idx] = [gt_p[m] for gt_p, m in zip(gt_points[idx],mask) if m]
            
#             gt_points = gt_points - [j,i]
#             mask = (gt_points[:,0] >=0 ) * (gt_points[:,0] <= self.crop_size) * (gt_points[:,1]>=0) * (gt_points[:,1]<=self.crop_size)
#             gt_points = gt_points[mask]
#             density_map = cv2.resize(density_map,(self.crop_size//self.downsample,self.crop_size//self.downsample))
            dmaps = [cv2.resize(dmap,(self.crop_size//self.downsample,self.crop_size//self.downsample)) for dmap in dmaps]
            dmaps = np.stack(dmaps)
        else:
            dmaps = [cv2.resize(dmap,(w//self.downsample,h//self.downsample)) for dmap in dmaps]
            dmaps = np.stack(dmaps)
        
        frames = torch.from_numpy(frames)
        dmaps = torch.from_numpy(dmaps)
            
        if not isinstance(self.transforms,type(None)):
            t,ch,h,w = frames.shape
            frames = frames.view(t*ch,h,w).permute(1,2,0).numpy()
            dmaps = dmaps.permute(1,2,0).numpy()
            for tfms in self.transforms:
                aug = tfms(**{
                    'image': frames,
                    'mask': dmaps
                })
                frames, dmaps = aug['image'], aug['mask']
                
            frames = torch.from_numpy(frames).permute(2,0,1).view(t,ch,h,w)
            dmaps = torch.from_numpy(dmaps).permute(2,0,1)
        return frames, dmaps*self.LOG_PARA, fns, gt_points
    
    def _tsn_sampling(self,idx):
        '''
        Note:
        This method broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.
        
        Note: 
        Minor changes:
        1) When frames don't have enough to sample, resample at the given region until we reach segments
            pitfall: if last idx is passed, then tsn sampling will duplicate segment times of last frame
        2) offset index when we have enough frames
        
        Args: 
            idx: call by __getitem__
        
        Returns:
            List of frames sampled in tensor
            List of density maps sampled in tensor
            List of file names
            List of point annotation
        '''
        frames, d_maps, fns, gt_points = [],[],[],[]
        length = self.segment * self.num_sample
        start_frame = idx
        
        end_idx = ((idx // 350) + 1) * 350 - 1
        if end_idx > self.__len__() - 1:
            end_idx = self.__len__() - 1
        # Edge case when we don't have enough to sample, result repeated frames
        if start_frame + length > end_idx:
            idxs = np.sort(np.random.randint(start_frame,end_idx+1,self.segment))
        
        # Sample segment times, sampling gap equals to num_sample
        else:
            end_frame = start_frame + length - 1
            idxs = (np.arange(start_frame,end_frame+1,self.num_sample)
                    + np.random.randint(self.num_sample,size=self.segment))
        for idx in idxs:
            fn = self.image_fnames[idx]
            image,dmap,points = self._load_one_frame(fn)
            frames.append(image)
            d_maps.append(dmap)
            fns.append(fn)
            gt_points.append(points)
        return np.stack(frames),np.stack(d_maps),fns,np.array(gt_points)
    
    def _consective_sampling(self,idx):
        '''
        Choose consective frames from given positin idx
        
        Args:
            idx: call by __getitem__
            
        Returns:
            List of frames sampled in tensor
            List of density maps sampled in tensor
            List of file names
            List of point annotation
        '''
        frames, d_maps, fns, gt_points = [],[],[],[]
        
        end_idx = ((idx // 350) + 1) * 350 - 1
        
        frame_diff = end_idx - idx
        if frame_diff >= self.seq_len:
            start_frame = idx
        elif frame_diff < self.seq_len:
            # random back off when sampling dont have enough samples
            idx -= frame_diff
            start_frame = round((idx - frame_diff) * np.random.rand())
        else:
            raise ValueError('start_frame init error...')
        for n in range(idx,idx+self.seq_len):
                fn = self.image_fnames[n]
                image,dmap,points = self._load_one_frame(fn)
                frames.append(image)
                d_maps.append(dmap)
                fns.append(fn)
                gt_points.append(points)
        
        return np.stack(frames),np.stack(d_maps),fns,np.array(gt_points)
    
    def _load_one_frame(self,fn):
        y_fn, p_fn = self._prepare_fn(fn)
        image = cv2.imread(fn)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image/255.
        image = self.item_tfms(image)
        d_map, points = self._get_gts(y_fn,p_fn)
        d_map = torch.from_numpy(d_map)
        return image, d_map, points
    
    def _prepare_fn(self,fn):
        file_name = fn.split('/')[-1].split('.')[0]
        y_fn = self.dmap_folder + '/' + file_name + '.npy'
        p_fn = self.gt_folder + '/' + file_name[3:] + '.mat'
        return y_fn, p_fn
    
    def _get_gts(self,y_fn,p_fn):
        d_map = np.load(y_fn,allow_pickle=True)
        if not self.gt_folder:
            return (None,0)
        test_data = loadmat(p_fn)
        points = test_data['annotation'].astype(int)
        points[:,0] = points[:,0] / 2720 * 1360
        points[:,1] = points[:,1] / 1530 * 784
        return d_map, points
    
    def _random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j
    
    def _normalize(self,img):
        img -= self.mean[None,None,...]
        img /= self.std[None,None,...]
        return img

train_dataset = Video_Counting_Dataset(path=path, image_fnames=glob(sm_train_images+'/*.jpg'),
                                       dmap_folder='/sm_dmap_lstn',
                                       gt_folder='/annotation',
                                       #tfms=[get_train_transforms(),get_train_image_only_transforms()],
                                       sample=1, method='valid',downsample=args.downsample
)
valid_dataset = Video_Counting_Dataset(path=path, image_fnames=glob(sm_test_images+'/*.jpg'),
                                       dmap_folder='/sm_dmap_lstn',
                                       gt_folder='/annotation',
                                       sample=1, method='valid',downsample=args.downsample #, segment=1
)

imgs,dmaps,fns,gt_points = train_dataset[0]

imgs.shape

class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.n_epochs 
    lr = 0.0002

    folder = 'LSTN-8.7-NoCrop'
    downsample = args.downsample
    split_num = 2

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


# # Model

import torch.nn.functional as F
from torchvision import models


class LSTN(nn.Module):
    def __init__(self, input_size=(360, 640), h_blocks=1, w_blocks=2):
        super(LSTN, self).__init__()
        self.h_blocks = h_blocks
        self.w_blocks = w_blocks
        h_size = int(input_size[0]/4/self.h_blocks)
        w_size = int(input_size[1]/4/self.w_blocks)

        self.vgg16 = VGG16()
        self.stn = STN((h_size, w_size))
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: frame t with size (B, 3, 360, 640)
        :return:
                map_t0: density map at time t0 from VGG-16 with size (B, 1, 90, 160)
                map_t1_blocks: a list of density map blocks at time t1 with size (B, 1, 90/H, 160/W) for each block
                               for Mall dataset H=1 W=2, 2 blocks, then (B, 1, 90, 80)
                               for UCSD dataset H=2 W=2, 4 blocks, then (B, 1, 45, 80)
        """
        map_t0 = self.vgg16(x)
        map_t1_blocks = []

        h_chunks = torch.chunk(map_t0, self.h_blocks, dim=2)
        for cnk in h_chunks:
            cnks = torch.chunk(cnk, self.w_blocks, dim=3)
            for c_ in cnks:
                t1_block = self.stn(c_)
                map_t1_blocks.append(t1_block)
        map_t1_blocks = torch.stack(map_t1_blocks, dim=-1)
        return map_t0, map_t1_blocks


class STN(nn.Module):
    def __init__(self, h_w_size=(90, 80)):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # calculate the input size for linear layer
        h = (h_w_size[0] - 7) + 1
        h = int((h-2)/2)+1
        h = (h - 5) + 1
        h = int((h - 2) / 2) + 1

        w = (h_w_size[1] - 7) + 1
        w = int((w-2)/2)+1
        w = (w - 5) + 1
        w = int((w - 2) / 2) + 1

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * h * w, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: block of output from VGG16 at time t with size (B, 1, 90, 80) or (B, 1, 45, 80)
                 for Mall dataset H=1 W=2, 2 blocks, then (B, 1, 90, 80)
                 for UCSD dataset H=2 W=2, 4 blocks, then (B, 1, 45, 80)
        :return: output size is the same to input size
        """
        xs = self.localization(x)
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class VGG16(nn.Module):
    def __init__(self, load_weights=False, fix_weights=True):
        super(VGG16, self).__init__()
        # Two M layer pulled out from original vgg16.
        # Last three layer in self.cfg are additional to the original vgg16 first 13 layers.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 128, 64]
        self.layers = make_layers(self.cfg)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            set_parameter_requires_grad(self.layers[0:22], fix_weights)  # fix weights for the first ten layers
            self.layers[0:16].load_state_dict(mod.features[0:16].state_dict())
            for i in [16, 18, 20]:
                self.layers[i].load_state_dict(mod.features[i+1].state_dict())
                #self.layers[i].weight.copy_(mod.features[i+1].weight)
                #self.layers[i].bias.copy_(mod.features[i+1].bias)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: frame t with size (B, 3, 360, 640)
        :return: output size (B, 1, 90, 160)
        """
        x = self.layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
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


# # Train

def SummaryLoss_LSTN(preds_t0, preds_t1_blocks, gts, imgs):
    return lstn_loss(preds_t0, preds_t1_blocks, gts, imgs, lamda=0.001, beta=30)

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
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            #self.model = nn.DataParallel(self.model)
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
        self.criterion = SummaryLoss_LSTN
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

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, mae_loss: {mae_loss.avg:.8f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, mae_loss = self.validation(validation_loader)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.8f}, time: {(time.time() - t):.5f}')
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
                        f'summary_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
                

                #preds = self.model(images)
            with torch.no_grad():
                batch_size = images.shape[0]
                h_split,w_split = (
                    images.shape[-2]//self.config.split_num,
                    images.shape[-1]//self.config.split_num #bs,T,c,h,w
                )
                #images = images.to(self.device).float()
                images = images.cuda()
                #images = images.transpose(1,2)
                density_maps = density_maps.cuda()
                #density_maps = density_maps.to(self.device).float()
                
                
                
                with torch.cuda.amp.autocast(): #native fp16
                    ### added
                    #bs,t,c,h,w = images.shape
                    preds_t0, preds_t1_blocks = self.model(images)
                    preds_t1_blocks = torch.unbind(preds_t1_blocks, dim=-1)
                    #preds = preds.view(bs,t,h//self.config.downsample,w//self.config.downsample)
                    
#                     preds = torch.zeros([bs,t,h,w])
#                     for r in range(self.config.split_num):
#                         for c in range(self.config.split_num):
#                             preds[:,:,h_split*r:h_split*(r+1),w_split*c:w_split*(c+1)] = (
#                                 self.model(images[:,:,:,h_split*r:h_split*(r+1),w_split*c:w_split*(c+1)]).view(8,1,392,-1)#.view(4,1,765,-1)
#                             )
                    
                    loss = self.criterion(preds_t0, preds_t1_blocks, density_maps, images)
                    metric_loss = self.metric(preds_t0.squeeze(),density_maps,self.config.downsample)
                mae_loss.update(metric_loss.detach().item(),batch_size)
                summary_loss.update(loss.detach().item(), batch_size)
                
            #if step == 10:
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
                        f'summary_loss: {summary_loss.avg:.8f}, ' + \
                        f'mae_loss: {mae_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            #images = images.to(self.device).float()
            images = images.cuda().float()
            #images = images.transpose(1,2)
            batch_size = images.shape[0]
            #density_maps = density_maps.to(self.device).float()
            density_maps = density_maps.cuda().float()
            
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(): #native fp16
                #print(images.shape)
                preds_t0, preds_t1_blocks = self.model(images)
                preds_t1_blocks = torch.unbind(preds_t1_blocks, dim=-1)
                loss = self.criterion(preds_t0, preds_t1_blocks, density_maps, images)
                metric_loss = self.metric(preds_t0.squeeze().cpu().detach(),density_maps.cpu().detach(),self.config.downsample)
            self.scaler.scale(loss).backward()
            
            # loss = loss / self.iters_to_accumulate # gradient accumulation
            
#             with amp.scale_loss(loss,self.optimizer) as scaled_loss: # apex
#                 scaled_loss.backward()
            #loss.backward()

            
            mae_loss.update(metric_loss.detach().item(),batch_size)
            summary_loss.update(loss.cpu().detach().item(), batch_size)
            
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
            #if step == 10:
            #    break

        return summary_loss, mae_loss
    
    def save(self, path):
        #self.model.cpu()
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
    frames, dmaps, fns, gt_points = zip(*batch)
    return torch.stack(frames), torch.stack(dmaps), fns, gt_points

def run_training():
    device = torch.device('cuda:0')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        #sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size//4,
        num_workers=TrainGlobalConfig.num_workers//2,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        #sampler=val_sampler,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    #fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)

net = LSTN(input_size=(784,1360), h_blocks=2, w_blocks=2).cuda()
net = nn.DataParallel(net)
net = TimeDistributed(net)

if args.mode == 'train':
    run_training()
else:
    test_net = LSTN(input_size=(784,1360), h_blocks=2, w_blocks=2).cuda()
    test_net = nn.DataParallel(test_net)
    test_net = TimeDistributed(test_net)

    checkpoint = torch.load(f'/mnt/home/zpengac/USERDIR/count/drone_benchmark/LSTN-8.6-NoCrop2/best-checkpoint-012epoch.bin')
    test_net.load_state_dict(checkpoint['model_state_dict'])
    test_net.eval()

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as pnsr

    pre_count = []
    gt_count = []
    gt_points = []
    avg_ssim = AverageMeter()
    avg_pnsr = AverageMeter()
    for step, (imgs, dmapss, fns, points) in enumerate(val_loader):
        with torch.no_grad():
            imgs = imgs.cuda().float()
            predss, _ = test_net(imgs)
            predss = predss / LOG_PARA * (TrainGlobalConfig.downsample**2)
        dmapss = dmapss / LOG_PARA * (TrainGlobalConfig.downsample**2)
        
        for i in range(5):
            preds = predss[:,i,...]
            dmaps = dmapss[:,i,...]
            for pred, dmap in zip(preds, dmaps):
                pred_array = pred.detach().cpu().numpy().squeeze()
                dmap_array = dmap.detach().cpu().numpy().squeeze()
                avg_ssim.update(ssim(dmap_array, pred_array, data_range=dmap_array.max()-dmap_array.min()))
                avg_pnsr.update(pnsr(dmap_array, pred_array, data_range=dmap_array.max()-dmap_array.min()))
        
        pre_count.extend(predss.sum(dim=[-1,-2]).detach().cpu().numpy())
        gt_count.extend(dmapss.sum(dim=[-1,-2]).detach().cpu().numpy())
        
        gt_p = []
        for p in points:
            gt_p.append(len(p))
        gt_points.extend(gt_p)

    pre_count_new = [pre.reshape(-1) for pre in pre_count]
    pre_count_new = np.concatenate(pre_count_new)
    gt_count_new = [gt.reshape(-1) for gt in gt_count]
    gt_count_new = np.concatenate(gt_count_new)

    mae = mean_absolute_error(pre_count_new,gt_count_new)
    mse = mean_squared_error(pre_count_new,gt_count_new)
    nae = mae * len(pre_count_new) / np.sum(gt_count_new)

    def count_parameters_in_MB(model):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

    print(f'#Paras: {count_parameters_in_MB(test_net)}')
    print(f'MAE: {mae}, MSE: {mse}, NAE: {nae}')
    print(f'SSIM: {avg_ssim.avg}, PNSR: {avg_pnsr.avg}')