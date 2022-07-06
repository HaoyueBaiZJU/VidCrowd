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

from torchvision import transforms

from utils import visualize, plot_data
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
sm_train_images = path + '/sz_392_train_images'
sm_test_images = path + '/sz_392_test_images'
sm_dmaps = path + '/sz_392_dmaps'

LOG_PARA = args.log_para


def _split_fn(f):
    f = f.split('/')[-1].split('.')[0]
    return int(f[:2]),int(f[-3:])


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
                 sample=0,segment=5,num_sample=5):
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
            
            gt_points = np.array([p_l - [j,i] for p_l in gt_points])
            mask = [(p[:,0]>=0) * (p[:,0]<=384) * (p[:,1]>=0) * (p[:,1]<=384) for p in gt_points]
            gt_points = [gt_p[m] for gt_p, m in zip(gt_points,mask)]
#             gt_points = gt_points - [j,i]
#             mask = (gt_points[:,0] >=0 ) * (gt_points[:,0] <= self.crop_size) * (gt_points[:,1]>=0) * (gt_points[:,1]<=self.crop_size)
#             gt_points = gt_points[mask]
#             density_map = cv2.resize(density_map,(self.crop_size//self.downsample,self.crop_size//self.downsample))
            
            
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
        return torch.stack(frames),torch.stack(d_maps),fns,np.array(gt_points)
    
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
        elif frame_dff < self.seq_len:
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
        
        return torch.stack(frames),torch.stack(d_maps),fns,np.array(gt_points)
    
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
                                       dmap_folder='/sz_392_dmaps',
                                       gt_folder='/annotation',
                                       tfms=[get_train_transforms(),get_train_image_only_transforms()],
                                       sample=1, crop_size=args.crop_size,method='valid'
)
valid_dataset = Video_Counting_Dataset(path=path, image_fnames=glob(sm_test_images+'/*.jpg'),
                                       dmap_folder='/sz_392_dmaps',
                                       gt_folder='/annotation',
                                       sample=1, method='valid', segment=1
)


mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
def denoramlize(img):
    return img * std[...,None,None] + mean[...,None,None]


class TrainGlobalConfig:
    num_workers = args.num_workers
    batch_size = args.batch_size
    n_epochs = args.n_epochs 
    lr = 0.0002

    folder = 'sm_conv_lstm_7_20_392_680'
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


"""
ConvLSTM for video-based crowd counting
Reference: Spatiotemporal Modeling for Crowd Counting in Videos, ICCV2017
Link: https://openaccess.thecvf.com/content_ICCV_2017/papers/Xiong_Spatiotemporal_Modeling_for_ICCV_2017_paper.pdf
"""

import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    @torch.cuda.amp.autocast()
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        load_weights = True
        hdim=64

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=3, # original 3
                                            hidden_dim=128, #64
                                            kernel_size=(3, 3),
                                            bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=128, #64
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)
                                        
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=hdim,
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=hdim,
                                            hidden_dim=hdim,
                                            kernel_size=(3, 3),
                                            bias=True)
        self.decoder_CNN = nn.Conv3d(in_channels=hdim, # hdim
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

 
    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        #outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def autoencoder2(self, x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):
        outputs = []
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=h_t2,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs
    
    @torch.cuda.amp.autocast()
    def forward(self,x):
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        #future_seq = 3
        outputs = self.autoencoder2(x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)
        
        #outputs = torch.stack(x, 1)
        #outputs = x.permute(0, 2, 1, 3, 4)
        #outputs = self.decoder_CNN(outputs)

        #outputs = F.upsample(outputs, scale_factor=8)
#         output1 = F.upsample(outputs[:,:,0,:,:], scale_factor=8)
#         output2 = F.upsample(outputs[:,:,1,:,:], scale_factor=8)
#         output3 = F.upsample(outputs[:,:,2,:,:], scale_factor=8)
#         output = torch.stack((output1, output2, output3), 1)

        return outputs.squeeze()

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

        self.base_dir = f'/mnt/home/hheat/USERDIR/counting-bench/model/{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = nn.DataParallel(self.model)
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
            
                

                #preds = self.model(images)
            with torch.no_grad():
                batch_size = images.shape[0]
                h_split,w_split = (
                    images.shape[-2]//self.config.split_num,
                    images.shape[-1]//self.config.split_num #bs,T,c,h,w
                )
                #images = images.to(self.device).float()
                images = images.cuda()
                density_maps = density_maps.cuda()
                #density_maps = density_maps.to(self.device).float()
                
                
                
                with torch.cuda.amp.autocast(): #native fp16
                    ### added
                    bs,t,c,h,w = images.shape
                    preds = self.model(images)
                    preds = preds.view(bs,t,h,w)
                    
#                     preds = torch.zeros([bs,t,h,w])
#                     for r in range(self.config.split_num):
#                         for c in range(self.config.split_num):
#                             preds[:,:,h_split*r:h_split*(r+1),w_split*c:w_split*(c+1)] = (
#                                 self.model(images[:,:,:,h_split*r:h_split*(r+1),w_split*c:w_split*(c+1)]).view(8,1,392,-1)#.view(4,1,765,-1)
#                             )
                    
                    loss = self.criterion(preds,density_maps)
                    metric_loss = self.metric(preds,density_maps,self.config.downsample)
                mae_loss.update(metric_loss.detach().item(),batch_size)
                summary_loss.update(loss.detach().item(), batch_size)


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
            
            #images = images.to(self.device).float()
            images = images.cuda().float()
            batch_size = images.shape[0]
            #density_maps = density_maps.to(self.device).float()
            density_maps = density_maps.cuda().float()
            
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(): #native fp16
                bs,t,c,h,w = images.shape
                preds = self.model(images)
                preds = preds.view(bs,t,h,w)
                loss = self.criterion(preds,density_maps)
                metric_loss = self.metric(preds.cpu().detach(),density_maps.cpu().detach(),self.config.downsample)
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
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    #fitter.load(f'{fitter.base_dir}/last-checkpoint.bin')
    fitter.fit(train_loader, val_loader)
    
net = ConvLSTM().cuda()

run_training()