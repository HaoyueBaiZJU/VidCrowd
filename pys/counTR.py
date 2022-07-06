import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from fastai.callback.hook import hook_outputs
from fastai.vision.models.unet import UnetBlock
from fastai.layers import ConvLayer, BatchNorm,PixelShuffle_ICNR,SigmoidRange
from fastai.torch_core import apply_init


class PermuteReShape(nn.Module):
    def __init__(self, order,shape):
        super().__init__()
        self.order = order
        self.shape = shape
        
    @torch.cuda.amp.autocast()
    def forward(self,x):
        return x.permute(self.order).view(self.shape)

class UpScaleOrig(nn.Module):
    def __init__(self,sz=448,mode='nearest'):
        super().__init__()
        self.sz = sz
        self.mode = mode
        
    @torch.cuda.amp.autocast()
    def forward(self,x):
        return F.interpolate(x, (self.sz,self.sz), mode=self.mode)


class MyUnetBlock(UnetBlock):
    
    @torch.cuda.amp.autocast()
    def forward(self,up_in):
        s = self.hook.stored
        bs, feat, ch = s.shape
        sz = int(np.sqrt(feat))
        s = s.permute(0,2,1).view(bs,ch,sz,sz)
        up_out = self.shuf(up_in)
        ssh = s.shape[-2:]
        if ssh != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, s.shape[-2:], mode='nearest')
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        out = self.conv2(self.conv1(cat_x))
        return out

class CounTR(nn.Module):
    def __init__(self, encoder, act_cls, layer_idx, blur=False,
                 imsize=448, init=nn.init.kaiming_normal_, norm_type=None,
                 self_attention=False, y_range=None, **kwargs):
        super().__init__()
        h_layer = []
        for name,layer in encoder[2].named_modules():
            if 'mlp.fc2' in name and isinstance(layer,torch.nn.Linear):
                h_layer.append(layer)
        self.sfs = hook_outputs([l for i,l in enumerate(h_layer) if i in layer_idx],detach=False)
        encoder.eval()
        x = encoder(torch.rand(2,3,imsize,imsize)).detach()
        bs, feat, ni = x.shape
        sz = int(np.sqrt(feat))
        middle_conv = nn.Sequential(ConvLayer(ni, ni*2, act_cls=act_cls, norm_type=norm_type),
                                    ConvLayer(ni*2, ni, act_cls=act_cls, norm_type=norm_type)).eval()
        x = middle_conv(x.permute(0,2,1).view(-1,ni,sz,sz)).detach()
        layers = [encoder,nn.ReLU()]
#                   PermuteReShape(order=(0,2,1),shape=(-1,ni,sz,sz)),
#                   middle_conv]
        
        ##### Done downward path #####
        
        ##### Refactor Upward path #####
        decoder_layers = [middle_conv]
        for i in reversed(range(len(self.sfs))):
            final_d = True
            if i == 0: final_d = False
            ni = x.shape[1]
            _, _, ch = self.sfs[i].stored.shape
            unet_block = MyUnetBlock(ni, ch, self.sfs[i],
                                 final_div=final_d,blur=blur,
                                 self_attention=self_attention,act_cls=act_cls,
                                 init=init,norm_type=norm_type).eval()
            decoder_layers.append(unet_block)
            x = unet_block(x).detach()
        #### Refactor Upward path#####
        
        ni = x.shape[1]
        if imsize != x.shape[-1]:
            decoder_layers.append(PixelShuffle_ICNR(ni,act_cls=act_cls,norm_type=norm_type))
        decoder_layers.append(UpScaleOrig(imsize))
        decoder_layers += [ConvLayer(ni,1,ks=1,act_cls=None,norm_type=norm_type,**kwargs)]
        apply_init(nn.Sequential(decoder_layers[0]),init)
        #apply_init(nn.Sequential(layers[3]),init)
        if y_range is not None: decoder_layers.append(SigmoidRange(*y_range))
        del x
        #self.model = nn.Sequential(*layers)
        self.backbone = nn.Sequential(*layers)
        self.decoder = nn. Sequential(*decoder_layers)
        
    @torch.cuda.amp.autocast()    
    def forward(self,x):
        #out = self.model(x)
        x = self.backbone(x)
        _, feat, ni = x.shape
        sz = int(np.sqrt(feat))
        x = x.permute(0,2,1).view(-1,ni,sz,sz)
        x = self.decoder(x)
        return x
    
    def __del__(self):
        if hasattr(self, "sfs"): self.sfs.remove()

### usage

# from counTR import CounTR

# import timm
# import torch.nn as nn

# timm.models.swin_transformer.default_cfgs['swin_base_patch4_window7_224']['input_size'] = (3,448,448)
# timm.models.swin_transformer.default_cfgs['swin_base_patch4_window7_224']

# backbone = nn.Sequential(*list(timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0).children())[:-2])
# act_cls = nn.ReLU
# idx = [1,3,21,23]
# net = CounTR(backbone,act_cls=act_cls,layer_idx=idx,y_range=[0,1])
# net.eval()
# with torch.no_grad():
#     preds = net(torch.rand(2,3,448,448))
# preds.shape


# def get_pad(im_sz, crop_sz=448,**kwargs):
#     h,w = im_sz
#     h_mul = h // crop_sz + 1
#     w_mul = w // crop_sz + 1
#     pad_h = (crop_sz * h_mul - h) % crop_sz
#     pad_w = (crop_sz * w_mul - w) % crop_sz
#     assert pad_h%2 == 0
#     assert pad_w%2 == 0
#     return pad_h//2, pad_w//2

# import torch
# net.eval()
# with torch.no_grad():
#     imgs = torch.rand(2,3,1530,2720)
#     imgs = torch.nn.functional.pad(imgs,(pad_w,pad_w,pad_h,pad_h))
#     for m in range(7):
#         for n in range(4):
#             imgs_patch = imgs[:,:,448*n:448*(n+1),448*m:448*(m+1)]
#             preds_map = net(imgs_patch)
# preds_map.shape


