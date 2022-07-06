import torch
import torch.nn as nn

def lstn_loss(preds_t0, preds_t1_blocks, gts, imgs, lamda=0.001, beta=30):
    '''
    Args:
        preds_t0: Tensor with shape (B x T x 1 x H x W)
        preds_t1_blocks: List of tensors with shape (B x T x 1 x H/h x W/w)
        gts: Tensor with shape (B x T x H x W)
        imgs: Tensor with shape (B x T x 3 x H x W)
        lamda: Float
        beta: Float
    '''
    batch_size = preds_t0.shape[0]
    seq_length = preds_t0.shape[1]
    result_reg_loss = torch.zeros(batch_size).cuda()
    result_lst_loss = torch.zeros(batch_size).cuda()
    
    for i in range(seq_length):
        pred_t0 = preds_t0[:,i,...].squeeze()
        pred_t1_blocks = [preds[:,i,...].squeeze() for preds in preds_t1_blocks]
        gt0 = gts[:,i,...]
        gt1 = None if i == seq_length -1 else gts[:,i+1,...]
        img0 = imgs[:,i,...]
        img1 = None if i == seq_length -1 else imgs[:,i+1,...]

        result_reg_loss += reg_loss(pred_t0, gt0)
        result_lst_loss += lst_loss(pred_t1_blocks, gt1, img0, img1, beta)
        
    result_reg_loss /= (2 * seq_length)
    result_lst_loss /= (2 * seq_length -2)
    result = result_reg_loss + lamda * result_lst_loss
    return torch.mean(result)

def mlstn_loss(preds_t012, preds_t3, gts, imgs, lamda=0.001, beta=30):
    '''
    Args:
        preds_t012: Tensor with shape (B x T-2 x 3 x 1 x H x W)
        preds_t3: Tensor with shape (B x T-2 x 1 x H x W)
        gts: Tensor with shape (B x T x H x W)
        imgs: Tensor with shape (B x T x 3 x H x W)
        lamda: Float
        beta: Float
    '''
    batch_size = gts.shape[0]
    seq_length = gts.shape[1]
    result_reg_loss = torch.zeros(batch_size).cuda()
    result_lst_loss = torch.zeros(batch_size).cuda()
    
    assert seq_length >= 3
    for i in range(seq_length-3):
        pred_t0 = preds_t012[:,i,0,...].squeeze()
        pred_t3_block = [preds_t3[:,i,...].squeeze()]
        gt0 = gts[:,i,...]
        gt3 = gts[:,i+3,...]
        img2 = imgs[:,i+2,...]
        img3 = imgs[:,i+3,...]

        result_reg_loss += reg_loss(pred_t0, gt0)
        result_lst_loss += lst_loss(pred_t3_block, gt3, img2, img3, beta)
        
    result_reg_loss += reg_loss(preds_t012[:,-1,...].squeeze(), gts[:,seq_length-3:seq_length,...]).sum(dim=1)
    result_reg_loss /= (2 * seq_length)
    result_lst_loss /= (2 * seq_length - 6)
    result = result_reg_loss + lamda * result_lst_loss
    return torch.mean(result)

def reg_loss(preds, gts):
    return nn.MSELoss(reduction='none')(preds, gts).mean(dim=[-1,-2])

def lst_loss(preds_blocks, gts, imgs0, imgs1, beta):
    '''
    Args:
        preds_blocks: List of tensors with shape (B x H/h x W/w)
        gts: Tensor with shape (B x H x W)
        imgs0: Tensor with shape (B x 3 x H x W)
        imgs1: Tensor with shape (B x 3 x H x W)
        beta: Float
    '''
    
    batch_size = preds_blocks[0].shape[0]
    result = torch.zeros(batch_size).cuda()
    
    if gts is None or imgs1 is None:
        return result
    
    gts_blocks = []
    imgs0_blocks = []
    imgs1_blocks = []
    
    num_blocks = len(preds_blocks)
    assert num_blocks == 1 or num_blocks == 2 or num_blocks == 4
    block_w = 2 if num_blocks >= 2 else 1
    block_h = 2 if num_blocks == 4 else 1
    
    gts_chunks = torch.chunk(gts, block_w, dim=-2)
    imgs0_chunks = torch.chunk(imgs0, block_w, dim=-2)
    imgs1_chunks = torch.chunk(imgs1, block_w, dim=-2)
    for gts_chunk, imgs0_chunk, imgs1_chunk in zip(gts_chunks, imgs0_chunks, imgs1_chunks):
        gts_subchunks = torch.chunk(gts_chunk, block_h, dim=-1)
        imgs0_subchunks = torch.chunk(imgs0_chunk, block_h, dim=-1)
        imgs1_subchunks = torch.chunk(imgs1_chunk, block_h, dim=-1)
        for gts_subchunk, imgs0_subchunk, imgs1_subchunk in zip(gts_subchunks, imgs0_subchunks, imgs1_subchunks):
            gts_blocks.append(gts_subchunk)
            imgs0_blocks.append(imgs0_subchunk)
            imgs1_blocks.append(imgs1_subchunk)
    
    for preds_block, gts_block, imgs0_block, imgs1_block in zip(preds_blocks, gts_blocks, imgs0_blocks, imgs1_blocks):
        result += img_similarity(imgs0_block, imgs1_block, beta) * reg_loss(preds_block, gts_block)
        
    return result

def img_similarity(imgs0, imgs1, beta):
    result = reg_loss(imgs0, imgs1).mean(dim=1)
    result = torch.exp(-result / (2 * beta * beta))
    return result
    