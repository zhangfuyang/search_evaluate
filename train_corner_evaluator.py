import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from dataset import CornerDataset
from new_scoreAgent import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time 

import pdb

# save configuration file 
print(config)
os.makedirs(config['save_path'], exist_ok=True)
f = open(os.path.join(config['save_path'], 'config.xml'), 'wb')
f.write(dicttoxml(config))
f.close()
print('save config.xml done.')

# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

# Initialize network
model = Model(config['data_folder'], backbone_channel=64)
model.to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_stepsize'], gamma=0.5)
cornerloss = nn.L1Loss()
heatmaploss = nn.MSELoss()

# Initialize data loader 
dataset = CornerDataset(config['data_folder'], phase='train', edge_linewidth=2, render_pad=-1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)

# Main training code 
num_batch = len(dataset) / config['batchsize']
for epoch in range(config['num_epochs']):
    start_time = time.time()
    for iteration, data in enumerate(dataloader):
        img = data['img_norm'].to(device)
        heatmap_large = data['gt_heatmap_large'].to(device)
        heatmap_small = data['gt_heatmap_small'].to(device)
        convmpn_mask = data['convmpn_mask'].to(device)
        incorrect_corner_mask = data['incorrect_corner_mask'].to(device)

        # predicted heatmap
        heatmap_pred = model.getheatmap(convmpn_mask)
        
        mask_gt_large = torch.where(heatmap_large>0.5)
        mask_gt_small = torch.where(heatmap_small>0.5)
        mask_pred = torch.where(heatmap_pred>0.1)
        
        # predicted incorrect corner mask
        #img_volume = model.imgvolume(img)
        #corner_pred = model.cornerEvaluator(convmpn_mask, img_volume, binmap=None, heatmap=heatmap)

        # compute loss 
        #corner_l = cornerloss(corner_pred, incorrect_corner_mask.unsqueeze(1))
        heatmap_l = F.smooth_l1_loss(heatmap_pred[mask_pred], heatmap_small[mask_pred]) +\
                    F.smooth_l1_loss(heatmap_pred[mask_gt_small], heatmap_small[mask_gt_small]) 

        optimizer.zero_grad()
        loss =  heatmap_l
        loss.backward()
        optimizer.step()
        '''
        if (iteration+1) % config['print_freq'] == 0:
            print('[Epoch %d: %d/%d] corner loss: %.4f, heatmap loss: %.4f, total loss: %.4f' % 
                   (epoch, iteration+1, num_batch, corner_l.item(), heatmap_l.item(), loss.item()))
        '''
        if (iteration+1) % config['print_freq'] == 0:
            print('[Epoch %d: %d/%d] heatmap loss: %.4f, total loss: %.4f' % 
                   (epoch, iteration+1, num_batch, heatmap_l.item(), loss.item()))

    scheduler.step()

    print("--- %s seconds for one epoch ---" % (time.time() - start_time))
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    print('Learning Rate: %s' % str(lr))
    start_time = time.time()

    # save model weights
    if (epoch+1) % config['save_freq'] == 0:
        model.store_weight(config['save_path'], str(epoch+1))

       

