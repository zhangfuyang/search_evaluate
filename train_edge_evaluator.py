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
        convmpn_mask_good = data['convmpn_mask_good'].to(device)
        convmpn_mask_bad = data['convmpn_mask_bad'].to(device)

        img_volume = model.imgvolume(img)  # (bs, 64, 256, 256)
        heatmap_pred = model.getheatmap(img) #(bs, 2, 256, 256)
        convmpn_pred = model.edgeEvaluator(convmpn_mask, img_volume, heatmap_pred.detach()) #(bs, 1, 256, 256)
        
        mask_gt_large = torch.where(heatmap_large>0.5)
        mask_gt_small = torch.where(heatmap_small>0.5)
        mask_convmpn = torch.where(convmpn_mask>0.5)
        
        # compute loss 
        #corner_l = cornerloss(corner_pred, incorrect_corner_mask.unsqueeze(1))
        heatmap_relax = 0.9*heatmap_small + 0.1*(1-heatmap_small)
        convmpn_relax = 0.9*convmpn_mask_bad + 0.1*(1-convmpn_mask_bad)
        heatmap_l = F.smooth_l1_loss(heatmap_pred, heatmap_relax) +\
                    2.0*F.smooth_l1_loss(heatmap_pred[mask_gt_small], heatmap_relax[mask_gt_small])

        convmpn_l = 2.0*F.smooth_l1_loss(convmpn_pred[mask_convmpn], convmpn_relax[mask_convmpn]) +\
                    0.5*F.smooth_l1_loss(convmpn_pred, convmpn_relax)

        optimizer.zero_grad()
        loss =  heatmap_l + convmpn_l
        loss.backward()
        optimizer.step()
       
        if (iteration+1) % config['print_freq'] == 0:
            print('[Epoch %d: %d/%d] convmpn loss: %.4f, heatmap loss: %.4f, total loss: %.4f' % 
                   (epoch, iteration+1, num_batch, convmpn_l.item(), heatmap_l.item(), loss.item()))

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

       

