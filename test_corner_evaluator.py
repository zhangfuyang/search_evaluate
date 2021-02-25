import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from dataset import CornerDataset
from new_scoreAgent2 import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import time 
from utils import alpha_blend

import pdb


# Set cuda environment 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

# Initialize network
prefix = '60'
model = Model(config['data_folder'], backbone_channel=64)
model.load_weight(config['save_path'], prefix)
model.to(device)
model.eval()

# Initialize data loader 
dataset = CornerDataset(config['data_folder'], phase='valid', edge_linewidth=2, render_pad=-1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Main testing code 
for iteration, data in enumerate(dataloader):
    raw_img = data['raw_img'][0].detach().cpu().numpy()
    img = data['img_norm'].to(device)
    heatmap = data['gt_heatmap_small'].to(device)
    convmpn_mask = data['convmpn_mask'].to(device)
    incorrect_corner_mask = data['incorrect_corner_mask'].to(device)

    # predicted heatmap
    heatmap_pred = model.getheatmap(img) #torch.cat((img, convmpn_mask),1)
        
    # predicted incorrect corner mask
    #img_volume = model.imgvolume(img)
    #corner_pred = model.cornerEvaluator(convmpn_mask, img_volume, binmap=None, heatmap=heatmap)

    '''
    print(torch.max(corner_pred))
    corner_pred = corner_pred.squeeze().detach().cpu().numpy()
    corner_pred = np.clip(corner_pred, 0, 1)
    plt.imshow(corner_pred, cmap='gray', vmin=0, vmax=1)
    plt.show()
    '''

    heatmap_pred = heatmap_pred.squeeze().detach().cpu().numpy()
    heatmap_pred = np.clip(heatmap_pred, 0, 1)

    heatmap_gt = heatmap.squeeze().detach().cpu().numpy()
    heatmap_gt = np.clip(heatmap_gt, 0, 1)

    convmpn_mask = convmpn_mask.squeeze().detach().cpu().numpy()
    convmpn_mask = np.clip(convmpn_mask, 0, 1)

    #heatmap_pred[heatmap_pred<0.5] = 0.0
    #heatmap_pred[heatmap_pred>=0.5] = 1.0

    f, axarr = plt.subplots(2,2) 
    axarr[0,0].imshow(raw_img)
    axarr[0,0].title.set_text('rgb image')
    axarr[0,0].axis('off')
    axarr[1,1].imshow(heatmap_pred[0])
    axarr[1,1].title.set_text('pred heatmap')
    axarr[1,1].axis('off')
    axarr[0,1].imshow(alpha_blend(heatmap_gt[1], heatmap_gt[0]))
    axarr[0,1].title.set_text('gt heatmap')
    axarr[0,1].axis('off')
    axarr[1,0].imshow(convmpn_mask[0])
    axarr[1,0].title.set_text('conv-mpn result')
    axarr[1,0].axis('off')
    '''
    axarr[2,0].imshow(heatmap_pred[1])
    axarr[2,0].title.set_text('pred corner')
    axarr[2,0].axis('off')

    axarr[2,1].imshow(heatmap_pred[0])
    axarr[2,1].title.set_text('pred line')
    axarr[2,1].axis('off')
    '''

    plt.show()

    
    #save_pic_path = os.path.join(config['save_path'], 'int')
    #os.makedirs(save_pic_path, exist_ok=True)
    #plt.savefig(os.path.join(save_pic_path, data['name'][0]+'.jpg'), bbox_inches='tight', dpi=150)
    #plt.clf()
    

      
           

