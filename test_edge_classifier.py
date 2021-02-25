import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from config import config
from dataset_classifier import CornerDataset
from new_scoreAgent import scoreEvaluator_with_train as Model
from dicttoxml import dicttoxml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time 
from utils import alpha_blend

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
prefix = '60'
model = Model(config['data_folder'], backbone_channel=64)
model.load_weight(config['save_path'], prefix)
model.to(device)
model.eval()

# Initialize data loader 
dataset = CornerDataset(config['data_folder'], phase='valid', edge_linewidth=2, render_pad=-1)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# Main testing code 
total = 0
total_pos = 0
total_neg = 0
total_correct = 0
total_correct_pos = 0
total_correct_neg = 0

for iteration, data in enumerate(dataloader):
    print(iteration)
    img = data['img_norm'].to(device)
    heatmap_large = data['gt_heatmap_large'].to(device)
    heatmap_small = data['gt_heatmap_small'].to(device)
    convmpn_mask = data['convmpn_mask'].to(device)
    edge_masks = data['edge_masks'].to(device).squeeze()
    edge_label = data['edge_label'].squeeze().detach().cpu().numpy()    

    with torch.no_grad():
        img_volume = model.imgvolume(img)  # (bs, 64, 256, 256)
        heatmap_pred = model.getheatmap(img) #(bs, 2, 256, 256)

        for ii in range(edge_masks.shape[0]):
            edge = edge_masks[ii].unsqueeze(0).unsqueeze(1)
            gt_label = edge_label[ii]
            
            concat_input = torch.cat((edge, convmpn_mask, img_volume, heatmap_pred.detach()), 1)
            pred_logits = model.convnet(concat_input)
            pred_label = pred_logits.argmax(1).detach().cpu().numpy()[0]

            if pred_label == gt_label:
                total_correct += 1
                if gt_label == 1:
                    total_correct_pos += 1
                else:
                    total_correct_neg += 1
            if gt_label == 1:
                total_pos += 1
            else:
                total_neg += 1
            total += 1

       
print('[TEST] Total Edge Accuracy is {}%, Good Edge Accuracy is {}%, Bad Edge Accuracy is {}%'.format(
    round(100.0*total_correct/total, 5),
    round(100.0*total_correct_pos/total_pos, 5),
    round(100.0*total_correct_neg/total_neg, 5)))

print(total_correct_pos)
print(total_pos)
print(total_correct_neg)
print(total_neg)
print(total_correct)
print(total)