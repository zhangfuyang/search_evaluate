import torch
import torch.nn as nn
from model import cornerModel, region_model
from drn import drn_c_26
from new_utils import *
import os
import skimage
import matplotlib.pyplot as plt
import threading
import time
from new_config import *


class gpu_thread(threading.Thread):
    def __init__(self, threadID, out, sub_update_list):
        threading.Thread.__init__(self)
        self.threadId = threadID
        self.out = out
        self.sub_update_list = sub_update_list

    def run(self):
        start_time = time.time()
        print('[Thread {}] start store edge score'.format(self.threadId))
        self.out = self.out.cpu()
        edge_batch_score = self.out.exp()[:,0] / self.out.exp().sum(1)
        edge_batch_score = edge_batch_score.numpy()
        for edge_i in range(len(self.sub_update_list)):
            score = -1.9*edge_batch_score[edge_i]*edge_batch_score[edge_i]-0.1*edge_batch_score[edge_i]+1
            edge_ele = self.sub_update_list[edge_i][1]
            edge_ele.store_score(score)

        print('[Thread {}] End, in {}s'.format(self.threadId, time.time()-start_time))

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down6 = Down(512,  1024// factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up5 = Up(64, 32 // factor, bilinear)
        self.up6 = Up(32, 16, bilinear)
        self.out = nn.Sequential(
            nn.Conv2d(16, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.out(x)

class UNet_big(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_big, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down6 = Down(1024,  2048// factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128 // factor, bilinear)
        self.up5 = Up(128, 64 // factor, bilinear)
        self.up6 = Up(64, 32, bilinear)
        self.out = nn.Sequential(
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        return self.out(x)

class edge_net(nn.Module):
    def __init__(self, backbone_channel=64, edge_bin_size=36):
        super(edge_net, self).__init__()
        self.backbone_channel = backbone_channel
        self.edge_bin_size = edge_bin_size
        self.decoder1 = nn.Sequential(
            DoubleConv(6+backbone_channel, 64), # coord_conv (2), mask (2), edge (1), corner_map(1)
            DoubleConv(64, 128),
            Down(128,128),
            Down(128,128),
            Down(128,256)
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(256+self.edge_bin_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

        self.decoder3 = nn.Sequential(
            nn.Linear(256+self.edge_bin_size, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, edge_mask, mask, image_volume, corner_map, edge_bin):
        '''
        :param edge_mask: Nx1xWxH
        :param mask: Nx2xWxH
        :param image_volume: Nxbackbone_channelxWxH
        :param edge_bin: Nxedge_bin_size
        :return:
        '''

        x_channel = torch.arange(edge_mask.shape[2], device=edge_mask.device).repeat(1, edge_mask.shape[3], 1)
        y_channel = torch.arange(edge_mask.shape[3],
                                 device=edge_mask.device).repeat(1, edge_mask.shape[2], 1).transpose(1,2)
        x_channel = x_channel.float() / (edge_mask.shape[2]-1)
        y_channel = y_channel.float() / (edge_mask.shape[3]-1)

        x_channel = x_channel*2-1
        y_channel = y_channel*2-1

        x_channel = x_channel.repeat(edge_mask.shape[0], 1, 1, 1)
        y_channel = y_channel.repeat(edge_mask.shape[0], 1, 1, 1)

        input_ = torch.cat((x_channel, y_channel, edge_mask, mask, image_volume, corner_map), 1)

        out = self.decoder1(input_)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)

        input_ = torch.cat((out, edge_bin), 1)
        out_regress = self.decoder2(input_)
        out_label = self.decoder3(input_)

        return out_regress, out_label


class corner_unet(nn.Module):
    def __init__(self, bilinear=False, backbone_channel=64):
        super(corner_unet, self).__init__()
        self.bilinear = bilinear
        self.backbone_channel = backbone_channel
        self.inc = DoubleConv(2, 16)
        self.down1 = Down(16+self.backbone_channel, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.up51 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up52 = DoubleConv(32+16, 32)
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, mask, image_volume):
        x1 = self.inc(mask)
        x2 = self.down1(torch.cat((x1, image_volume),1))
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)  # x6: (1, 1024, 8, 8)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up51(x)
        x = self.up52(torch.cat((x1,x),1))
        return self.out(x)


class scoreEvaluator_with_train(nn.Module):

    def to(self, *args, **kwargs):
        super(scoreEvaluator_with_train, self).to(*args, **kwargs)
        self.device = args[0]

    def __init__(self, datapath, backbone_channel=64, edge_bin_size=36, data_scale=1, corner_bin=False):
        super(scoreEvaluator_with_train, self).__init__()
        self.backbone_channel = backbone_channel
        self.edge_bin_size = edge_bin_size
        self.data_scale = data_scale
        self.backbone = UNet_big(3, backbone_channel) # for image
        channel_size = backbone_channel + bin_size if corner_bin else backbone_channel
        channel_size = channel_size + 2 if use_heat_map else channel_size
        self.cornerNet = corner_unet(backbone_channel=channel_size)
        self.edgeNet = edge_net(backbone_channel=channel_size, edge_bin_size=edge_bin_size)
        self.img_cache = imgCache(datapath)
        self.corner_bin = corner_bin
        self.region_cache = regionCache(os.path.join(data_folder,'result/corner_edge_region/entire_region_mask'))
        self.device = 'cpu'
        if self.corner_bin:
            self.corner_bin_Net = UNet_big(3, bin_size)
        if use_heat_map:
            self.heatmapNet = UNet_big(3, 2)

    def getheatmap(self, img):
        return self.heatmapNet(img)

    def getbinmap(self, img):
        return self.corner_bin_Net(img)

    def imgvolume(self, img):
        return self.backbone(img)

    def cornerEvaluator(self, mask, img_volume, binmap=None, heatmap=None):
        '''
        :param mask: graph mask Nx2xhxw
        :return: Nx1xhxw
        '''
        volume = img_volume
        if self.corner_bin:
            volume = torch.cat((volume, binmap), 1)
        if use_heat_map:
            volume = torch.cat((volume, heatmap), 1)
        out = self.cornerNet(mask, volume)
        return out

    def edgeEvaluator(self, edge_mask, mask, img_volume, corner_map, edge_bin, binmap=None, heatmap=None):
        volume = img_volume
        if self.corner_bin:
            volume = torch.cat((volume, binmap), 1)
        if use_heat_map:
            volume = torch.cat((volume, heatmap), 1)
        out = self.edgeNet(edge_mask, mask, volume, corner_map, edge_bin)
        return out

    def regionEvaluator(self):
        pass

    def corner_map2score(self, corners, corner_map):
        corner_state = np.ones(corners.shape[0])
        scale_corners = corners*self.data_scale
        for corner_i in range(scale_corners.shape[0]):
            loc = np.round(scale_corners[corner_i]).astype(np.int)
            if loc[0] <= 1:
                x0 = 0
            else:
                x0 = loc[0] - 1
            if loc[0] >= 254:
                x1 = 256
            else:
                x1 = loc[0] + 2

            if loc[1] <= 1:
                y0 = 0
            else:
                y0 = loc[1] - 1
            if loc[1] >= 254:
                y1 = 256
            else:
                y1 = loc[1] + 2
            heat = corner_map[x0:x1, y0:y1]
            #if classficiation:
            corner_state[corner_i] = 1-2*heat.sum()/(heat.shape[0]*heat.shape[1])
            #else:
            #corner_state[corner_i] = heat.sum()/(heat.shape[0]*heat.shape[1])
        return corner_state

    def get_score_list(self, candidate_list, all_edge=False):
        first = candidate_list[0]
        img = self.img_cache.get_image(first.name)
        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = torch.cuda.FloatTensor(img, device=self.device).unsqueeze(0)
        heatmap = None
        with torch.no_grad():
            img_volume = self.imgvolume(img)
            if use_heat_map:
                heatmap = self.getheatmap(img)
        for candidate_ in candidate_list:
            self.get_score(candidate_, all_edge=all_edge, img_volume=img_volume, heatmap=heatmap)

    def get_score(self, candidate, all_edge=False, img_volume=None, heatmap=None):
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()

        img = self.img_cache.get_image(candidate.name)
        img = img.transpose((2,0,1))
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]

        mask = render(corners, edges, render_pad=-1, scale=self.data_scale)

        img = torch.cuda.FloatTensor(img, device=self.device).unsqueeze(0)
        mask = torch.cuda.FloatTensor(mask, device=self.device).unsqueeze(0)
        
        # corner and image volume
        with torch.no_grad():
            if img_volume is None:
                img_volume = self.imgvolume(img)
            bin_map = None
            if use_heat_map and heatmap is None:
                heatmap = self.getheatmap(img)

            corner_pred = self.cornerEvaluator(mask, img_volume, binmap=bin_map, heatmap=heatmap)
        corner_map = corner_pred.cpu().detach().numpy()[0][0]
        corner_state = self.corner_map2score(corners, corner_map)  

        # corner score
        graph.store_score(corner_score=corner_state)

        # edges that need to be predicted
        edge_update_list = []
        for edge_ele in graph.getEdges():
            if edge_ele.get_score() is None or all_edge:
                edge_update_list.append(edge_ele)

        batchs = patch_samples(len(edge_update_list), 16)
        for idx, batch in enumerate(batchs):
            inputs = []
            for update_i in batch:
                edge_ele = edge_update_list[update_i]
                loc1 = edge_ele.x[0].x
                loc2 = edge_ele.x[1].x
                loc1 = (round(loc1[0]*self.data_scale), round(loc1[1]*self.data_scale))
                loc2 = (round(loc2[0]*self.data_scale), round(loc2[1]*self.data_scale))
                temp_mask = cv2.line(
                    np.ones((int(256*self.data_scale),int(256*self.data_scale)))*-1,
                    loc1[::-1], loc2[::-1], 1.0,
                    thickness=2)[np.newaxis, np.newaxis, ...]
                inputs.append(temp_mask)
            edge_input_mask = np.concatenate(inputs, 0)
            edge_input_mask = torch.cuda.FloatTensor(edge_input_mask, device=self.device)
            expand_shape = edge_input_mask.shape[0]
            
            with torch.no_grad():
                if self.corner_bin:
                    bin_map_extend = bin_map.expand(expand_shape,-1,-1,-1)
                else:
                    bin_map_extend = None
                if use_heat_map:
                    heatmap_extend = heatmap.expand(expand_shape,-1,-1,-1)
                else:
                    heatmap_extend = None
                
                edge_batch_pred, _ = self.edgeEvaluator(
                    edge_input_mask,
                    mask.expand(expand_shape,-1,-1,-1),
                    img_volume.expand(expand_shape,-1,-1,-1),
                    corner_pred.expand(expand_shape,-1,-1,-1),
                    torch.zeros(expand_shape,self.edge_bin_size, device=self.device),
                    binmap=bin_map_extend,
                    heatmap=heatmap_extend
                )
            edge_batch_pred = edge_batch_pred.cpu().detach()  
            edge_batch_score = edge_batch_pred#.exp()[:,1]/edge_batch_pred.exp().sum(1)
            
            
            edge_batch_score = edge_batch_score.numpy()
            for edge_i, update_i in enumerate(batch):
                edge_ele = edge_update_list[update_i]
                edge_ele.store_score(1-2*edge_batch_score[edge_i]*edge_batch_score[edge_i])  # -1 to 1
                # edge score
                #edge_ele.store_score(edge_batch_score[edge_i])
        
        # region
        gt_mask = self.region_cache.get_region(candidate.name)
        gt_mask = gt_mask > 0.4
        conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
        conv_mask = 1 - conv_mask
        conv_mask = conv_mask.astype(np.uint8)
        labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

        background_label = region_mask[0,0]
        all_masks = []
        for region_i in range(1, labels):
            if region_i == background_label:
                continue
            the_region = region_mask == region_i
            if the_region.sum() < 20:
                continue
            all_masks.append(the_region)

        pred_mask = (np.sum(all_masks, 0) + (1 - conv_mask))>0

        iou = IOU(pred_mask, gt_mask)
        region_score = np.array([iou])
        graph.store_score(region_score=region_score)

    def store_weight(self, path, prefix):
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'backbone')), 'wb') as f:
            torch.save(self.backbone.state_dict(), f)
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'corner')), 'wb') as f:
            torch.save(self.cornerNet.state_dict(), f)
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'edge')), 'wb') as f:
            torch.save(self.edgeNet.state_dict(), f)
        if self.corner_bin:
            with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'binnet')), 'wb') as f:
                torch.save(self.corner_bin_Net.state_dict(), f)
        if use_heat_map:
            with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'heatmapnet')), 'wb') as f:
                torch.save(self.heatmapNet.state_dict(), f)

    def load_weight(self, path, prefix):
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'backbone')), 'rb') as f:
            self.backbone.load_state_dict(torch.load(f))
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'corner')), 'rb') as f:
            self.cornerNet.load_state_dict(torch.load(f))
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'edge')), 'rb') as f:
            self.edgeNet.load_state_dict(torch.load(f))
        if self.corner_bin:
            with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'binnet')), 'rb') as f:
                self.corner_bin_Net.load_state_dict(torch.load(f))
        if use_heat_map:
            with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'heatmapnet')), 'rb') as f:
                self.heatmapNet.load_state_dict(torch.load(f))

