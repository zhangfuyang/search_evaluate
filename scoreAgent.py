import torch
import torch.nn as nn
from model import cornerModel, region_model
from drn import drn_c_26
from new_utils import *
import os
import torch.nn.functional as F
from config import config



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        if batchnorm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm):
        super(Down, self).__init__()
        if batchnorm:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
                DoubleConv(in_channels, out_channels, batchnorm=batchnorm))
        else:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2),
                nn.InstanceNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
                DoubleConv(in_channels, out_channels, batchnorm=batchnorm))

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, batchnorm=batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels, out_channels, batchnorm=batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet_big(nn.Module):
    def __init__(self, n_channels, n_classes, batchnorm, useSigmoid=True, bilinear=False):
        super(UNet_big, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32, batchnorm=batchnorm)
        self.down1 = Down(32, 64, batchnorm=batchnorm)
        self.down2 = Down(64, 128, batchnorm=batchnorm)
        self.down3 = Down(128, 256, batchnorm=batchnorm)
        self.down4 = Down(256, 512, batchnorm=batchnorm)
        self.down5 = Down(512, 1024, batchnorm=batchnorm)
        factor = 2 if bilinear else 1
        self.down6 = Down(1024,  2048// factor,  batchnorm=batchnorm)
        self.up1 = Up(2048, 1024 // factor,  batchnorm, bilinear)
        self.up2 = Up(1024, 512 // factor,  batchnorm, bilinear)
        self.up3 = Up(512, 256 // factor,  batchnorm, bilinear)
        self.up4 = Up(256, 128 // factor,  batchnorm, bilinear)
        self.up5 = Up(128, 64 // factor,  batchnorm, bilinear)
        self.up6 = Up(64, 32,  batchnorm, bilinear)
        if useSigmoid:
            self.out = nn.Sequential(
                nn.Conv2d(32, n_classes, kernel_size=1),
                nn.Sigmoid())
        else:
            self.out = nn.Sequential(
                nn.Conv2d(32, n_classes, kernel_size=1),
                nn.BatchNorm2d(n_classes),
                nn.ReLU())

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


class UNnet(nn.Module):
    def __init__(self, batchnorm, bilinear=False, backbone_channel=64):
        super(UNnet, self).__init__()
        self.bilinear = bilinear
        self.backbone_channel = backbone_channel
        self.inc = DoubleConv(2, 16, batchnorm=batchnorm)
        self.down1 = Down(16+self.backbone_channel, 64, batchnorm=batchnorm)
        self.down2 = Down(64, 128, batchnorm=batchnorm)
        self.down3 = Down(128, 256, batchnorm=batchnorm)
        self.down4 = Down(256, 512, batchnorm=batchnorm)
        factor = 2 if bilinear else 1
        self.down5 = Down(512, 1024 // factor, batchnorm=batchnorm)
        self.up1 = Up(1024, 512 // factor, batchnorm, bilinear)
        self.up2 = Up(512, 256 // factor, batchnorm, bilinear)
        self.up3 = Up(256, 128 // factor, batchnorm, bilinear)
        self.up4 = Up(128, 64 // factor, batchnorm, bilinear)
        self.up51 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.up52 = DoubleConv(32+16, 32, batchnorm=batchnorm)
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),  
            nn.Sigmoid())

    def forward(self, mask, image_volume):
        x1 = self.inc(mask)
        x2 = self.down1(torch.cat((x1, image_volume),1))
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up51(x)
        x = self.up52(torch.cat((x1,x),1))
        return self.out(x)



class scoreEvaluator_with_train(nn.Module):

    def __init__(self, datapath, device, backbone_channel=64, use_heat_map=True):
        super(scoreEvaluator_with_train, self).__init__()
        self.backbone_channel = backbone_channel
        self.backbone = UNet_big(3, backbone_channel, batchnorm=True, useSigmoid=False) # for image
        channel_size = backbone_channel# + bin_size if corner_bin else backbone_channel
        channel_size = channel_size + 2# if use_heat_map else channel_size
        self.cornerNet = UNnet(batchnorm=True, backbone_channel=channel_size)
        self.edgeNet = UNnet(batchnorm=True, backbone_channel=channel_size)
        self.img_cache = imgCache(datapath)
        self.region_cache = regionCache(os.path.join(config['data_folder'], 'result/corner_edge_region/entire_region_mask'))
        self.heatmapNet = UNet_big(3, 2, batchnorm=True, useSigmoid=True)
        self.device = device
        

    def getheatmap(self, img):
        heatmap = self.heatmapNet(img)
        return heatmap

    def imgvolume(self, img):
        imgvolume = self.backbone(img)
        return imgvolume

    def cornerEvaluator(self, mask, img_volume, heatmap):
        '''
        :param mask: graph mask Nx2xhxw
        :return: Nx1xhxw
        '''
        volume = img_volume
        volume = torch.cat((volume, heatmap), 1)
        out = self.cornerNet(mask, volume)
        return out

    def edgeEvaluator(self, mask, img_volume, heatmap):
        volume = img_volume
        volume = torch.cat((volume, heatmap), 1)
        out = self.edgeNet(mask, volume)
        return out

    def regionEvaluator(self):
        pass

    def corner_map2score(self, corners, corner_map):
        corner_state = np.ones(corners.shape[0])
        scale_corners = corners
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
            corner_state[corner_i] = 1-2*heat.sum()/(heat.shape[0]*heat.shape[1])
        return corner_state


    def get_score_list(self, candidate_list):
        for candidate_ in candidate_list:
            self.get_score(candidate_)

   
    def get_score(self, candidate, img_volume=None, heatmap=None):
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()

        if img_volume is None or heatmap is None:
            img = self.img_cache.get_image(candidate.name)
            img = img.transpose((2,0,1))
            img = (img - np.array(config['mean'])[:, np.newaxis, np.newaxis]) / np.array(config['std'])[:, np.newaxis, np.newaxis]
            img = torch.cuda.FloatTensor(img, device=self.device).unsqueeze(0)
            # corner and image volume
            with torch.no_grad():
                img_volume = self.imgvolume(img)
                heatmap = self.getheatmap(img)

        mask = render(corners, edges, render_pad=-1, scale=1)
        mask = torch.cuda.FloatTensor(mask, device=self.device).unsqueeze(0)
        
        # corner and image volume
        with torch.no_grad():
            corner_pred = self.cornerEvaluator(mask, img_volume, heatmap)
        corner_map = corner_pred.squeeze().cpu().detach().numpy()  
        corner_map = np.clip(corner_map, 0, 1) 
        corner_state = self.corner_map2score(corners, corner_map) 
        
        # corner score
        graph.store_score(corner_score=corner_state)

        gt_mask = mask.squeeze()[0].detach().cpu().numpy()>0
        with torch.no_grad():
            edge_pred = self.edgeEvaluator(mask, img_volume, heatmap)
        pred_bad_edges = edge_pred.squeeze().detach().cpu().numpy()
        pred_bad_edges = np.clip(pred_bad_edges, 0, 1)*gt_mask 
        
        for edge_ele in graph.getEdges():
            loc1 = edge_ele.x[0].x
            loc2 = edge_ele.x[1].x
            loc1 = (round(loc1[0]), round(loc1[1]))
            loc2 = (round(loc2[0]), round(loc2[1]))
            edge_mask = cv2.line(
                    np.ones((int(256),int(256)))*0,
                    loc1[::-1], loc2[::-1], 1.0,
                    thickness=2)
            ratio = np.sum(np.multiply(pred_bad_edges, edge_mask))/np.sum(edge_mask)
            edge_ele.store_score(1-2*ratio)

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
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'edge')), 'wb') as f:
            torch.save(self.edgeNet.state_dict(), f)
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'corner')), 'wb') as f:
            torch.save(self.cornerNet.state_dict(), f)
       

    def load_weight(self, path, prefix):
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'backbone')), 'rb') as f:
            self.backbone.load_state_dict(torch.load(f))
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'corner')), 'rb') as f:
            self.cornerNet.load_state_dict(torch.load(f))
        with open(os.path.join(path, '{}_{}.pt'.format(prefix, 'edge')), 'rb') as f:
            self.edgeNet.load_state_dict(torch.load(f))
       
