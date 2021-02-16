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
            #nn.BatchNorm2d(mid_channels),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
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
            #nn.BatchNorm2d(in_channels),
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
            nn.Linear(256, 2)
        )
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))

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
        out = self.decoder2(input_)

        return out

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
        x6 = self.down5(x5)
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
        self.region_cache = regionCache(os.path.join(data_folder, 'result/corner_edge_region/entire_region_mask'))
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
            corner_state[corner_i] = 1-2*heat.sum()/(heat.shape[0]*heat.shape[1])
        return corner_state

    def get_score(self, candidate, all_edge=False):
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
            img_volume = self.imgvolume(img)
            if self.corner_bin:
                bin_map = self.getbinmap(img)
            else:
                bin_map = None
            if use_heat_map:
                heatmap = self.getheatmap(img)
            else:
                heatmap = None

            corner_pred = self.cornerEvaluator(mask, img_volume, binmap=bin_map, heatmap=heatmap)
        corner_map = corner_pred.cpu().detach().numpy()[0][0]
        corner_state = self.corner_map2score(corners, corner_map)
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

                edge_batch_pred = self.edgeEvaluator(
                    edge_input_mask,
                    mask.expand(expand_shape,-1,-1,-1),
                    img_volume.expand(expand_shape,-1,-1,-1),
                    corner_pred.expand(expand_shape,-1,-1,-1),
                    torch.zeros(expand_shape,self.edge_bin_size, device=self.device),
                    binmap=bin_map_extend,
                    heatmap=heatmap_extend
                )
            edge_batch_pred = edge_batch_pred.cpu().detach()
            edge_batch_score = edge_batch_pred.exp()[:,1]/edge_batch_pred.exp().sum(1)
            edge_batch_score = edge_batch_score.numpy()
            for edge_i, update_i in enumerate(batch):
                edge_ele = edge_update_list[update_i]
                edge_ele.store_score(1-2*edge_batch_score[edge_i]*edge_batch_score[edge_i])

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


class scoreEvaluator():
    def __init__(self, modelPath, useHeatmap=(), useGT=(), dataset=None):
        self.useHeatmap = useHeatmap
        self.useGT = useGT
        self.dataset = dataset
        if 'cornerModelPath' in modelPath.keys():
            self.cornermodelPath = modelPath['cornerModelPath']
            self.cornerEvaluator = cornerModel()
            with open(self.cornermodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.cornerEvaluator.load_state_dict(state_dict)
                self.cornerEvaluator.double()
                self.cornerEvaluator.cuda()
                self.cornerEvaluator.eval()
        else:
            self.cornerEvaluator = None

        if 'region' in self.useHeatmap:
            self.region_use_heatmap = True
            #self.regionHeatmapPath = modelPath['regionHeatmapPath']
            self.regionHeatmapPath = modelPath['regionEntireMaskPath']
        else:
            self.region_use_heatmap = False

        if 'edgeModelPath' in modelPath.keys():
            self.edgemodelPath = modelPath['edgeModelPath']
            self.edgeEvaluator = drn_c_26(pretrained=True, num_classes=2, in_channels=6)
            with open(self.edgemodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.edgeEvaluator.load_state_dict(state_dict, resnet_pretrained=False)
                self.edgeEvaluator.double()
                self.edgeEvaluator.cuda()
                self.edgeEvaluator.eval()
        else:
            self.edgeEvaluator = None

        if 'regionModelPath' in modelPath.keys():
            self.regionmodelPath = modelPath['regionModelPath']
            self.regionEvaluator = region_model(iters=modelPath['region_iter'])
            with open(self.regionmodelPath, 'rb') as f:
                state_dict = torch.load(f)
                self.regionEvaluator.load_state_dict(state_dict)
                self.regionEvaluator.double()
                self.regionEvaluator.cuda()
                self.regionEvaluator.eval()
        else:
            self.regionEvaluator = None

    def corner_map2score(self, corners, map):
        corner_state = np.ones(corners.shape[0])
        for corner_i in range(corners.shape[0]):
            loc = np.round(corners[corner_i]).astype(np.int)
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
            heat = map[x0:x1, y0:y1]
            corner_state[corner_i] = 1-heat.sum()/heat.shape[0]/heat.shape[1] * 2  #[-1, 1]
        return corner_state

    def get_list_candidates_score(self, img, graphs, name=None):
        scores = []
        configs = []
        for graph_i in range(len(graphs)):
            score, score_config = self.get_score(img=img, corners=graphs[graph_i][0], edges=graphs[graph_i][1], name=name)
            scores.append(score)
            configs.append(score_config)
        #N = 16
        #scores = np.array([])
        #group = np.ceil(len(graphs)/N).astype(np.int)
        #for group_i in range(group):
        #    if group_i == group-1:
        #        gs = graphs[N*group_i:]
        #    else:
        #        gs = graphs[N*group_i:N*group_i+N]
        #    masks = []
        #    for i in range(len(gs)):
        #        corners = gs[i][0]
        #        edges = gs[i][1]
        #        mask = render(corners, edges, -1, 2)
        #        masks.append(mask)
        #    masks = torch.Tensor(masks).double()
        #    corners_group = [gs[i][0] for i in range(len(gs))]
        #    scores_group = self.get_score(img, masks, corners_group)
        #    scores = np.concatenate((scores, scores_group), 0)

        return scores, configs

    def load_image(self, name):
        img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        img = img.transpose((2,0,1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - np.array(mean)[:, np.newaxis, np.newaxis]) / np.array(std)[:, np.newaxis, np.newaxis]
        img = torch.cuda.DoubleTensor(img)
        img = img.unsqueeze(0)
        return img

    def get_fast_score_list(self, candidate_list):
        # load image
        name = candidate_list[0].name
        img = self.load_image(name)

        ########################### corner ################################
        corner_time = time.time()
        batchs = patch_samples(len(candidate_list), 16)
        for batch in batchs:
            inputs = []
            for candidate_i in batch:
                candidate = candidate_list[candidate_i]
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
                graph_mask = torch.cuda.DoubleTensor(graph_mask)
                inputs.append(torch.cat((img, graph_mask), 1))
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat(inputs, 0))
            corner_map = corner_map.detach().cpu().numpy()
            for idx in range(corner_map.shape[0]):
                graph = candidate_list[batch[idx]].graph
                corners = graph.getCornersArray()
                corners_state = self.corner_map2score(corners, corner_map[idx,0])
                corners_score = np.array(corners_state)
                graph.store_score(corner_score=corners_score)

        print('corner time', time.time()-corner_time)
        ############################ edge ##################################
        edge_time = time.time()

        # extract all elements that need recounting
        update_list = []
        for candidate in candidate_list:
            graph = candidate.graph
            edges = graph.getEdges()
            for edge_ele in edges:
                if edge_ele.get_score() is None:
                    update_list.append((graph, edge_ele))

        # split into batches
        batchs = patch_samples(len(update_list), 16)
        for bi, batch in enumerate(batchs):
            inputs = []
            for update_i in batch:
                update_unit = update_list[update_i]
                graph = update_unit[0]
                edge_ele = update_unit[1]
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
                graph_mask = torch.cuda.DoubleTensor(graph_mask)
                loc1 = edge_ele.x[0].x
                loc2 = edge_ele.x[1].x
                temp_mask = cv2.line(np.ones((256,256))*-1, loc1[::-1], loc2[::-1],
                                     1.0, thickness=2)[np.newaxis, ...]
                temp_mask = torch.cuda.DoubleTensor(temp_mask).unsqueeze(0)
                inputs.append(torch.cat((img, graph_mask, temp_mask), 1))
            with torch.no_grad():
                out = self.edgeEvaluator(torch.cat(inputs, 0))
            out = out.detach()

            out = out.cpu()
            edge_batch_score = out.exp()[:,0] / out.exp().sum(1)
            edge_batch_score = edge_batch_score.numpy()
            for edge_i in range(edge_batch_score.shape[0]):
                score = -1.9*edge_batch_score[edge_i]*edge_batch_score[edge_i]-0.1*edge_batch_score[edge_i]+1
                edge_ele = update_list[batch[edge_i]][1]
                edge_ele.store_score(score)

        print('edge time', time.time()-edge_time)
        ########################## region ###################################
        region_time = time.time()
        gt_mask = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'))

        if 'entire' in self.regionHeatmapPath:
            gt_mask = gt_mask > 0.4
            for candidate in candidate_list:
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                all_masks = []
                regions_number = []
                for region_i in range(1, labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    all_masks.append(the_region)
                    regions_number.append(region_i)

                pred_mask = (np.sum(all_masks, 0) + (1 - conv_mask))>0

                iou = IOU(pred_mask, gt_mask)
                regions_score = np.array([iou])

                graph.store_score(region_score=regions_score)

        else:
            for candidate in candidate_list:
                graph = candidate.graph
                corners = graph.getCornersArray()
                edges = graph.getEdgesArray()
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                false_region_id = []
                region_num = 0
                pred_region_masks = []
                for region_i in range(1,labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    region_num += 1
                    pred_region_masks.append((the_region, region_i))

                temp_used_pred = set()
                temp_gt_map = []
                pred_map_gt = [-1 for _ in range(len(pred_region_masks))]
                for gt_i in range(gt_mask.shape[0]):
                    best_iou = 0.5
                    best_pred_idx = -1
                    for pred_i in range(len(pred_region_masks)):
                        if pred_i in temp_used_pred:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_i])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_i
                    if best_pred_idx != -1:
                        temp_used_pred.add(best_pred_idx)
                        pred_map_gt[best_pred_idx] = gt_i
                    temp_gt_map.append(best_pred_idx)
                for pred_i in range(len(pred_region_masks)):
                    if pred_map_gt[pred_i] == -1:
                        best_iou = 0
                        best_gt_idx = -1
                        for gt_i in range(gt_mask.shape[0]):
                            if temp_gt_map[gt_i] != -1:
                                continue
                            iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_i])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_i
                        pred_map_gt[pred_i] = best_gt_idx

                regions_state = []
                regions_number = []
                for pred_i in range(len(pred_region_masks)):
                    gt_idx = pred_map_gt[pred_i]
                    if gt_idx == -1:
                        iou = 0
                    else:
                        iou = IOU(pred_region_masks[pred_i][0], gt_mask[gt_idx])
                    regions_state.append(iou*2-1)
                    regions_number.append(pred_region_masks[pred_i][1])

                regions_score = np.array([regions_state])

                graph.store_score(region_score=regions_score)
        print('region', time.time()-region_time)

    def get_fast_score(self, candidate):
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()
        graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
        graph_mask = torch.cuda.DoubleTensor(graph_mask)
        name = candidate.name

        #load image
        img = self.load_image(name)

        #######################   corner   ###############################
        if self.cornerEvaluator is not None:
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat((img, graph_mask), 1))
            corner_map = corner_map.cpu().numpy()[0,0]
            corners_state = self.corner_map2score(corners, corner_map)
            corners_score = np.array(corners_state)
            graph.store_score(corner_score=corners_score)

        #######################   edge     ###############################
        edge_elements = graph.getEdges()
        edge_index = []
        for edge_i in range(len(edge_elements)):
            if edge_elements[edge_i].get_score() is None:
                edge_index.append(edge_i)

        if self.edgeEvaluator is not None:
            batchs = patch_samples(len(edge_index), 16)

            edge_score = np.array([])
            for batch in batchs:
                temp_masks = []
                for edge_i in batch:
                    a = edges[edge_index[edge_i],0]
                    b = edges[edge_index[edge_i],1]
                    temp_mask = cv2.line(np.ones((256,256))*-1, (int(corners[a,1]), int(corners[a,0])),
                                         (int(corners[b,1]), int(corners[b,0])),
                                         1.0, thickness=2)[np.newaxis, ...]
                    temp_mask = torch.Tensor(temp_mask).unsqueeze(0).double()
                    temp_masks.append(temp_mask)
                temp_masks = torch.cat(temp_masks, 0)
                with torch.no_grad():
                    edge_masks = temp_masks.cuda()
                    #####
                    graph_mask_ex = graph_mask.expand(edge_masks.shape[0], -1, -1, -1)
                    images = img.expand(edge_masks.shape[0], -1, -1, -1)
                    out = self.edgeEvaluator(torch.cat((images, graph_mask_ex, edge_masks), 1))
                out = out.cpu()
                edge_batch_score = out.exp()[:,0] / out.exp().sum(1)
                edge_batch_score = edge_batch_score.numpy()
                edge_score = np.append(edge_score, edge_batch_score)
            edges_state = []
            for edge_i in range(edge_score.shape[0]):
                edges_state.append(-1.9*edge_score[edge_i]*edge_score[edge_i]-0.1*edge_score[edge_i]+1)
            edges_score = np.array(edges_state)
            for idx, edge_i in enumerate(edge_index):
                edge_elements[edge_i].store_score(edges_score[idx])

            #######################   region   ###############################
        if self.region_use_heatmap:
            if 'entire' in self.regionHeatmapPath:
                gt_mask = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'))
                gt_mask = gt_mask > 0.4
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                all_masks = []
                regions_number = []
                for region_i in range(1, labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    all_masks.append(the_region)
                    regions_number.append(region_i)

                pred_mask = (np.sum(all_masks, 0) + (1 - conv_mask))>0

                iou = IOU(pred_mask, gt_mask)
                regions_score = np.array([iou])
            else:
                gt_masks = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'), allow_pickle=True)
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                false_region_id = []
                region_num = 0
                pred_region_masks = []
                for region_i in range(1,labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    region_num += 1
                    pred_region_masks.append((the_region, region_i))

                temp_used_pred = set()
                temp_gt_map = []
                pred_map_gt = [-1 for _ in range(len(pred_region_masks))]
                for gt_i in range(gt_masks.shape[0]):
                    best_iou = 0.5
                    best_pred_idx = -1
                    for pred_i in range(len(pred_region_masks)):
                        if pred_i in temp_used_pred:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_i
                    if best_pred_idx != -1:
                        temp_used_pred.add(best_pred_idx)
                        pred_map_gt[best_pred_idx] = gt_i
                    temp_gt_map.append(best_pred_idx)
                for pred_i in range(len(pred_region_masks)):
                    if pred_map_gt[pred_i] == -1:
                        best_iou = 0
                        best_gt_idx = -1
                        for gt_i in range(gt_masks.shape[0]):
                            if temp_gt_map[gt_i] != -1:
                                continue
                            iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_i
                        pred_map_gt[pred_i] = best_gt_idx

                regions_state = []
                regions_number = []
                for pred_i in range(len(pred_region_masks)):
                    gt_idx = pred_map_gt[pred_i]
                    if gt_idx == -1:
                        iou = 0
                    else:
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_idx])
                    regions_state.append(iou*2-1)
                    regions_number.append(pred_region_masks[pred_i][1])

                regions_score = np.array([regions_state])

            graph.store_score(region_score=regions_score)



    def get_score(self, candidate):
        '''
        :param candidate: class Candidate.
        :return:
        '''
        graph = candidate.graph
        corners = graph.getCornersArray()
        edges = graph.getEdgesArray()
        graph_mask = render(corners, edges, -1, 2)[np.newaxis, ...]
        graph_mask = torch.Tensor(graph_mask).double()
        graph_mask = graph_mask.cuda()
        name = candidate.name

        # load image
        img = self.load_image(name)

        #######################   corner   ###############################
        if self.cornerEvaluator is not None:
            with torch.no_grad():
                corner_map = self.cornerEvaluator(torch.cat((img, graph_mask), 1))
            corner_map = corner_map.cpu().numpy()[0,0]
            corners_state = self.corner_map2score(corners, corner_map)
            corners_score = np.array(corners_state)

        #######################   edge     ###############################
        if self.edgeEvaluator is not None:
            batchs = patch_samples(edges.shape[0], 16)

            edge_score = np.array([])
            for batch in batchs:
                temp_masks = []
                for edge_i in batch:
                    a = edges[edge_i,0]
                    b = edges[edge_i,1]
                    temp_mask = cv2.line(np.ones((256,256))*-1, (int(corners[a,1]), int(corners[a,0])),
                                         (int(corners[b,1]), int(corners[b,0])),
                                         1.0, thickness=2)[np.newaxis, ...]
                    temp_mask = torch.Tensor(temp_mask).unsqueeze(0).double()
                    temp_masks.append(temp_mask)
                temp_masks = torch.cat(temp_masks, 0)
                with torch.no_grad():
                    edge_masks = temp_masks.cuda()
                    #####
                    graph_mask_ex = graph_mask.expand(edge_masks.shape[0], -1, -1, -1)
                    images = img.expand(edge_masks.shape[0], -1, -1, -1)
                    out = self.edgeEvaluator(torch.cat((images, graph_mask_ex, edge_masks), 1))
                out = out.cpu()
                edge_batch_score = out.exp()[:,0] / out.exp().sum(1)
                edge_batch_score = edge_batch_score.numpy()
                edge_score = np.append(edge_score, edge_batch_score)
            edges_state = []
            for edge_i in range(edge_score.shape[0]):
                edges_state.append(-1.9*edge_score[edge_i]*edge_score[edge_i]-0.1*edge_score[edge_i]+1)
            edges_score = np.array(edges_state)

        #######################   region   ###############################
        if self.region_use_heatmap:
            if 'entire' in self.regionHeatmapPath:
                gt_mask = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'))
                gt_mask = gt_mask > 0.4
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                all_masks = []
                regions_number = []
                for region_i in range(1, labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    all_masks.append(the_region)
                    regions_number.append(region_i)

                pred_mask = (np.sum(all_masks, 0) + (1 - conv_mask))>0

                iou = IOU(pred_mask, gt_mask)
                regions_score = np.array([iou])
            else:
                gt_masks = np.load(os.path.join(self.regionHeatmapPath, name+'.npy'), allow_pickle=True)
                conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
                conv_mask = 1 - conv_mask
                conv_mask = conv_mask.astype(np.uint8)
                labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)

                background_label = region_mask[0,0]
                false_region_id = []
                region_num = 0
                pred_region_masks = []
                for region_i in range(1,labels):
                    if region_i == background_label:
                        continue
                    the_region = region_mask == region_i
                    if the_region.sum() < 20:
                        continue
                    region_num += 1
                    pred_region_masks.append((the_region, region_i))

                temp_used_pred = set()
                temp_gt_map = []
                pred_map_gt = [-1 for _ in range(len(pred_region_masks))]
                for gt_i in range(gt_masks.shape[0]):
                    best_iou = 0.5
                    best_pred_idx = -1
                    for pred_i in range(len(pred_region_masks)):
                        if pred_i in temp_used_pred:
                            continue
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_i
                    if best_pred_idx != -1:
                        temp_used_pred.add(best_pred_idx)
                        pred_map_gt[best_pred_idx] = gt_i
                    temp_gt_map.append(best_pred_idx)
                for pred_i in range(len(pred_region_masks)):
                    if pred_map_gt[pred_i] == -1:
                        best_iou = 0
                        best_gt_idx = -1
                        for gt_i in range(gt_masks.shape[0]):
                            if temp_gt_map[gt_i] != -1:
                                continue
                            iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_i])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_i
                        pred_map_gt[pred_i] = best_gt_idx

                regions_state = []
                regions_number = []
                for pred_i in range(len(pred_region_masks)):
                    gt_idx = pred_map_gt[pred_i]
                    if gt_idx == -1:
                        iou = 0
                    else:
                        iou = IOU(pred_region_masks[pred_i][0], gt_masks[gt_idx])
                    regions_state.append(iou*2-1)
                    regions_number.append(pred_region_masks[pred_i][1])

                regions_score = np.array([regions_state])

        graph.store_score(corner_score=corners_score, edge_score=edges_score,
                          region_score=regions_score)

