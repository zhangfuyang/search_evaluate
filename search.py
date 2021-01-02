import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from dataset import myDataset
from scoreAgent import scoreEvaluator
import skimage
from utils import *
from SVG_utils import svg_generate
from PIL import Image

data_folder = '/local-scratch/fuyang/cities_dataset'
beam_width = 4
beam_depth = 5
is_visualize = False
is_save = True
save_path = '/local-scratch/fuyang/result/beam_search_new/corner_edge_region_threading_region_60/'

test_dataset = myDataset(data_folder, phase='valid', edge_linewidth=2, render_pad=-1)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=1,
                                          drop_last=True)

evaluator = scoreEvaluator({'cornerModelPath':'/local-scratch/fuyang/result/corner_v2/gt_mask_with_gt/models/440.pt',
                            'edgeModelPath':'/local-scratch/fuyang/result/corner_edge_region/edge_graph_drn26_with_search_v2/models/best.pt',
                            'regionModelPath': '/local-scratch/fuyang/result/corner_edge_region/region_graph_iter_0/models/70.pt',
                            'region_iter': 0,
                            'edgeHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/edge_heatmap_unet/all_edge_masks',
                            'regionHeatmapPath': '/local-scratch/fuyang/result/corner_edge_region/all_region_masks',
                            'regionEntireMaskPath': '/local-scratch/fuyang/result/corner_edge_region/entire_region_mask'
                            }, useHeatmap=('region'), useGT=(), dataset=test_dataset, continuous_score=True)

for idx, data in enumerate(test_loader):
    name = data['name'][0]
    img = data['img'][0]
    graph_data = test_dataset.getDataByName(name)
    conv_data = graph_data['conv_data']
    corners = conv_data['corners']
    edges = conv_data['edges']
    if edges.shape[0] <= 50:
        continue
    gt_data = graph_data['gt_data']
    gt_corners = gt_data['corners']
    gt_edges = gt_data['edges']
    gt_corners, gt_edges = sort_graph(gt_corners, gt_edges)
    gt_score, gt_config = evaluator.get_score(img=img, corners=gt_corners, edges=gt_edges, name=name)

    corners, edges = sort_graph(corners, edges)
    epoch = min(beam_depth, corners.shape[0])

    original_score, original_config = evaluator.get_score(img=img, corners=corners, edges=edges, name=name)


    prev_graphs = [(corners, edges, np.zeros(corners.shape[0]), np.zeros(edges.shape[0]), edges)]
    # corners, edges, corner_flag, edge_flage, edge_existed_once
    prev_score = [original_score]
    prev_config = [original_config]
    graph_gallery = []
    graph_gallery.append({'graph': prev_graphs.copy(), 'score': prev_score.copy(), 'config': prev_config.copy()})
    print(name)
    print(edges.shape[0])
    for epoch_i in range(epoch):
        print("======================== epoch ", epoch_i, " =======================")
        start_time = time.time()
        current_graphs = []
        for prev_i in range(len(prev_graphs)):
            print("==================prev ", prev_i, "========================")
            prev_ = prev_graphs[prev_i]
            current_graphs.extend(graph_enumerate(prev_))

        print("all prev graph enumerate done.", "overall", len(current_graphs), "graphs")

        current_graphs = reduce_duplicate_graph(current_graphs)

        print("reduce duplicate graph done.", "overall", len(current_graphs), "graphs")

        candidates_score, candidates_config = evaluator.get_list_candidates_score(img, current_graphs, name=name)

        print("finish evaluating all candidates")
        if len(candidates_score) < beam_width:
            pick = np.arange(len(candidates_score))
        else:
            pick = np.argsort(candidates_score)[-beam_width:]

        prev_graphs = [current_graphs[temp_i] for temp_i in pick]
        prev_score = [candidates_score[temp_i] for temp_i in pick]
        prev_config = [candidates_config[temp_i] for temp_i in pick]

        graph_gallery.append({'graph': prev_graphs.copy(), 'score': prev_score.copy(), 'config': prev_config.copy()})
        print("========================finish epoch: ", epoch_i, "=====================================")
        print("spend time: ", (time.time()-start_time)/60)

    best_score = -1000
    best_graphs = []
    best_scores = []
    best_configs = []
    for k in range(len(graph_gallery)):
        current_graphs = graph_gallery[k]['graph']
        current_scores = graph_gallery[k]['score']
        current_configs = graph_gallery[k]['config']
        for graph_i in range(len(current_graphs)):
            if current_scores[graph_i] > best_score:
                best_score = current_scores[graph_i]
                best_graphs = [current_graphs[graph_i]]
                best_scores = [current_scores[graph_i]]
                best_configs = [current_configs[graph_i]]
            elif current_scores[graph_i] == best_score:
                best_graphs.append(current_graphs[graph_i])
                best_scores.append(current_scores[graph_i])
                best_configs.append(current_configs[graph_i])
    if is_save:
        base_path = os.path.join(save_path, name)
        os.makedirs(base_path, exist_ok=True)
        base_name = 'gt_pred'
        svg = svg_generate(gt_corners, gt_edges, base_name, samecolor=True)
        svg.saveas(os.path.join(base_path, base_name+'.svg'))
        dump_config_file(gt_corners, gt_edges, gt_score, gt_config, base_path, base_name)

        for k in range(len(graph_gallery)):
            current_graphs = graph_gallery[k]['graph']
            current_scores = graph_gallery[k]['score']
            current_configs = graph_gallery[k]['config']
            for graph_i in range(len(current_graphs)):
                corners_ = current_graphs[graph_i][0]
                edges_ = current_graphs[graph_i][1]
                base_name = 'iter_'+str(k)+'_num_'+str(graph_i)
                svg = svg_generate(corners_, edges_, base_name, samecolor=True)
                svg.saveas(os.path.join(base_path, base_name+'.svg'))
                dump_config_file(corners_, edges_, current_scores[graph_i],
                                 current_configs[graph_i], base_path, base_name)

        ##### best ####
        fig = plt.figure(frameon=False)
        fig.set_size_inches(1,1)
        ax = plt.Axes(fig, [0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        for graph_i in range(len(best_graphs)):
            corners_ = best_graphs[graph_i][0]
            edges_ = best_graphs[graph_i][1]
            base_name = 'best_'+str(graph_i)
            svg = svg_generate(corners_, edges_, base_name, samecolor=True)
            svg.saveas(os.path.join(base_path, base_name+'.svg'))
            dump_config_file(corners_, edges_, best_scores[graph_i], best_configs[graph_i],
                             base_path, base_name)
            break

    ### visualize graphs
    if is_visualize:
        img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        original_mask = render(corners, edges)
        for k in range(len(graph_gallery)):
            current_graphs = graph_gallery[k]['graph']
            current_scores = graph_gallery[k]['score']
            current_configs = graph_gallery[k]['config']
            for graph_i in range(len(current_graphs)):
                plt.subplot(3,4,1)
                plt.title('iter: '+str(k))
                plt.imshow(img)
                corners_ = current_graphs[graph_i][0]
                edges_ = current_graphs[graph_i][1]
                next_i = visualize_config(gt_corners, gt_edges, gt_config, gt_score, 2,3,4)

                next_i = visualize_config(corners_, edges_, current_configs[graph_i],
                                 current_scores[graph_i], next_i, 3, 4)

                plt.subplot(3,4,next_i)
                plt.title('original graph')
                plt.imshow(original_mask[0]+original_mask[1])
                plt.show()





