import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from new_utils import *
import pickle
from new_scoreAgent import scoreEvaluator_with_train

#########
# events_type = 1
# 1. two different nodes, remove edge
# 2. same node, remove node
# events_type = 2
# 1. pick two different nodes, and add a new edge
#########
def get_corner_id(loc, corners):
    loc = np.array(loc)
    temp_dist = 1000
    temp_id = -1
    for i in range(corners.shape[0]):
        d = np.sqrt(np.sum((corners[i] - loc)**2))
        if temp_dist > d:
            temp_dist = d
            temp_id = i

    return temp_id

class render_interactive():
    def __init__(self, candidate, evaluator):
        self.events = []
        self.events_type = []
        self.init_candidate = candidate
        self.eval = evaluator
        self.curr_candidate = candidate.copy()
        self.new_edge_cache = []


    def start(self):
        def draw(curr_candidate):
            # interactive figure
            mask = render(curr_candidate.graph.getCornersArray(), curr_candidate.graph.getEdgesArray())
            mask = np.concatenate((np.transpose(mask, (1,2,0)), np.zeros((256,256,1))), 2)
            ax1.imshow(mask)
            ax1.set_title(str(self.curr_candidate.graph.graph_score()))

            # corner map
            temp_mask = np.zeros((256,256))
            temp_mask[0:8,:] = np.arange(256)/255
            for ele in curr_candidate.graph.getCorners():
                temp_mask = cv2.circle(temp_mask, ele.x[::-1], 3, (-ele.get_score()+1)/2, -1)
            ax4.imshow(temp_mask, vmin=0, vmax=1)
            ax4.set_title(str(self.curr_candidate.graph.corner_score()))
            # edge map
            temp_mask = np.zeros((256,256))
            temp_mask[0:8,:] = np.arange(256)/255
            for ele in curr_candidate.graph.getEdges():
                A = ele.x[0]
                B = ele.x[1]
                temp_mask = cv2.line(temp_mask, A.x[::-1], B.x[::-1], (-ele.get_score()+1)/2, thickness=2)
            ax5.imshow(temp_mask, vmin=0, vmax=1)
            ax5.set_title(str(self.curr_candidate.graph.edge_score()))

            ax6.set_title(str(self.curr_candidate.graph.region_score()))

        def on_press(event):
            print(event.ydata, event.xdata, event.x, event.y, event.inaxes)
            if event.inaxes == ax1:
                if event.button == 1:
                    # delete
                    # corner first
                    new_graph = Graph(self.curr_candidate.graph.getCornersArray(), self.curr_candidate.graph.getEdgesArray())
                    delete_corner = None
                    dist = 5
                    for corner in new_graph.getCorners():
                        d = l2_distance(corner.x, (event.ydata, event.xdata))
                        if d < dist:
                            delete_corner = corner
                            dist = d
                    if delete_corner is not None:
                        new_graph.remove(delete_corner)
                    else:
                        # edge
                        for edge in new_graph.getEdges():
                            corner1 = edge.x[0].x
                            corner2 = edge.x[1].x
                            line1 = np.array((event.ydata, event.xdata)) - np.array(corner1)
                            line2 = np.array((event.ydata, event.xdata)) - np.array(corner2)
                            cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
                            cos = min(1, max(-1, cos))
                            if cos < -0.95:
                                new_graph.remove(edge)
                                break
                    self.curr_candidate = Candidate.initial(new_graph, self.init_candidate.name)
                    self.eval.get_score(self.curr_candidate, all_edge=True)
                    draw(self.curr_candidate)
                elif event.button == 3:
                    # add
                    new_graph = Graph(self.curr_candidate.graph.getCornersArray(), self.curr_candidate.graph.getEdgesArray())
                    the_corner = None
                    dist = 5
                    for corner in new_graph.getCorners():
                        d = l2_distance(corner.x, (event.ydata, event.xdata))
                        if d < dist:
                            the_corner = corner
                            dist = d
                    if the_corner is None:
                        # add new corner
                        new_graph.add_corner(Element((int(event.ydata), int(event.xdata))))
                        self.curr_candidate = Candidate.initial(new_graph, self.init_candidate.name)
                        self.eval.get_score(self.curr_candidate, all_edge=True)
                        draw(self.curr_candidate)
                    else:
                        # add new edge
                        if len(self.new_edge_cache) == 1:
                            corner1 = new_graph.getRealElement(self.new_edge_cache.pop(0))
                            new_graph.add_edge(the_corner, corner1)
                            self.curr_candidate = Candidate.initial(new_graph, self.init_candidate.name)
                            self.eval.get_score(self.curr_candidate, all_edge=True)
                            draw(self.curr_candidate)
                        else:
                            self.new_edge_cache.append(the_corner)

            else:
                self.curr_candidate = self.init_candidate.copy()
                self.eval.get_score(self.curr_candidate)
                draw(self.curr_candidate)

            fig.canvas.draw_idle()
        self.events = []
        self.events_type = []
        fig = plt.figure()
        fig.canvas.mpl_connect("button_press_event", on_press)


        self.eval.get_score(self.curr_candidate, all_edge=True)

        # interactive figure
        ax1 = fig.add_subplot(231)
        mask = render(self.curr_candidate.graph.getCornersArray(), self.curr_candidate.graph.getEdgesArray())
        mask = np.concatenate((np.transpose(mask, (1,2,0)), np.zeros((256,256,1))), 2)
        ax1.imshow(mask)
        ax1.set_title(str(self.curr_candidate.graph.graph_score()))
        # rgb
        ax2 = fig.add_subplot(232)
        name = self.init_candidate.name
        img = skimage.img_as_float(plt.imread(os.path.join('/local-scratch/fuyang/cities_dataset/rgb', name+'.jpg')))
        ax2.imshow(img)
        # gt
        ax3 = fig.add_subplot(233)
        gt_data = np.load(os.path.join('/local-scratch/fuyang/cities_dataset/data/gt', name+'.npy'), allow_pickle=True).tolist()
        gt_mask = render(gt_data['corners'], gt_data['edges'])
        gt_mask = np.concatenate((np.transpose(gt_mask, (1,2,0)), np.zeros((256,256,1))), 2)
        ax3.imshow(gt_mask)

        # corner map
        ax4 = fig.add_subplot(234)
        temp_mask = np.zeros((256,256))
        temp_mask[0:8,:] = np.arange(256)/255
        for ele in self.curr_candidate.graph.getCorners():
            temp_mask = cv2.circle(temp_mask, ele.x[::-1], 3, (-ele.get_score()+1)/2, -1)
        ax4.imshow(temp_mask, vmin=0, vmax=1)
        ax4.set_title(str(self.curr_candidate.graph.corner_score()))
        # edge map
        ax5 = fig.add_subplot(235)
        temp_mask = np.zeros((256,256))
        temp_mask[0:8,:] = np.arange(256)/255
        for ele in self.curr_candidate.graph.getEdges():
            A = ele.x[0]
            B = ele.x[1]
            temp_mask = cv2.line(temp_mask, A.x[::-1], B.x[::-1], (-ele.get_score()+1)/2, thickness=2)
        ax5.imshow(temp_mask, vmin=0, vmax=1)
        ax5.set_title(str(self.curr_candidate.graph.edge_score()))

        # maskrcnn
        ax6 = fig.add_subplot(236)
        maskrcnn_mask = plt.imread('/local-scratch/fuyang/result/corner_edge_region/entire_region_mask/'+name+'.png')
        ax6.imshow(maskrcnn_mask)
        ax6.set_title(str(self.curr_candidate.graph.region_score()))
        plt.show()
        plt.close()


f_name = '/local-scratch/fuyang/result/beam_search_v2/strong_constraint_heatmap_without_corner_heatmap/' \
         'valid_prefix_4_result/1554266277.87/iter_0_num_0.obj'
#f_name = '/local-scratch/fuyang/result/beam_search_v2/strong_constraint/valid_result/1553902528.15/iter_0_num_0.obj'
f = open(f_name, 'rb')
save_path = '/local-scratch/fuyang/result/beam_search_v2/strong_constraint_grad_to_heatmap_big'
evaluator_search = scoreEvaluator_with_train('/local-scratch/fuyang/cities_dataset',
                                             backbone_channel=64, edge_bin_size=36)
evaluator_search.load_weight(save_path, '6')
evaluator_search.to('cuda:0')
evaluator_search.eval()
candidate_ = pickle.load(f)
render_ = render_interactive(candidate_, evaluator_search)
render_.start()

