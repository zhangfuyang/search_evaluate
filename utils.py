import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt
import threading

def IOU(maskA, maskB):
    return np.logical_and(maskA, maskB).sum() / np.logical_or(maskA, maskB).sum()

def EuclideanDistance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)

    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED<0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def getcornerdirection(corner_id, corners, edges):
    place = np.where(edges == corner_id)
    neighbor_id = edges[place[0], 1-place[1]]

    distance = corners[corner_id] - corners[neighbor_id]
    corner_dir = np.arctan2(distance[:, 0], distance[:, 1]) * 180 / np.pi + 185
    corner_dir = corner_dir % 360
    corner_dir = corner_dir / 18
    corner_dir = np.round(corner_dir).astype(np.int)
    corner_dir[np.where(corner_dir==20)] = 0

    out = np.zeros(20)
    out[corner_dir] = 1
    return out


def samedirection(conv_corner_id, gt_corner_id, conv_corners, gt_corners, conv_edges, gt_edges):
    # degree
    if np.where(conv_edges == conv_corner_id)[0].shape[0] != np.where(gt_edges == gt_corner_id)[0].shape[0]:
        return False

    # direction
    place = np.where(conv_edges == conv_corner_id)
    neighbor_id = conv_edges[place[0], 1-place[1]]

    distance = conv_corners[conv_corner_id] - conv_corners[neighbor_id]
    conv_dir = np.arctan2(distance[:,0], distance[:,1]) * 180 / np.pi


    place = np.where(gt_edges == gt_corner_id)
    neighbor_id = gt_edges[place[0], 1-place[1]]

    distance = gt_corners[gt_corner_id] - gt_corners[neighbor_id]
    gt_dir = np.arctan2(distance[:,0], distance[:,1]) * 180 / np.pi

    flag = True
    for dir_i in range(conv_dir.shape[0]):
        temp = (gt_dir - conv_dir[dir_i]) % 360
        temp_id = -1
        temp_degree = 360
        for dir_j in range(temp.shape[0]):
            x = temp[dir_j] if 360 - temp[dir_j] > temp[dir_j] else 360 - temp[dir_j]
            if x < temp_degree:
                temp_degree = x
                temp_id = dir_j
        if temp_degree > 15:
            flag = False
            break

        gt_dir = np.delete(gt_dir, temp_id)

    return flag


def remove_dangling_edge_corner(corners, edges):
    while True:
        flag = False
        for corner_i in range(corners.shape[0]):
            if np.where(edges == corner_i)[0].shape[0] <= 1:
                flag = True
                corners, edges = remove_a_corner(corner_i, corners, edges)
                break
        if flag:
            continue
        break
    return (corners, edges)

def get_wrong_corners(corners, gt_corners, edges, gt_edges):
    dist_matrix = EuclideanDistance(gt_corners, corners)
    assigned_id = set()
    gt_match = []
    conv2gt_map = {}
    corners_gt_dir = np.zeros((corners.shape[0], 20))
    for gt_i in range(gt_corners.shape[0]):
        sort_id = np.argsort(dist_matrix[gt_i]).__array__()[0]
        flag = True
        for id_ in sort_id:
            if dist_matrix[gt_i, id_] > 7:
                break
            temete = samedirection(id_, gt_i, corners, gt_corners, edges, gt_edges)
            corners_gt_dir[id_] = getcornerdirection(gt_i, gt_corners, gt_edges)
            if temete == False:
                break
            elif id_ not in assigned_id:
                assigned_id.add(id_)
                gt_match.append(id_)
                conv2gt_map[id_] = gt_i
                flag = False
                break
        if flag:
            gt_match.append(None)

    return set(range(corners.shape[0])) - assigned_id, gt_match, conv2gt_map, corners_gt_dir


def distance2edgeMap(corner1, corner2):
    x0, y0 = corner1[1], corner1[0]
    x1, y1 = corner2[1], corner2[0]
    a = y1-y0
    b = -x1+x0
    c = x1*y0-x0*y1

    #l1, l2
    c0 = a*y0-b*x0
    c1 = a*y1-b*x1

    # coord map
    coord_x = np.tile(np.arange(256), 256).reshape(256,256)
    coord_y = coord_x.T
    coord = np.stack((coord_y, coord_x), 2)

    mask = np.min((np.sqrt(np.sum((corner1-coord)**2,2)), np.sqrt(np.sum((corner2-coord)**2,2))), 0)

    place = np.where((b*coord[:,:,1]-a*coord[:,:,0]+c0)*(b*coord[:,:,1]-a*coord[:,:,0]+c1) < 0)
    mask[place] = np.abs(a*place[1]+b*place[0]+c) / np.sqrt(a**2+b**2)

    return mask

def render_dir_map(corners, dirs, render_pad=0, corner_size=3):
    mask = np.zeros((20,256,256)) * render_pad
    for corner_i in range(corners.shape[0]):
        loc = np.round(corners[corner_i]).astype(np.int)
        for i in range(corner_size):
            for j in range(corner_size-i):
                if loc[0]+i < 256 and loc[1]+j < 256:
                    mask[:,loc[0]+i, loc[1]+j] = dirs[corner_i]
                if loc[0]+i < 256 and loc[1]-j >= 0:
                    mask[:,loc[0]+i, loc[1]-j] = dirs[corner_i]
                if loc[0]-i >= 0 and loc[1]+j < 256:
                    mask[:,loc[0]-i, loc[1]+j] = dirs[corner_i]
                if loc[0]-i >= 0 and loc[1]-j >= 0:
                    mask[:,loc[0]-i, loc[1]-j] = dirs[corner_i]
    return mask


def render(corners, edges, render_pad=0, edge_linewidth=2, corner_size=3):
    mask = np.ones((2,256, 256)) * render_pad
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a,1]), int(corners[a,0])),
                           (int(corners[b,1]), int(corners[b,0])), 1.0, thickness=edge_linewidth)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i,1]), int(corners[corner_i,0])), corner_size, 1.0, -1)

    return mask


def render_edge(corners, edges, render_pad=0, edge_linewidth=2):
    mask = np.ones((1,256, 256)) * render_pad
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a,1]), int(corners[a,0])),
                           (int(corners[b,1]), int(corners[b,0])), 1.0, thickness=edge_linewidth)

    return mask


def render_corner(corners, render_pad=0):
    mask = np.ones((1,256, 256)) * render_pad
    for corner_i in range(corners.shape[0]):
        mask[0] = cv2.circle(mask[0], (int(corners[corner_i,1]), int(corners[corner_i,0])), 3, 1.0, -1)

    return mask

def add_one_edge(new_edge, edges, edges_flag, edges_existed, flag_value=1):
    if new_edge[0] > new_edge[1]:
        temp = new_edge[0].copy()
        new_edge[0] = new_edge[1]
        new_edge[1] = temp
    place = 0
    while place < edges_existed.shape[0]:
        if (edges_existed[place,0] > new_edge[0]) or \
                (edges_existed[place,0] == new_edge[0] and edges_existed[place,1] > new_edge[1]):
            break
        if edges_existed[place,0] == new_edge[0] and edges_existed[place,1] == new_edge[1]:
            return None
        place += 1
    edges_existed_new = np.insert(edges_existed, place, new_edge, 0)

    place = 0
    while place < edges.shape[0]:
        if (edges[place,0] > new_edge[0]) or \
                (edges[place,0] == new_edge[0] and edges[place,1] > new_edge[1]):
            break
        place += 1
    edges_new = np.insert(edges, place, new_edge, 0)
    edges_flag_new = np.insert(edges_flag, place, flag_value)

    return edges_new, edges_flag_new, edges_existed_new

def add_an_edge(corner1, corner2, corners, edges, corners_flag, edges_flag, edges_existed):
    result = add_one_edge([corner1, corner2], edges, edges_flag, edges_existed)
    if result is None:
        return None
    else:
        new_edges, new_edges_flag, new_edges_existed = result
    return (corners, new_edges, corners_flag, new_edges_flag, new_edges_existed)

def remove_a_corner(cornerID, corners, edges, corners_flag, edges_flag, edges_existed):
    if corners_flag[cornerID] != 0: # added corner
        return None
    new_corners = np.delete(corners, cornerID, 0)
    new_corners_flag = np.delete(corners_flag, cornerID)
    place = np.where(edges == cornerID)
    new_edges = np.delete(edges, place[0], 0)
    new_edges_flag = np.delete(edges_flag, place[0])
    new_edges_existed = np.delete(edges_existed, place[0], 0)
    for edges_i in range(new_edges.shape[0]):
        if new_edges[edges_i, 0] > cornerID:
            new_edges[edges_i, 0] = new_edges[edges_i, 0] - 1
        if new_edges[edges_i, 1] > cornerID:
            new_edges[edges_i, 1] = new_edges[edges_i, 1] - 1
    for edges_i in range(new_edges_existed.shape[0]):
        if new_edges_existed[edges_i, 0] > cornerID:
            new_edges_existed[edges_i, 0] = new_edges_existed[edges_i, 0] - 1
        if new_edges_existed[edges_i, 1] > cornerID:
            new_edges_existed[edges_i, 1] = new_edges_existed[edges_i, 1] - 1

    return (new_corners, new_edges, new_corners_flag, new_edges_flag, new_edges_existed)

def remove_an_edge(edgeID, corners, edges, corners_flag, edges_flag, edges_existed):
    if edges_flag[edgeID] != 0:
        return None
    corner1 = edges[edgeID,0]
    corner2 = edges[edgeID,1]
    if corner1 < corner2:
        corner1 = edges[edgeID,1]
        corner2 = edges[edgeID,0]
    remove_corner_list = []
    if np.where(edges == corner1)[0].shape[0] == 1:
        remove_corner_list.append(corner1)
    if np.where(edges == corner2)[0].shape[0] == 1:
        remove_corner_list.append(corner2)
    new_edges = np.delete(edges, edgeID, 0)
    new_edges_flag = np.delete(edges_flag, edgeID)
    new_corners = corners.copy()
    new_corners_flag = corners_flag
    new_edges_existed = edges_existed
    for cornerid in remove_corner_list:
         new_corners, new_edges, new_corners_flag, new_edges_flag, new_edges_existed = \
             remove_a_corner(cornerid, new_corners, new_edges, new_corners_flag, new_edges_flag, edges_existed)
    return (new_corners, new_edges, new_corners_flag, new_edges_flag, new_edges_existed)

def get_corner_degree(cornerID, corners, edges):
    place = np.where(edges == cornerID)
    return place[0].shape[0]

def has_edge(corner1, corner2, edges):
    for edge_i in range(edges.shape[0]):
        if edges[edge_i,0] == corner1 and edges[edge_i,1] == corner2:
            return True
        if edges[edge_i,1] == corner1 and edges[edge_i,0] == corner2:
            return True
    return False

def check_corner_colinear(cornerID, corners, edges):
    if get_corner_degree(cornerID, corners, edges) != 2:
        return False
    place = np.where(edges == cornerID)
    neighbor_id = edges[place[0], 1-place[1]]
    if has_edge(neighbor_id[0], neighbor_id[1], edges):
        return False
    line1 = corners[cornerID] - corners[neighbor_id[0]]
    line2 = corners[neighbor_id[1]] - corners[cornerID]
    cos = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
    if np.arccos(cos) < np.pi / 8: # 20 degree
        return True
    return False

def remove_colinear_corner(cornerID, corners, edges, corners_flag, edges_flag, edges_existed):
    """
    should check corner degree before call this function
    """
    place = np.where(edges == cornerID)
    neighbor_id = edges[place[0], 1-place[1]]
    result = add_one_edge(neighbor_id, edges, edges_flag, edges_existed, flag_value=0)
    if result is None:
        return None
    edges_new, edges_flag_new, edges_existed_new = result
    return remove_a_corner(cornerID, corners, edges_new, corners_flag, edges_flag_new, edges_existed_new)

def reduce_duplicate_graph(graphs):
    while True:
        flag = False
        for graph_i in range(len(graphs)):
            cornersA = graphs[graph_i][0]
            edgesA = graphs[graph_i][1]
            cornersA_flag = graphs[graph_i][2]
            edgesA_flag = graphs[graph_i][3]
            assert cornersA.shape[0] == cornersA_flag.shape[0]
            assert edgesA.shape[0] == edgesA_flag.shape[0]
            for graph_j in range(graph_i+1, len(graphs)):
                cornersB = graphs[graph_j][0]
                edgesB = graphs[graph_j][1]
                if cornersA.shape[0] != cornersB.shape[0] or edgesA.shape[0] != edgesB.shape[0]:
                    continue
                if (cornersA - cornersB).sum() == 0 and (edgesA - edgesB).sum() == 0:
                    #TODO need to change if allow adding corners
                    # Or if corners and edges have order
                    flag = True
                    del graphs[graph_j]
                    break
            if flag:
                break
        if flag is False:
            break
    return graphs

def swap_two_corner_place(corners, edges, id1, id2):
    for edge_i in range(edges.shape[0]):
        if edges[edge_i,0] == id1:
            edges[edge_i,0] = id2
        elif edges[edge_i, 0] == id2:
            edges[edge_i,0] = id1
        if edges[edge_i,1] == id1:
            edges[edge_i,1] = id2
        elif edges[edge_i, 1] == id2:
            edges[edge_i,1] = id1
    temp = corners[id1].copy()
    corners[id1] = corners[id2]
    corners[id2] = temp
    return corners, edges

def swap_two_edge_place(edges, id1, id2):
    temp = edges[id1].copy()
    edges[id1] = edges[id2]
    edges[id2] = temp
    return edges


def sort_graph(corners, edges):

    for corner_i in range(corners.shape[0]):
        min_id = -1
        min_pos = corners[corner_i]
        for corner_j in range(corner_i+1, corners.shape[0]):
            if (corners[corner_j,0] < min_pos[0]) or \
                    (corners[corner_j,0]==min_pos[0] and corners[corner_j,1]<min_pos[1]):
                min_pos = corners[corner_j]
                min_id = corner_j
        if min_id != -1:
            corners, edges = swap_two_corner_place(corners, edges, corner_i, min_id)

    for edge_i in range(edges.shape[0]):
        if edges[edge_i,0] > edges[edge_i,1]:
            temp = edges[edge_i,0]
            edges[edge_i,0] = edges[edge_i,1]
            edges[edge_i,1] = temp

    for edge_i in range(edges.shape[0]):
        min_id = -1
        min_pos = edges[edge_i]
        for edge_j in range(edge_i+1, edges.shape[0]):
            if (edges[edge_j,0] < min_pos[0]) or \
                    (edges[edge_j,0]==min_pos[0] and edges[edge_j,1]<min_pos[1]):
                min_pos = edges[edge_j]
                min_id = edge_j
        if min_id != -1:
            edges = swap_two_edge_place(edges, edge_i, min_id)

    return corners, edges


def remove_a_corner_wrap(corners, edges, flag=None):
    candidates = []
    if flag is None:
        corners_flag = np.zeros(corners.shape[0])
        edges_flag = np.zeros(edges.shape[0])
        edges_existed = edges.copy()
    else:
        corners_flag = flag[0]
        edges_flag = flag[1]
        edges_existed = flag[2]
    for corner_i in range(corners.shape[0]):
        result = remove_a_corner(corner_i, corners, edges, corners_flag, edges_flag, edges_existed)
        if result is not None:
            candidates.append(result)
    return candidates


def remove_colinear_corner_wrap(corners, edges, flag=None):
    candidates = []
    if flag is None:
        corners_flag = np.zeros(corners.shape[0])
        edges_flag = np.zeros(edges.shape[0])
        edges_existed = edges.copy()
    else:
        corners_flag = flag[0]
        edges_flag = flag[1]
        edges_existed = flag[2]
    for corner_i in range(corners.shape[0]):
        if check_corner_colinear(corner_i, corners, edges):
            result = remove_colinear_corner(corner_i, corners, edges, corners_flag, edges_flag, edges_existed)
            if result is not None:
                candidates.append(result)
    return candidates

def remove_an_edge_wrap(corners, edges, flag=None):
    candidates = []
    if flag is None:
        corners_flag = np.zeros(corners.shape[0])
        edges_flag = np.zeros(edges.shape[0])
        edges_existed = edges.copy()
    else:
        corners_flag = flag[0]
        edges_flag = flag[1]
        edges_existed = flag[2]
    for edge_i in range(edges.shape[0]):
        result = remove_an_edge(edge_i, corners, edges, corners_flag, edges_flag, edges_existed)
        if result is not None:
            candidates.append(result)
    return candidates

def check_intersection(corner11, corner12, corner21, corner22):
    y1 = corner11[0]
    x1 = corner11[1]
    y2 = corner12[0]
    x2 = corner12[1]
    a = y1-y2
    b = x2-x1
    c = x1*y2-x2*y1
    flag1 = (a*corner21[1]+b*corner21[0]+c) * (a*corner22[1] + b*corner22[0]+c)

    y1 = corner21[0]
    x1 = corner21[1]
    y2 = corner22[0]
    x2 = corner22[1]
    a = y1-y2
    b = x2-x1
    c = x1*y2-x2*y1
    flag2 = (a*corner11[1]+b*corner11[0]+c) * (a*corner12[1] + b*corner12[0]+c)

    if flag1 < 0 and flag2 < 0:
        return True

    return False

def check_edge_colinear(corner11, corner12, corner21, corner22):

    k1 = (corner11[0] - corner12[0]) / (corner11[1] - corner12[1] + 1e-8)
    k2 = (corner21[0] - corner22[0]) / (corner21[1] - corner22[1] + 1e-8)

    angle1 = np.arctan(k1)
    angle2 = np.arctan(k2)
    delta = (angle1 - angle2)/np.pi*180 % 180
    delta = min(delta, 180 - delta)
    if delta < 5:
        return False
    # unfinished

def add_a_corner_wrap(corners, edges, flag=None):
    candidates = []
    if flag is None:
        corners_flag = np.zeros(corners.shape[0])
        edges_flag = np.zeros(edges.shape[0])
        edges_existed = edges.copy()
    else:
        corners_flag = flag[0]
        edges_flag = flag[1]
        edges_existed = flag[2]
    for x in range(0,256,10):
        for y in range(0,256,10):
            result = add_a_corner(x, y, corners, edges, corners_flag, edges_flag, edges_existed)
            if result is not None:
                candidates.append(result)
    return candidates


def add_a_corner():
    pass

def check_add_edge_correct(corners, edges, idx1, idx2):
    if has_edge(idx1, idx2, edges):
        return False
    # intersection
    for edge_i in range(edges.shape[0]):
        if check_intersection(corners[idx1], corners[idx2], corners[edges[edge_i,0]], corners[edges[edge_i,1]]):
            return False
    # colinear
    #for edge_i in range(edges.shape[0]):
    #    if check_edge_colinear(corners[idx1], corners[idx2], corners[edges[edge_i,0]], corners[edges[edge_i,1]]):
    #        return False

    return True


def add_two_unconnected_corners_wrap(corners, edges, flag=None):
    # TODO need to improve
    candidates = []
    if flag is None:
        corners_flag = np.zeros(corners.shape[0])
        edges_flag = np.zeros(edges.shape[0])
        edges_existed = edges.copy()
    else:
        corners_flag = flag[0]
        edges_flag = flag[1]
        edges_existed = flag[2]
    for corner_i in range(corners.shape[0]):
        for corner_j in range(corner_i+1, corners.shape[0]):
            if check_add_edge_correct(corners, edges, corner_i, corner_j) is False:
                continue
            result = add_an_edge(corner_i, corner_j, corners, edges, corners_flag, edges_flag, edges_existed)
            if result is not None:
                candidates.append(result)
    return candidates


class _thread(threading.Thread):
    def __init__(self, threadID, name, graph, lock, result_list, func):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.graph = graph
        self.lock = lock
        self.result_list = result_list
        self.func = func
    def run(self):
        print('running id: ', self.name)
        start_time = time.time()
        corners = self.graph[0]
        edges = self.graph[1]
        corners_flag = self.graph[2]
        edges_flag = self.graph[3]
        edges_existed = self.graph[4]
        candidates = self.func(corners, edges, (corners_flag, edges_flag, edges_existed))
        self.lock.acquire()
        self.result_list.extend(candidates)
        self.lock.release()
        print(self.name, "spend time: ", (time.time()-start_time)/60)

def graph_enumerate(graph):
    candidates = []
    lock = threading.Lock()

    thread1 = _thread(1, 'remove_a_corner', graph, lock, candidates, remove_a_corner_wrap)
    thread2 = _thread(2, 'remove_colinear_corner', graph, lock, candidates, remove_colinear_corner_wrap)
    thread3 = _thread(3, 'remove_an_edge', graph, lock, candidates, remove_an_edge_wrap)
    thread4 = _thread(4, 'add_an_edge', graph, lock, candidates, add_two_unconnected_corners_wrap)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    threads = []
    threads.append(thread1)
    threads.append(thread2)
    threads.append(thread3)
    threads.append(thread4)

    for t in threads:
        t.join()

    print('graph enumerate finished, with ' + str(len(candidates)) + ' candidates')

    #### remove a corner ###
    #for corner_i in range(corners.shape[0]):
    #    candidates.append(remove_a_corner(corner_i, corners, edges))

    #### remove a two degree corner and connect neighbors ###
    #for corner_i in range(corners.shape[0]):
    #    if check_corner_colinear(corner_i, corners, edges):
    #        candidates.append(remove_colinear_corner(corner_i, corners, edges))

    #### remove an edge ###
    #for edge_i in range(edges.shape[0]):
    #    candidates.append(remove_an_edge(edge_i, corners, edges))

    #### add two unconnected corners ###
    ##for corner_i in range(corners.shape[0]):
    ##    for corner_j in range(corner_i+1, corners.shape[0]):
    #        if has_edge(corner_i, corner_j, edges):
    #            continue
    #        candidates.append(add_an_edge(corner_i, corner_j, corners, edges))

    return candidates


def get_region_mask(corners, edges):
    conv_mask = render(corners=corners, edges=edges, render_pad=0, edge_linewidth=1)[0]
    conv_mask = 1 - conv_mask
    conv_mask = conv_mask.astype(np.uint8)
    labels, region_mask = cv2.connectedComponents(conv_mask, connectivity=4)
    return region_mask


def patch_samples(edge_num, batch_size):
    num = edge_num // batch_size
    patchs = []
    for i in range(num):
        patchs.append([i*batch_size+j for j in range(batch_size)])

    if edge_num % batch_size != 0:
        patchs.append([j for j in range(batch_size*num, edge_num)])

    return patchs



def dump_config_file(corners_, edges_, score, config, base_path, base_name):
    corners_state = config['corners_state']
    edges_state = config['edges_state']
    regions_state = config['regions_state']
    regions_id = config['regions_id']

    false_corner_id = [ti for ti in range(len(corners_state)) if corners_state[ti] < 0]
    false_edge_id = [ti for ti in range(len(edges_state)) if edges_state[ti] < 0]
    false_region_id = [(regions_id[ti], regions_state[ti]) for ti in range(len(regions_state)) if regions_state[ti] < 0]

    with open(os.path.join(base_path, base_name+'_info.txt'), 'w') as f:
        f.write(str(round(score,2))+'\n'+
                str(len(false_corner_id))+'\n'+
                str(len(false_edge_id))+'\n'+
                str(len(false_region_id))+'\n'+
                str(round(config['corner'], 2))+'\n'+
                str(round(config['edge'], 2))+'\n'+
                str(round(config['region'], 2))+'\n'
                )
    temp_mask = np.zeros((256,256))
    for corner_i in false_corner_id:
        temp_mask = cv2.circle(temp_mask, (int(corners_[corner_i,1]), int(corners_[corner_i,0])), 3, 1, -1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1,1)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name+'_corner.png'), dpi=256)
    #im = Image.fromarray(temp_mask)
    #im.save(os.path.join(base_path, base_name+'_corner.png'))

    temp_mask = np.zeros((256,256))
    for edge_i in false_edge_id:
        corner1 = np.round(corners_[edges_[edge_i,0]])
        corner2 = np.round(corners_[edges_[edge_i,1]])
        temp_mask = cv2.line(temp_mask, (int(corner1[1]), int(corner1[0])),
                             (int(corner2[1]), int(corner2[0])), 1, thickness=1)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name+'_edge.png'), dpi=256)
    #im = Image.fromarray(temp_mask)
    #im.save(os.path.join(base_path, base_name+'_edge.png'))

    temp_mask = np.zeros((256,256))
    region_mask = get_region_mask(corners_, edges_)
    for region_i, region_score in false_region_id:
        temp_mask += -region_score * (region_mask == region_i)
    ax.imshow(temp_mask, aspect='auto')
    fig.savefig(os.path.join(base_path, base_name+'_region.png'), dpi=256)
    plt.close()

def visualize_config(corners, edges, config, score, start_i, I,J):
    mask = render(corners, edges)
    corners_state = config['corners_state']
    edges_state = config['edges_state']
    regions_state = config['regions_state']
    regions_id = config['regions_id']
    print('corners: '+str(config['corner']))
    print('edges: '+str(config['edge']))
    print('regions: '+str(config['region']))

    false_corner_id = [ti for ti in range(len(corners_state)) if corners_state[ti] < 0]
    false_edge_id = [ti for ti in range(len(edges_state)) if edges_state[ti] < 0]
    false_region_id = [(regions_id[ti], regions_state[ti]) for ti in range(len(regions_state)) if regions_state[ti] < 0]

    plt.subplot(I,J,start_i)
    plt.title('graph score: ' + str(round(score, 2)))
    plt.imshow(mask[0] + mask[1])
    plt.subplot(I,J,start_i+1)
    plt.title('false corner: '+str(len(false_corner_id)))
    temp_mask = np.zeros((256,256))
    for corner_i in false_corner_id:
        temp_mask = cv2.circle(temp_mask, (int(corners[corner_i,1]), int(corners[corner_i,0])), 3, -corners_state[corner_i], -1)
    plt.imshow(temp_mask)
    plt.subplot(I,J,start_i+2)
    plt.title('false edge: ' + str(len(false_edge_id)))
    temp_mask = np.zeros((256,256))
    for edge_i in false_edge_id:
        corner1 = np.round(corners[edges[edge_i,0]])
        corner2 = np.round(corners[edges[edge_i,1]])
        temp_mask = cv2.line(temp_mask, (int(corner1[1]), int(corner1[0])),
                             (int(corner2[1]), int(corner2[0])), -edges_state[edge_i], thickness=2)
    plt.imshow(temp_mask)
    plt.subplot(I,J,start_i+3)
    plt.title('false region: ' + str(len(false_region_id)))
    temp_mask = np.zeros((256,256))
    region_mask = get_region_mask(corners, edges)
    for region_i, region_score in false_region_id:
        temp_mask += -region_score * (region_mask == region_i)
    plt.imshow(temp_mask)
    return start_i+4
