import numpy as np
import matplotlib.pyplot as plt
import cv2


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


def IOU(maskA, maskB):
    return np.logical_and(maskA, maskB).sum() / np.logical_or(maskA, maskB).sum()


def render(corners, edges, render_pad=0, edge_linewidth=2, corner_size=3):
    mask = np.ones((2, 256, 256)) * render_pad
    for edge_i in range(edges.shape[0]):
        a = edges[edge_i, 0]
        b = edges[edge_i, 1]
        mask[0] = cv2.line(mask[0], (int(corners[a,1]), int(corners[a,0])),
                           (int(corners[b,1]), int(corners[b,0])), 1.0, thickness=edge_linewidth)
    for corner_i in range(corners.shape[0]):
        mask[1] = cv2.circle(mask[1], (int(corners[corner_i,1]), int(corners[corner_i,0])), corner_size, 1.0, -1)

    return mask


def patch_samples(edge_num, batch_size):
    num = edge_num // batch_size
    patchs = []
    for i in range(num):
        patchs.append([i*batch_size+j for j in range(batch_size)])

    if edge_num % batch_size != 0:
        patchs.append([j for j in range(batch_size*num, edge_num)])

    return patchs


def visualization(candidate):
    corners = candidate.graph.getCornersArray()
    edges = candidate.graph.getEdgesArray()
    mask = render(corners, edges)
    mask = np.transpose(np.concatenate((mask, np.zeros((1,256,256))), 0), (1,2,0))
    plt.imshow(mask)
    plt.show()


def check_intersection(edge1, edge2):
    corner11 = edge1.x[0].x
    corner12 = edge1.x[1].x
    corner21 = edge2.x[0].x
    corner22 = edge2.x[1].x

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
