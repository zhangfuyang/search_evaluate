import numpy as np
from new_utils import sort_graph, check_intersection

SAFE_NUM = 3


class Element:
    def __init__(self, x, safe_count=0):
        assert type(x) is tuple
        self.x = x
        self.__score = None
        self.safe_count = safe_count

    def store_score(self, score):
        self.__score = score

    def get_score(self):
        return self.__score

    def equal(self, ele):
        if type(self.x[0]) != type(ele.x[0]):
            return False
        if type(self.x[0]) == int:
            # corner
            return True if self.x[0] == ele.x[0] and self.x[1] == ele.x[1] else False
        if type(self.x[0]) == Element:
            # edge
            if self.x[0].equal(ele.x[0]) and self.x[1].equal(ele.x[1]):
                return True
            if self.x[1].equal(ele.x[0]) and self.x[0].equal(ele.x[1]):
                return True
            return False
        raise BaseException('no implement type')


class Graph:
    def __init__(self, corners, edges):
        corners, edges = sort_graph(corners, edges)

        self.__corners = []
        for corner_i in range(corners.shape[0]):
            self.__corners.append(
                Element(
                    tuple(
                        (int(corners[corner_i,0]),int(corners[corner_i,1]))
                    )
                )
            )
        self.__edges = []
        for edge_i in range(edges.shape[0]):
            self.__edges.append(Element((self.__corners[edges[edge_i,0]], self.__corners[edges[edge_i,1]])))
        self.__regions = []
        self.__regions.append(Element(())) # we use entire region here

    @classmethod
    def initialFromTuple(cls, corners, edges):
        edge_index = []
        for item in edges:
            a = corners.index(item[0])
            b = corners.index(item[1])
            edge_index.append((a,b))
        edge_index = np.array(edge_index)
        corners = np.array(corners)
        return cls(corners, edge_index)

    def store_score(self, corner_score=None, edge_score=None, region_score=None):
        '''
        :param corner_score: np array size: len(corners)
        :param edge_score:  np array size: len(edges)
        :param region_score: np.array size: len(regions)
        :return:
        '''
        if corner_score is not None:
            for idx, element in enumerate(self.__corners):
                element.store_score(corner_score[idx])
        if edge_score is not None:
            for idx, element in enumerate(self.__edges):
                element.store_score(edge_score[idx])
        if region_score is not None:
            for idx, element in enumerate(self.__regions):
                element.store_score(region_score[idx])
        return

    def getCornersArray(self):
        c = []
        for ele in self.__corners:
            c.append(ele.x)
        return np.array(c)

    def getEdgesArray(self):
        c = []
        for ele in self.__edges:
            corner1 = ele.x[0]
            corner2 = ele.x[1]
            idx1 = self.__corners.index(corner1)
            idx2 = self.__corners.index(corner2)
            c.append([idx1, idx2])
        return np.array(c)

    def getCorners(self):
        return self.__corners

    def getRegions(self):
        return self.__regions

    def getEdges(self):
        return self.__edges

    def graph_score(self):
        corner_score = 0
        for ele in self.__corners:
            corner_score += ele.get_score()
        edge_score = 0
        for ele in self.__edges:
            edge_score += ele.get_score()
        region_score = 0
        for ele in self.__regions:
            region_score += ele.get_score()
        return corner_score + 2*edge_score + 60*region_score

    def corner_score(self):
        corner_score = 0
        for ele in self.__corners:
            corner_score += ele.get_score()
        return corner_score

    def edge_score(self):
        edge_score = 0
        for ele in self.__edges:
            edge_score += ele.get_score()
        return edge_score

    def region_score(self):
        region_score = 0
        for ele in self.__regions:
            region_score += ele.get_score()
        return region_score

    def remove(self, ele):
        '''
        :param ele: remove eles as well as some other related elements
        :return: set() of removed elements
        '''
        # corner
        removed = set()
        if ele in self.__corners:
            self.__corners.remove(ele)
            removed.add(ele)
            # remove edge that has the corner
            for idx in reversed(range(len(self.__edges))):
                edge_ele = self.__edges[idx]
                if ele in edge_ele.x:
                    removed = removed.union(self.remove(edge_ele))
        # edge
        elif ele in self.__edges:
            self.__edges.remove(ele)
            removed.add(ele)
            corner1 = ele.x[0]
            corner2 = ele.x[1]
            if corner1.safe_count == 0:
                # can be delete
                _count = 0
                for edge_ele in self.__edges:
                    if corner1 in edge_ele.x:
                        _count += 1
                if _count == 0:
                    removed = removed.union(self.remove(corner1))
            if corner2.safe_count == 0:
                # can be delete
                _count = 0
                for edge_ele in self.__edges:
                    if corner2 in edge_ele.x:
                        _count += 1
                if _count == 0:
                    removed = removed.union(self.remove(corner2))
        return removed

    def has_edge(self, ele1, ele2):
        """
        :param ele1: corner1
        :param ele2: corner2
        :return: edge or none
        """
        for edge_ele in self.__edges:
            if ele1 in edge_ele.x and ele2 in edge_ele.x:
                return edge_ele
        return None

    def add_edge(self, ele1, ele2):
        temp = self.has_edge(ele1, ele2)
        if temp is not None:
            temp.safe_count = SAFE_NUM
            return temp
        new_ele = Element((ele1, ele2), safe_count=SAFE_NUM)
        self.__edges.append(new_ele)
        return new_ele

    def add_corner(self, ele):
        for corner in self.__corners:
            if corner.x == ele.x:
                corner.safe_count = SAFE_NUM
                return corner
        ele.safe_count = SAFE_NUM
        self.__corners.append(ele)
        return ele

    def checkColinearCorner(self, ele):
        if self.getCornerDegree(ele) != 2:
            return False
        edge_in = []
        for edge_ele in self.__edges:
            if ele in edge_ele.x:
                edge_in.append(edge_ele)
                if len(edge_in) == 2:
                    break
        two_neighbor = {edge_in[0].x[0], edge_in[0].x[1], edge_in[1].x[0], edge_in[1].x[1]}
        two_neighbor.remove(ele)
        two_neighbor = tuple(two_neighbor)
        if self.has_edge(two_neighbor[0], two_neighbor[1]) is not None:
            return False

        line1 = np.array(ele.x) - np.array(two_neighbor[0].x)
        line2 = np.array(two_neighbor[1].x) - np.array(ele.x)
        cos = np.dot(line1, line2) / (np.linalg.norm(line1)*np.linalg.norm(line2))
        if np.arccos(cos) < np.pi / 9: # 20 degree
            return True
        return False

    def checkIntersectionEdge(self, ele):
        for edge_ele in self.__edges:
            if check_intersection(edge_ele, ele):
                return True
        return False

    def getCornerDegree(self, ele):
        degree = 0
        for edge_ele in self.__edges:
            if ele in edge_ele.x:
                degree += 1
        return degree

    def getEdgeConnected(self, ele):
        out_ = set()
        if type(ele.x[0]) == int:
            # corner
            for edge_ele in self.__edges:
                if ele in edge_ele.x:
                    out_.add(edge_ele)
            return out_
        if type(ele.x[0]) == Element:
            # Edge
            out_ = out_.union(self.getEdgeConnected(ele.x[0]))
            out_ = out_.union(self.getEdgeConnected(ele.x[1]))
            if ele in out_:
                out_.remove(ele)
            return out_

    def getRealElement(self, ele):
        #edge
        if type(ele.x[0]) == Element:
            for e in self.__edges:
                if e.x[0].x == ele.x[0].x and e.x[1].x == ele.x[1].x:
                    return e
            raise BaseException("no same edge exists.")
        #corner
        elif type(ele.x[0]) == int:
            for c in self.__corners:
                if c.x == ele.x:
                    return c
            raise BaseException("no same corner exists.")

    def copy(self):
        corners = self.getCornersArray()
        edges = self.getEdgesArray()
        new_graph = Graph(corners, edges)
        for idx, ele in enumerate(self.__corners):
            new_graph.__corners[idx].store_score(self.__corners[idx].get_score())
        for idx, ele in enumerate(self.__edges):
            new_graph.__edges[idx].store_score(self.__edges[idx].get_score())
        for idx, ele in enumerate(self.__regions):
            new_graph.__regions[idx].store_score(self.__regions[idx].get_score)
        return new_graph

    def update_safe_count(self):
        for ele in self.__corners:
            if ele.safe_count > 0:
                ele.safe_count -= 1
        for ele in self.__edges:
            if ele.safe_count > 0:
                ele.safe_count -= 1

    def isNeighbor(self, element1, element2):
        '''
        :param element1:
        :param element2:
        :return: True / False
        '''
        if element1 == element2:
            return False
        if type(element1.x[0]) != type(element2.x[0]):
            # corner and edge
            return False
        if type(element1.x[0]) == int:
            # both are corner type
            for edge_ele in self.__edges:
                if edge_ele.x[0] == element1 and edge_ele.x[1] == element2:
                    return True
                if edge_ele.x[0] == element2 and edge_ele.x[1] == element1:
                    return True
            return False
        if type(element1.x[0]) == Element:
            # both are edge type
            if len({element1.x[0], element1.x[1], element2.x[0], element2.x[1]}) < 4:
                return True
            return False

    def equal(self, graph):
        if len(self.__corners) != len(graph.__corners) or \
                len(self.__edges) != len(graph.__edges):
            return False
        for corner_i in range(len(self.__corners)):
            if self.__corners[corner_i].equal(graph.__corners[corner_i]) is False:
                return False
        for edge_i in range(len(self.__edges)):
            if self.__edges[edge_i].equal(graph.__edges[edge_i]) is False:
                return False

        return True


class Candidate:
    def __init__(self, graph, name, corner_existed_before, edge_existed_before):
        '''
        :param graph: Class graph
        :param name: string, data name
        :param corner_existed_before: dict {(x_i,y_i):c_1 ...} indicates counts for corresponding corners, after one search,
                                     counts -= 1, if count == 0, remove from the set.
        :param edge_existed_before: dict {((x_i1,y_i1),(x_i2,y_i2)):ci}
        '''
        self.graph = graph
        self.name = name
        self.corner_existed_before = corner_existed_before
        self.edge_existed_before = edge_existed_before

    @classmethod
    def initial(cls, graph, name):
        return cls(graph, name, {}, {})

    def update(self):
        # all the existed before elements count - 1
        for key in self.corner_existed_before.keys():
            self.corner_existed_before[key] -= 1
        for key in self.edge_existed_before.keys():
            self.edge_existed_before[key] -= 1

        # check if some need to remove from existed before set
        for key in list(self.corner_existed_before.keys()):
            if self.corner_existed_before[key] == 0:
                self.corner_existed_before.pop(key)

        for key in list(self.edge_existed_before.keys()):
            if self.edge_existed_before[key] == 0:
                self.edge_existed_before.pop(key)

        # update graph
        self.graph.update_safe_count()

    def copy(self):
        corner_existed_before = self.corner_existed_before.copy()
        edge_existed_before = self.edge_existed_before.copy()
        new_graph = self.graph.copy()
        return Candidate(new_graph, self.name, corner_existed_before, edge_existed_before)

    def removable(self, ele):
        '''
        :param x: input is element
        :return:
        '''
        assert type(ele) == Element
        # edge
        return True if ele.safe_count == 0 else False

    def addable(self, ele):
        if type(ele.x[0]) == Element:
            # edge
            corner1_loc = ele.x[0].x
            corner2_loc = ele.x[1].x
            if (corner1_loc, corner2_loc) in self.edge_existed_before.keys() or \
                    (corner2_loc, corner1_loc) in self.edge_existed_before.keys():
                return False
            return True
        else:
            # corner
            if ele.x in self.corner_existed_before.keys():
                return False
            return True

    def addCorner(self, ele):
        if ele.x in self.corner_existed_before.keys():
            raise BaseException('cannot add the corner')
        new_ele = self.graph.add_corner(ele) # possible changed
        return new_ele

    def addEdge(self, ele1, ele2):
        corner1 = ele1
        corner2 = ele2
        assert corner1 in self.graph.getCorners()
        assert corner2 in self.graph.getCorners()
        if (corner1.x, corner2.x) in self.edge_existed_before.keys() or \
                (corner2.x, corner1.x) in self.edge_existed_before.keys():
            raise BaseException('cannot add the edge')
        new_ele = self.graph.add_edge(corner1, corner2)
        return new_ele

    def removeCorner(self, ele):
        if ele.x in self.corner_existed_before.keys():
            raise BaseException('already existed.')
        self.corner_existed_before[ele.x] = SAFE_NUM

    def removeEdge(self, ele):
        corner1 = ele.x[0]
        corner2 = ele.x[1]
        loc1 = corner1.x
        loc2 = corner2.x
        if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
            loc1 = corner2.x
            loc2 = corner1.x
        if (loc1, loc2) in self.edge_existed_before.keys():
            raise BaseException('already existed.')
        self.edge_existed_before[(loc1, loc2)] = SAFE_NUM

    def generate_new_candidate_remove_a_colinear_corner(self, ele):
        # need to check if ele is a colinear corner before
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)

        # find two neighbor corners
        temp = set()
        for element in new_graph.getEdgeConnected(ele):
            # edge
            if type(element.x[0]) == Element:
                temp.add(element.x[0])
                temp.add(element.x[1])
        temp.remove(ele)
        temp = tuple(temp)
        assert len(temp) == 2

        # add edge to two neighbor corners
        # (add before remove, in case the neighbor corners will be removed by zero degree)
        # special case no need to check existed_before, instead remove if in existed_before dict
        added = new_graph.add_edge(temp[0], temp[1])
        if (temp[0].x, temp[1].x) in self.edge_existed_before.keys():
            self.edge_existed_before.pop((temp[0].x, temp[1].x))
        if (temp[1].x, temp[0].x) in self.edge_existed_before.keys():
            self.edge_existed_before.pop((temp[1].x, temp[0].x))

        # remove
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            #edge
            if type(element.x[0]) == Element:
                new_candidate.removeEdge(element)
            #corner
            elif type(element.x[0]) == int:
                new_candidate.removeCorner(element)
            else:
                raise BaseException('wrong type.')


        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges OR new edges will be recounted
        for element in new_graph.getEdges():
            for modified_ele in removed.union({added}):
                if new_graph.isNeighbor(element, modified_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_remove_a_corner(self, ele):
        # need to check if ele is removable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            #edge
            if type(element.x[0]) == Element:
                corner1 = element.x[0]
                corner2 = element.x[1]
                loc1 = corner1.x
                loc2 = corner2.x
                if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
                    loc1 = corner2.x
                    loc2 = corner1.x
                if (loc1, loc2) in self.edge_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.edge_existed_before[(loc1, loc2)] = SAFE_NUM
            #corner
            elif type(element.x[0]) == int:
                if element.x in self.corner_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.corner_existed_before[element.x] = SAFE_NUM
            else:
                raise BaseException('wrong type.')

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges will be recounted
        for element in new_graph.getEdges():
            for removed_ele in removed:
                if new_graph.isNeighbor(element, removed_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_add_an_edge(self, ele1, ele2):
        # need to check addable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele1 = new_graph.getRealElement(ele1)
        ele2 = new_graph.getRealElement(ele2)

        # add edge
        new_ele = new_candidate.addEdge(ele1, ele2)

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges thay are neighbors to the added edges will be recounted
        for element in new_graph.getEdges():
            if new_graph.isNeighbor(element, new_ele):
                element.store_score(None)

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def generate_new_candidate_remove_an_edge(self, ele):
        # need to check if ele is removable before call this method
        new_candidate = self.copy()
        new_graph = new_candidate.graph
        ele = new_graph.getRealElement(ele)
        removed = new_graph.remove(ele)

        # add removed elements into existed before
        for element in removed:
            #edge
            if type(element.x[0]) == Element:
                corner1 = element.x[0]
                corner2 = element.x[1]
                loc1 = corner1.x
                loc2 = corner2.x
                if (loc1[0] > loc2[0]) or (loc1[0] == loc2[0] and loc1[1] > loc2[1]):
                    loc1 = corner2.x
                    loc2 = corner1.x
                if (loc1, loc2) in self.edge_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.edge_existed_before[(loc1, loc2)] = SAFE_NUM
            #corner
            elif type(element.x[0]) == int:
                if element.x in self.corner_existed_before.keys():
                    raise BaseException('already existed.')
                new_candidate.corner_existed_before[element.x] = SAFE_NUM
            else:
                raise BaseException('wrong type.')

        # modify scores that need to be recounted
        # all corners are recounted
        for element in new_graph.getCorners():
            element.store_score(None)

        # edges that are neighbors to the removed edges will be recounted
        for element in new_graph.getEdges():
            for removed_ele in removed:
                if new_graph.isNeighbor(element, removed_ele):
                    element.store_score(None)
                    break

        # all regions are recounted
        for element in new_graph.getRegions():
            element.store_score(None)

        return new_candidate

    def equal(self, candidate):
        return self.graph.equal(candidate.graph)

