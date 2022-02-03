import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path


class Network:

    """Class for reading and holding the network structure and demand parameters"""

    def __init__(self, city):
        """Load and initialize data for a city"""

        assert isinstance(city, str)
        self.city = city

        self.raw_edges = pd.read_csv("Locations/" + city + "/edges.csv")
        # limiting the capacity, if required
        # self.raw_edges.loc[self.raw_edges.capacity > 5000, 'capacity'] = 5000

        self.raw_vertices = pd.read_csv("Locations/" + city + "/vertices.csv")

        self.NumNodes = self.raw_vertices.shape[0]  # number of nodes
        self.NumEdges = self.raw_edges.shape[0]  # number of edges
        self.edge_to_nodes = list(zip(self.raw_edges.edge_tail, self.raw_edges.edge_head))

        self.next_from_edge = self._next_from_edge()
        self.prev_from_edge = self._prev_from_edge()
        self.prev_from_node = self._prev_from_node()
        self.next_from_node = self._next_from_node()

        self.capacity = self.raw_edges["capacity"].tolist()

        self.travel_time = self._estimate_tt_params()

        self.edge_latency = self.latency_list()

        self.tt_adj = self._create_tt_adj()
        self.shortest_path_predecessor = None

    def _next_from_edge(self):
        n = {}
        for i in range(self.NumEdges):
            head = self.raw_edges.iloc[i]['edge_head']
            n[i] = self.raw_edges.index[self.raw_edges['edge_tail'] == head].tolist()
        return n

    def _prev_from_edge(self):
        p = {}
        for i in range(self.NumEdges):
            tail = self.raw_edges.iloc[i]['edge_tail']
            p[i] = self.raw_edges.index[self.raw_edges['edge_head'] == tail].tolist()
        return p

    def _prev_from_node(self):
        elist = {}
        for i in range(self.NumNodes):
            elist[i] = self.raw_edges.index[self.raw_edges['edge_head'] == i].tolist()
        return elist

    def _next_from_node(self):
        elist = {}
        for i in range(self.NumNodes):
            elist[i] = self.raw_edges.index[self.raw_edges['edge_tail'] == i].tolist()
        return elist

    def _estimate_tt_params(self):
        # There was a +1 for the free flow time to ensure we don't divide by zero.
        self.raw_edges['free_flow_time'] = (self.raw_edges['length'] / self.raw_edges['speed'])
        self.raw_edges['time_sensitivity'] = 2 * self.raw_edges['free_flow_time'] / self.raw_edges['capacity']
        tt_params = list(zip(self.raw_edges.free_flow_time, self.raw_edges.time_sensitivity))
        return tt_params

    def _create_tt_adj(self):
        adj = np.zeros((self.NumNodes, self.NumNodes))
        for ind in range(self.NumEdges):
            adj[self.edge_to_nodes[ind][0], self.edge_to_nodes[ind][1]] = self.raw_edges.iloc[ind]['free_flow_time']
        return adj

    def next(self, node=None, edge=None):
        """ Returns next set of edges from a node or an edge"""
        if node is not None:
            return self.next_from_node[node]
        elif edge is not None:
            return self.next_from_edge[edge]

    def prev(self, node=None, edge=None):
        """ Returns previous set of edges from a node or an edge"""
        if node is not None:
            return self.prev_from_node[node]
        elif edge is not None:
            return self.prev_from_edge[edge]

    # TODO: Speed up this function
    def cost_weighted_tt_with_tolls(self, vot, tolls):
        adj_toll = np.zeros((self.NumNodes, self.NumNodes))
        for ind in range(self.NumEdges):
            i, j = self.edge_to_nodes[ind]
            adj_toll[i, j] = tolls[ind]
        return vot * self.tt_adj + adj_toll

    def latency_list(self):
        return [c0 for c0, c1 in self.travel_time]

    def latency_array(self):
        return np.array([c0 for c0, c1 in self.travel_time])

    def capacity_list(self):
        return self.capacity

    def capacity_array(self):
        return np.array(self.capacity)

    def compute_shortest_path(self, tolls):
        adj_toll = np.zeros((self.NumNodes, self.NumNodes))
        for ind in range(self.NumEdges):
            i, j = self.edge_to_nodes[ind]
            adj_toll[i, j] = tolls[ind]
        _, self.shortest_path_predecessor = shortest_path(self.tt_adj + adj_toll,
                                                          directed=True, return_predecessors=True)
        return None
