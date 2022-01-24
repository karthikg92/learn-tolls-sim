import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse.csgraph import shortest_path


class Optimizer:
    """Backend solvers for optimal paths"""

    def __init__(self, network, users):
        self.network = network
        self.users = users

        self.NumUsers = len(self.users.data)
        self.NumEdges = self.network.NumEdges

        # x has dims (N_edge X N_users)
        # f has dims (N_edge X 1)

        self.x_opt, self.f_opt = self._optimal_flow()

    def _shortest_path(self, predecessor, u):
        # extracting OD pair
        s = self.users.data[u]['orig']
        t = self.users.data[u]['dest']

        # extract path from predecessor matrix
        path = [(predecessor[s, t], t)]
        while path[0][0] != s:
            current = path[0][0]
            path.insert(0, (predecessor[s, current], current))

        return path

    def latency(self, e):
        c0, c1 = self.network.travel_time[e]
        return c0

    @staticmethod
    def x_to_f(x):
        f = np.sum(x, axis=1)
        return f

    def _optimal_flow(self, return_duals=False):

        """ Gurobi optimization model """

        # print(self.network.NumNodes, self.network.NumEdges, self.NumUsers)

        # Model initialization
        m = gp.Model('VoT')
        # m.params.NonConvex = 2
        m.setParam('OutputFlag', 0)

        # decision variable
        x_eu = m.addVars(self.NumEdges, self.NumUsers, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

        # introducing edge flows
        x_e = m.addVars(self.NumEdges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
        m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(self.NumEdges))

        # demand from origin constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['orig'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['orig'])]) == 0
            for u in range(self.NumUsers))

        # demand at destination constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['dest'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['dest'])]) == 0
            for u in range(self.NumUsers))

        # flow conservation
        for u in range(self.NumUsers):

            exclude_od_nodes = [n for n in range(self.network.NumNodes)]
            exclude_od_nodes.remove(self.users.data[u]['orig'])
            exclude_od_nodes.remove(self.users.data[u]['dest'])

            # assert len(exclude_od_nodes) == (self.network.NumNodes - 2)

            m.addConstrs(
                sum(x_eu[g, u] for g in self.network.prev(node=n)) ==
                sum(x_eu[g, u] for g in self.network.next(node=n))
                for n in exclude_od_nodes)

        # capacity constraints (testing the for loop so that we can extract duals later)
        for e in range(self.NumEdges):
            m.addConstr(x_e[e] <= self.network.capacity[e], name='capacity'+str(e))
#        m.addConstrs(x_e[e] <= self.network.capacity[e] for e in range(self.NumEdges))

        # objective function
        so_obj = sum([self.users.data[u]['vot'] *
                      x_eu[e, u] *
                      self.latency(e)
                      for e in range(self.NumEdges) for u in range(self.NumUsers)])
        m.setObjective(so_obj, GRB.MINIMIZE)

        # run the optimization
        m.optimize()

        # If infeasible, terminate program
        assert m.status != GRB.INFEASIBLE

        # extract the solution flows
        opt_x = np.zeros((self.NumEdges, self.NumUsers))
        x_dict = m.getAttr('x', x_eu)
        for e in range(self.NumEdges):
            for u in range(self.NumUsers):
                opt_x[e, u] = x_dict[e, u]

        opt_f = self.x_to_f(opt_x)

        if return_duals is True:
            # extract the dual variables corresponding to edge capacities
            duals = np.zeros(self.NumEdges)
            for e in range(self.NumEdges):
                constraint = m.getConstrByName('capacity'+str(e))
                # print(constraint.getAttr(GRB.Attr.Pi))
                # print(type(constraint.getAttr(GRB.Attr.Pi)))
                duals[e] = constraint.getAttr(GRB.Attr.Pi)

        if return_duals is False:
            return opt_x, opt_f
        if return_duals is True:
            return duals

    def toll_flow(self, tolls, is_feasible=False):

        x = np.zeros((self.NumEdges, self.NumUsers))
        edge_map_dict = dict(zip(self.network.edge_to_nodes, range(self.NumEdges)))

        # TODO: Potentially parallelize this for loop

        for u in range(self.NumUsers):
            vol = self.users.data[u]['vol']
            s = self.users.data[u]['orig']
            t = self.users.data[u]['dest']

            adj = self.network.cost_weighted_tt_with_tolls(self.users.data[u]['vot'], tolls)
            dist, predecessor = shortest_path(adj, directed=True, return_predecessors=True, indices=s)

            if is_feasible is False:
                while 1:
                    edge_index = edge_map_dict[(predecessor[t], t)]
                    x[edge_index, u] = vol
                    t = predecessor[t]
                    if t == s:
                        break
            if is_feasible is True:
                # TODO: This is not unique and depends on the order in which we deal with the users
                """ 
                Need to figure out greedy allocation with capacity constraints.
                This case involves having an outside option! When users do not choose the road network, 
                they will choose an o-d specific 'outside option' which has sufficient capacity
                """

        f = self.x_to_f(x)

        return x, f

    def benchmark_flow(self, tolls):
        """
        Compute the flow in response to the benchmark tolls.
        This computation needs to be done only once and involves the same routing for all uses at every time step
        ----- INCORECT!
        """

        x = np.zeros((self.NumEdges, self.NumUsers))
        edge_map_dict = dict(zip(self.network.edge_to_nodes, range(self.NumEdges)))

        for u in range(self.NumUsers):
            vol = self.users.data[u]['vol']
            s = self.users.data[u]['orig']
            t = self.users.data[u]['dest']

            # TODO: THIS NEEDS TO USE A SHORTEST PATH THAT NEEDS TO BE COMPUTED EVERYTIME
            # need to compute or use a predecessor matrix
            if self.network.shortest_path_predecessor is None:
                self.network.compute_shortest_path(tolls)
                predecessor = self.network.shortest_path_predecessor
            else:
                predecessor = self.network.shortest_path_predecessor

            while 1:
                edge_index = edge_map_dict[(predecessor[s, t], t)]
                x[edge_index, u] = vol
                t = predecessor[s, t]
                if t == s:
                    break

        f = self.x_to_f(x)

        return x, f

    def compute_benchmark_toll(self):
        num_sim = 10000

        # Model initialization
        m = gp.Model('VoT')
        # m.params.NonConvex = 2
        m.setParam('OutputFlag', 0)

        # decision variable
        x_eu = m.addVars(self.NumEdges, self.NumUsers, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

        # introducing edge flows
        x_e = m.addVars(self.NumEdges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
        m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(self.NumEdges))

        # demand from origin constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['orig'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['orig'])]) == 0
            for u in range(self.NumUsers))

        # demand at destination constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['dest'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['dest'])]) == 0
            for u in range(self.NumUsers))

        # flow conservation
        for u in range(self.NumUsers):
            exclude_od_nodes = [n for n in range(self.network.NumNodes)]
            exclude_od_nodes.remove(self.users.data[u]['orig'])
            exclude_od_nodes.remove(self.users.data[u]['dest'])

            # assert len(exclude_od_nodes) == (self.network.NumNodes - 2)

            m.addConstrs(
                sum(x_eu[g, u] for g in self.network.prev(node=n)) ==
                sum(x_eu[g, u] for g in self.network.next(node=n))
                for n in exclude_od_nodes)

        # capacity constraints (testing the for loop so that we can extract duals later)
        for e in range(self.NumEdges):
            m.addConstr(x_e[e] <= self.network.capacity[e], name='capacity' + str(e))
        #        m.addConstrs(x_e[e] <= self.network.capacity[e] for e in range(self.NumEdges))

        # objective function
        # Need to include all realizations of VOTs

        sum_vot = sum([self.users.vot_realization() for _ in range(num_sim)])

        #print(sum_vot)
        # for u in range(self.NumUsers):
        #     assert sum_vot[u] == 1

        obj = 1 / num_sim * sum([sum_vot[u] * x_eu[e, u] * self.latency(e)
                                 for e in range(self.NumEdges)
                                 for u in range(self.NumUsers)])

        m.setObjective(obj, GRB.MINIMIZE)

        # run the optimization
        print('[optimizer] Begin gurobi optimization')
        m.optimize()
        print('[optimizer] Complete gurobi optimization')

        # extracting the duals
        duals = np.zeros(self.NumEdges)
        for e in range(self.NumEdges):
            constraint = m.getConstrByName('capacity' + str(e))
            # print(constraint.getAttr(GRB.Attr.Pi))
            # print(type(constraint.getAttr(GRB.Attr.Pi)))
            duals[e] = constraint.getAttr(GRB.Attr.Pi)

        tolls = -1 * duals

        # print(tolls)
        return tolls

    def optimal_toll_response(self, tolls):

        # Model initialization
        m = gp.Model('VoT')
        # m.params.NonConvex = 2
        m.setParam('OutputFlag', 0)

        # decision variable
        x_eu = m.addVars(self.NumEdges, self.NumUsers, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

        # introducing edge flows
        x_e = m.addVars(self.NumEdges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
        m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(self.NumEdges))

        # demand from origin constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['orig'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['orig'])]) == 0
            for u in range(self.NumUsers))

        # demand at destination constraint
        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.prev(node=self.users.data[u]['dest'])]) ==
            self.users.data[u]['vol']
            for u in range(self.NumUsers))

        m.addConstrs(
            sum([x_eu[e, u] for e in self.network.next(node=self.users.data[u]['dest'])]) == 0
            for u in range(self.NumUsers))

        # flow conservation
        for u in range(self.NumUsers):
            exclude_od_nodes = [n for n in range(self.network.NumNodes)]
            exclude_od_nodes.remove(self.users.data[u]['orig'])
            exclude_od_nodes.remove(self.users.data[u]['dest'])

            # assert len(exclude_od_nodes) == (self.network.NumNodes - 2)

            m.addConstrs(
                sum(x_eu[g, u] for g in self.network.prev(node=n)) ==
                sum(x_eu[g, u] for g in self.network.next(node=n))
                for n in exclude_od_nodes)

        # no capacity constraints
        # for e in range(self.NumEdges):
        #     m.addConstr(x_e[e] <= self.network.capacity[e], name='capacity' + str(e))

        # check VOT values
        # for u in range(self.NumUsers):
        #     assert self.users.data[u]['vot'] == 1

        # objective function
        term1 = sum([x_e[e] * self.latency(e) for e in range(self.NumEdges)])
        term2 = sum([1 / self.users.data[u]['vot'] * x_eu[e, u] * tolls[e]
                     for e in range(self.NumEdges) for u in range(self.NumUsers)])
        toll_obj = term1 + term2
        m.setObjective(toll_obj, GRB.MINIMIZE)

        # run the optimization
        m.optimize()

        # If infeasible, terminate program
        assert m.status != GRB.INFEASIBLE

#        print("Objective value = ", m.getObjective().getValue()/1e9)

        # extract the solution flows
        x = np.zeros((self.NumEdges, self.NumUsers))
        x_dict = m.getAttr('x', x_eu)
        for e in range(self.NumEdges):
            for u in range(self.NumUsers):
                x[e, u] = x_dict[e, u]

        f = self.x_to_f(x)

        return x, f


def compute_stochastic_program_toll(network, users, num_sim=1000):

    num_edges = network.NumEdges
    num_users = users.num_users

    # Model initialization
    m = gp.Model('VoT')
    # m.params.NonConvex = 2
    m.setParam('OutputFlag', 0)

    # decision variable
    x_eu = m.addVars(num_edges, num_users, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

    # introducing edge flows
    x_e = m.addVars(num_edges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
    m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(num_edges))

    # demand from origin constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['orig'])]) == users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['orig'])]) == 0
        for u in range(num_users))

    # demand at destination constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['dest'])]) == users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['dest'])]) == 0
        for u in range(num_users))

    # flow conservation
    for u in range(num_users):
        exclude_od_nodes = [n for n in range(network.NumNodes)]
        exclude_od_nodes.remove(users.data[u]['orig'])
        exclude_od_nodes.remove(users.data[u]['dest'])

        # assert len(exclude_od_nodes) == (self.network.NumNodes - 2)

        m.addConstrs(
            sum(x_eu[g, u] for g in network.prev(node=n)) ==
            sum(x_eu[g, u] for g in network.next(node=n))
            for n in exclude_od_nodes)

    # capacity constraints (testing the for loop so that we can extract duals later)
    for e in range(num_edges):
        m.addConstr(x_e[e] <= network.capacity[e], name='capacity' + str(e))

    # objective function
    # Need to include all realizations of VOTs

    sum_vot = sum([users.vot_realization() for _ in range(num_sim)])

    obj = 1 / num_sim * sum([sum_vot[u] * x_eu[e, u] * network.edge_latency[e]
                             for e in range(num_edges)
                             for u in range(num_users)])

    m.setObjective(obj, GRB.MINIMIZE)

    # run the optimization
    m.optimize()

    # extracting the duals
    duals = np.zeros(num_edges)
    for e in range(num_edges):
        constraint = m.getConstrByName('capacity' + str(e))
        duals[e] = constraint.getAttr(GRB.Attr.Pi)

    tolls = -1 * duals
    return tolls


def compute_same_vot_toll(network, users):

    num_edges = network.NumEdges
    num_users = users.num_users

    # Model initialization
    m = gp.Model('VoT')
    # m.params.NonConvex = 2
    m.setParam('OutputFlag', 0)

    # decision variable
    x_eu = m.addVars(num_edges, num_users, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

    # introducing edge flows
    x_e = m.addVars(num_edges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
    m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(num_edges))

    # demand from origin constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['orig'])]) == users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['orig'])]) == 0
        for u in range(num_users))

    # demand at destination constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['dest'])]) == users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['dest'])]) == 0
        for u in range(num_users))

    # flow conservation
    for u in range(num_users):
        exclude_od_nodes = [n for n in range(network.NumNodes)]
        exclude_od_nodes.remove(users.data[u]['orig'])
        exclude_od_nodes.remove(users.data[u]['dest'])

        # assert len(exclude_od_nodes) == (self.network.NumNodes - 2)

        m.addConstrs(
            sum(x_eu[g, u] for g in network.prev(node=n)) ==
            sum(x_eu[g, u] for g in network.next(node=n))
            for n in exclude_od_nodes)

    # capacity constraints (testing the for loop so that we can extract duals later)
    for e in range(num_edges):
        m.addConstr(x_e[e] <= network.capacity[e], name='capacity' + str(e))

    # objective function
    obj = sum([x_e[e] * network.edge_latency[e] for e in range(num_edges)])

    m.setObjective(obj, GRB.MINIMIZE)

    # run the optimization
    m.optimize()

    # extracting the duals
    duals = np.zeros(num_edges)
    for e in range(num_edges):
        constraint = m.getConstrByName('capacity' + str(e))
        duals[e] = constraint.getAttr(GRB.Attr.Pi)

    tolls = -1 * duals
    return tolls


def user_equilibrium_with_tolls(network, users, tolls):

    num_edges = network.NumEdges
    num_users = users.num_users

    # Model initialization
    m = gp.Model('VoT')
    # m.params.NonConvex = 2
    m.setParam('OutputFlag', 0)

    # decision variable
    x_eu = m.addVars(num_edges, num_users, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_eu")

    # introducing edge flows
    x_e = m.addVars(num_edges, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x_e")
    m.addConstrs(x_eu.sum(e, '*') == x_e[e] for e in range(num_edges))

    # demand from origin constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['orig'])]) == users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['orig'])]) == 0
        for u in range(num_users))

    # demand at destination constraint
    m.addConstrs(
        sum([x_eu[e, u] for e in network.prev(node=users.data[u]['dest'])]) ==
        users.data[u]['vol']
        for u in range(num_users))

    m.addConstrs(
        sum([x_eu[e, u] for e in network.next(node=users.data[u]['dest'])]) == 0
        for u in range(num_users))

    # flow conservation
    for u in range(num_users):
        exclude_od_nodes = [n for n in range(network.NumNodes)]
        exclude_od_nodes.remove(users.data[u]['orig'])
        exclude_od_nodes.remove(users.data[u]['dest'])

        m.addConstrs(
            sum(x_eu[g, u] for g in network.prev(node=n)) ==
            sum(x_eu[g, u] for g in network.next(node=n))
            for n in exclude_od_nodes)

    # no capacity constraints
    # for e in range(self.NumEdges):
    #     m.addConstr(x_e[e] <= self.network.capacity[e], name='capacity' + str(e))

    # objective function
    term1 = sum([x_e[e] * network.edge_latency[e] for e in range(num_edges)])
    term2 = sum([1 / users.data[u]['vot'] * x_eu[e, u] * tolls[e]
                 for e in range(num_edges) for u in range(num_users)])
    toll_obj = term1 + term2
    m.setObjective(toll_obj, GRB.MINIMIZE)

    # run the optimization
    m.optimize()

    # If infeasible, terminate program
    assert m.status != GRB.INFEASIBLE

    # extract the solution flows
    x = np.zeros((num_edges, num_users))
    x_dict = m.getAttr('x', x_eu)
    for e in range(num_edges):
        for u in range(num_users):
            x[e, u] = x_dict[e, u]

    f = np.sum(x, axis=1)

    return x, f
