import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse.csgraph import shortest_path


def compute_stochastic_program_toll(network, users, num_sim=1000, constant_vot=False):

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

    if constant_vot is True:
        # print('[Debug] Ensuring that constant VOTs are used to compute stochastic program tolls')
        sum_vot = sum([users.vot_realization(fixed_vot=True) for _ in range(num_sim)])
        # print('[Debug] Expect same numbers. Actual:', sum(sum_vot), num_sim*num_users)
    else:
        sum_vot = sum([users.vot_realization() for _ in range(num_sim)])

    # obj = 1 / num_sim * sum([sum_vot[u] * x_eu[e, u] * network.edge_latency[e]
    #                          for e in range(num_edges)
    #                          for u in range(num_users)])
    #
    obj = 0
    for e in range(num_edges):
        for u in range(num_users):
            obj += sum_vot[u] * x_eu[e, u] * network.edge_latency[e]
    obj *= 1 / num_sim


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
    # for e in range(num_edges):
    #   m.addConstr(x_e[e] <= network.capacity[e], name='capacity' + str(e))

    # Additional constraint for debug purposes
    # if opt_obj is not None:
    #     m.addConstr(sum([users.data[u]['vot'] * x_eu[e, u] * network.edge_latency[e]
    #                      for e in range(num_edges) for u in range(num_users)]) <= opt_obj)

    # objective function
    # term1 = sum([x_e[e] * network.edge_latency[e] for e in range(num_edges)])
    # # TODO: VECTORIZE this term
    # term2 = sum([1 / users.data[u]['vot'] * x_eu[e, u] * tolls[e]
    #              for e in range(num_edges) for u in range(num_users)])
    # toll_obj = term1 + term2

    toll_obj = 0
    for e in range(num_edges):
        toll_obj += x_e[e] * network.edge_latency[e]
        for u in range(num_users):
            toll_obj += 1 / users.data[u]['vot'] * x_eu[e, u] * tolls[e]



    m.setObjective(toll_obj, GRB.MINIMIZE)

    # print('[Debug] Ensure correct VOTs are bring used in optimization')
    # vot_list = [users.data[u]['vot'] for u in range(num_users)]
    # print('[Debug] Expect 1,1. Actual: ', min(vot_list), max(vot_list))

    # run the optimization
    print("Starting optimization")
    m.optimize()
    print("Optimization complete")

    # If infeasible, terminate program
    assert m.status != GRB.INFEASIBLE

    # extract the solution flows
    x = np.zeros((num_edges, num_users))
    x_dict = m.getAttr('x', x_eu)
    for e in range(num_edges):
        for u in range(num_users):
            x[e, u] = x_dict[e, u]

    f = np.sum(x, axis=1)

    # print('[Debug] Expecting dim x as, ', num_edges, num_users)
    # print('[Debug] Expecting dim f as, ', num_edges)
    # print('[Debug] Dimensions of x and f are, ', np.shape(x), np.shape(f))

    return x, f


def optimal_flow(network, users):

    num_edges = network.NumEdges
    num_users = users.num_users

    """ Gurobi optimization model """

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

        m.addConstrs(
            sum(x_eu[g, u] for g in network.prev(node=n)) ==
            sum(x_eu[g, u] for g in network.next(node=n))
            for n in exclude_od_nodes)

    # capacity constraints (testing the for loop so that we can extract duals later)
    for e in range(num_edges):
        m.addConstr(x_e[e] <= network.capacity[e], name='capacity'+str(e))

    # objective function
    # so_obj = sum([users.data[u]['vot'] * x_eu[e, u] * network.edge_latency[e]
    #               for e in range(num_edges) for u in range(num_users)])
    #
    so_obj = 0
    for e in range(num_edges):
        for u in range(num_users):
            so_obj += users.data[u]['vot'] * x_eu[e, u] * network.edge_latency[e]

    m.setObjective(so_obj, GRB.MINIMIZE)

    # print('[Debug] Ensure correct VOTs are bring used in optimization')
    # vot_list = [users.data[u]['vot'] for u in range(num_users)]
    # print('[Debug] Expect 1,1. Actual: ', min(vot_list), max(vot_list))

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

    # print('[Debug] Expecting dim x as, ', num_edges, num_users)
    # print('[Debug] Expecting dim f as, ', num_edges)
    # print('[Debug] Dimensions of x and f are, ', np.shape(x), np.shape(f))

    return x, f
