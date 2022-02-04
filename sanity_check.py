import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from network import Network
from users import Users
from optimization import optimal_flow
from optimization import user_equilibrium_with_tolls
from optimization import compute_stochastic_program_toll


"""

Things to check:
1. If VOT = 1, then the regret of the stochastic program algorithm will be ~ 0
(idea: uniqueness issue will only screw up the capacity violation) [DONE]
2. If (1) works, create a parallel network: the uniqueness problem might go away [DONE]
3. If (1) works, then: scale the problem and hope that the uniqueness issue will only lead to a small capacity 
violation

Start the VOT justification code on titan: let the code run long enough

Work towards adding an outside option --> Need tolls to be a reasonable number.
Idea: Identify optimal user costs and then add a premium for the outside option.

To think: scale and units for all our quantities.

"""


class StochasticProgramWithConstantVOT:

    def __init__(self):

        # city name for the experiment
        self.city = 'Pigou_reprise'

        # Initialize network
        network = Network(self.city)

        # Initialize users
        users = Users(self.city)
        users.new_instance(fixed_vot=True)  # Ensure VOTs are 1
        print('[Debug] VOTs are 1 for the user. Expected: (1,1). Actual:', np.max(users.vot_array()),
              np.min(users.vot_array()))

        # evaluate performance
        print('[sanity_check] Evaluating stochastic programs performance')
        self.compute_constant_vot_performance(network, users)

    @staticmethod
    def compute_constant_vot_performance(network, users):

        # compute stochastic program tolls
        toll = compute_stochastic_program_toll(network, users, constant_vot=True)
        print('[Debug] Toll of stochastic program: ', toll)
        print('[Debug] Ensure tolls are positive. Expect value >=0. Actual: ', min(toll))

        # compute optimal solution
        x_opt, f_opt = optimal_flow(network, users)
        opt_obj = network.latency_array() @ x_opt @ users.vot_array()

        # compute user equilibrium
        x, f = user_equilibrium_with_tolls(network, users, toll)
        obj = network.latency_array() @ x @ users.vot_array()

        print('[Debug]: x =', x)
        print('[Debug]: x_opt =', x_opt)

        # compute regret (\sum_{e,u} x_eu * vot_u * latency_e)
        normalized_regret = (obj - opt_obj) / opt_obj

        # compute capacity violation
        violation_vec = (f - network.capacity_array())/network.capacity_array()
        normalized_violation = max(max(violation_vec), 0)

        print(obj, opt_obj)

        # print statistics
        print("[sanity_check][constant_vot_stochastic_program] Normalized regret = ", normalized_regret)
        print("[sanity_check][constant_vot_stochastic_program] Normalized violation = ", normalized_violation)

        return None


class TestOutsideOption:

    def __init__(self):
        # city name for the experiment
        self.city = 'SiouxFalls'

        # Initialize users
        users = Users(self.city)
        print('Number of users: ', users.num_users)

        # Initialize network
        network = Network(self.city)

        print('[Original Network]: Num Nodes: ', network.NumNodes)
        print('[Original Network]: Num Edges: ', network.NumEdges)

        network.add_outside_option(users)

        print('[New Network]: Num Nodes: ', network.NumNodes)
        print('[New Network]: Num Edges: ', network.NumEdges)



        #
