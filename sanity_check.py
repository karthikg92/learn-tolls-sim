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
from optimization import UserEquilibriumWithTolls
from optimization import OptimalFlow

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


class CheckIfNoiseHelpsStochasticProgram:

    def __init__(self):
        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'Pigou_reprise'

        # Initialize network
        network = Network(self.city)

        # Initialize users
        users = Users(self.city)
        users.new_instance(fixed_vot=True)


        self.run_experiments(network, users)

    def run_experiments(self, network: Network, users: Users):

        group_specific_vot_toll = self.compute_group_specific_mean_vot_toll(network, users)
        print('[SanityCheck] Optimal tolls without noise: ', group_specific_vot_toll)

        T_range = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

        # Initializing user equilibrium solver
        ue_with_tolls = UserEquilibriumWithTolls(network, users, group_specific_vot_toll)

        # initializing optimal flow solver
        opt_solver = OptimalFlow(network, users)

        # Initialize log
        log = pd.DataFrame(columns=['T', 'regret_stochastic', 'ttt_stochastic', 'vio_stochastic'])

        for T in T_range:
            print('[SanityCheck][CheckIfNoiseHelps] T = ', T)

            # Initialize performance parameters
            obj_stochastic = 0
            obj_opt = 0
            ttt_stochastic = 0
            vio_stochastic = np.zeros(network.NumEdges)

            for t in range(T):

                # Compute optimal flows
                opt_solver.set_obj(users)
                x_opt, _ = opt_solver.solve()
                obj_opt += network.latency_array() @ x_opt @ users.vot_array()

                # Update tolls

                noise_vector = 1e-5 * (np.random.rand(network.NumEdges) - 0.5)
                noise_vector[noise_vector < 0] = 0
                noise_vector[network.physical_num_edges:] = 0
                group_specific_vot_toll_with_noise = group_specific_vot_toll + noise_vector

                # Compute response to new noisy tolls
                ue_with_tolls.set_obj(users, group_specific_vot_toll_with_noise)
                # ue_with_tolls.set_obj(users, group_specific_vot_toll)
                x, f = ue_with_tolls.solve()

                print('tolls: ', group_specific_vot_toll_with_noise)
                print('x:', x)
                print('f:', f)

                obj_stochastic += network.latency_array() @ x @ users.vot_array()
                ttt_stochastic += f @ network.edge_latency
                vio_stochastic += f - network.capacity


            # logging output

            # Normalized regret
            stochastic_regret = (obj_stochastic - obj_opt) / obj_opt

            # Normalized capacity
            stochastic_vio = self.compute_normalized_violation(vio_stochastic, T, network.capacity_array())

            # Total travel time
            ttt_stochastic_avg = ttt_stochastic / T

            log = log.append({'T': T,
                              'regret_stochastic': stochastic_regret,
                              'ttt_stochastic': ttt_stochastic_avg,
                              'vio_stochastic': stochastic_vio},
                             ignore_index=True)

            # Invoke the plot function
            fig, axes = plt.subplots(nrows=1, ncols=3)
            fig.set_size_inches(18, 8)

            axes[0].plot(log['T'], log['regret_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
            axes[0].set_xlabel('T')
            axes[0].set_ylabel('Average Normalized Regret')
            axes[0].legend(loc="upper right")

            axes[1].plot(log['T'], log['vio_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
            axes[1].set_xlabel('T')
            axes[1].set_ylabel('Average Normalized Capacity Violation')
            axes[1].legend(loc="upper right")

            axes[2].plot(log['T'], log['ttt_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
            axes[2].set_xlabel('T')
            axes[2].set_ylabel('Average Travel Time')
            axes[2].legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(self.folder_path + 'comparison.png')
            plt.close()

        return None

    @staticmethod
    def compute_group_specific_mean_vot_toll(network, users):
        tolls = compute_stochastic_program_toll(network, users, constant_vot=True)
        return tolls

    @staticmethod
    def compute_normalized_violation(violation_vec, t, capacity_array):
        max_violation = max(violation_vec)
        max_index = np.where(violation_vec == max_violation)[0][0]
        average_normalized_violation = max(max_violation / capacity_array[max_index] / t, 0)
        return average_normalized_violation

    @staticmethod
    def create_folder():
        now = datetime.now()
        path_time = now.strftime("%m-%d-%H-%M-%S")
        try:
            root_folder_path = 'ResultLogs/SanityCheckExperiments_' + path_time + '/'
            os.mkdir(root_folder_path)
        finally:
            print('[SanityCheckExpts] Folder exists')
        return root_folder_path


class ComputeOptimalTTT:

    def __init__(self):

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Initialize users
        users = Users(self.city)
        users.new_instance(fixed_vot=True)

        # Initialize network
        network = Network(self.city)
        self.num_physical_edges = network.NumEdges
        network.add_outside_option(users)  # Adding the outside option links

        optimal_flow(network, users)

