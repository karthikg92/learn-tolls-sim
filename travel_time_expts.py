import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


from network import Network
from users import Users
from optimization import compute_stochastic_program_toll
from optimization import compute_same_vot_toll
from optimization import user_equilibrium_with_tolls
from optimization import optimal_flow


class TravelTimeExperiments:

    def __init__(self):
        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Range of T values for the experiment
        #self.Trange = [5, 25, 50, 100, 250, 500]
        self.Trange = [2]

        # Initialize networks and users

        # users for the experiment
        users = Users(self.city)

        # city network for the experiment
        network = Network(self.city)
        network.add_outside_option(users) # Adding the outside option links

        # Compute toll without VOT
        no_vot_toll = self.compute_no_vot_toll(network, users)

        # compute static toll with VOT consideration
        static_vot_toll = self.compute_static_vot_toll(network, users)

        # compare these methods with gradient descent
        # run for Trange time steps and log total travel times
        self.compare(network, users, no_vot_toll, static_vot_toll)

    def compare(self, network, users, no_vot_toll, static_vot_toll):

        log = pd.DataFrame(columns=['T', 'regret_gr_desc', 'regret_no_vot', 'regret_stochastic',
                                    'ttt_gr_desc', 'ttt_no_vot', 'ttt_stochastic',
                                    'vio_gr_desc', 'vio_no_vot', 'vio_stochastic'])

        for T in self.Trange:

            # Initialize performance parameters
            obj_gr_desc = 0
            obj_stochastic = 0
            obj_no_vot = 0
            obj_opt = 0

            ttt_gr_desc = 0
            ttt_stochastic = 0
            ttt_no_vot = 0

            vio_gr_desc = np.zeros(network.NumEdges)
            vio_stochastic = np.zeros(network.NumEdges)
            vio_no_vot = np.zeros(network.NumEdges)

            gr_desc_tolls = np.zeros(network.NumEdges)
            step_size = 1e-1 / np.sqrt(T)

            for t in range(T):
                print("[TravelTimeExperiments] Iteration %d of %d" % (t, T))

                # compute optimal flow
                x_opt, _ = optimal_flow(network, users)
                obj_opt += network.latency_array() @ x_opt @ users.vot_array()

                # No VOT consideration for toll computation
                x, f = user_equilibrium_with_tolls(network, users, no_vot_toll)
                obj_no_vot += network.latency_array() @ x @ users.vot_array()
                ttt_no_vot += f @ network.edge_latency
                vio_no_vot += f - network.capacity

                # Stochastic program tolls
                x, f = user_equilibrium_with_tolls(network, users, static_vot_toll)
                obj_stochastic += network.latency_array() @ x @ users.vot_array()
                ttt_stochastic += f @ network.edge_latency
                vio_stochastic += f - network.capacity

                # Gradient descent algorithm
                x, f = user_equilibrium_with_tolls(network, users, gr_desc_tolls)
                obj_gr_desc += network.latency_array() @ x @ users.vot_array()
                ttt_gr_desc += f @ network.edge_latency
                vio_gr_desc += f - network.capacity

                # Updating gradient descent tolls
                gr_desc_tolls += step_size * (f - np.array(network.capacity))
                gr_desc_tolls[gr_desc_tolls < 0] = 0

                # Draw a new user VOT realization for next time step
                users.new_instance()

            # Parameter logging

            # Normalized regret
            no_vot_regret = (obj_no_vot - obj_opt) / obj_opt
            stochastic_regret = (obj_stochastic - obj_opt) / obj_opt
            gr_desc_regret = (obj_gr_desc - obj_opt) / obj_opt

            # Normalized capacity
            no_vot_vio = self.compute_normalized_violation(vio_no_vot, T, network.capacity_array())
            stochastic_vio = self.compute_normalized_violation(vio_stochastic, T, network.capacity_array())
            gr_desc_vio = self.compute_normalized_violation(vio_gr_desc, T, network.capacity_array())

            # Total travel time
            ttt_no_vot_avg = ttt_no_vot / T
            ttt_gr_desc_avg = ttt_gr_desc / T
            ttt_stochastic_avg = ttt_stochastic / T

            log = log.append({'T': T,
                              'regret_gr_desc': gr_desc_regret,
                              'regret_no_vot': no_vot_regret,
                              'regret_stochastic': stochastic_regret,
                              'ttt_gr_desc': ttt_gr_desc_avg,
                              'ttt_no_vot': ttt_no_vot_avg,
                              'ttt_stochastic': ttt_stochastic_avg,
                              'vio_gr_desc': gr_desc_vio,
                              'vio_no_vot': no_vot_vio,
                              'vio_stochastic': stochastic_vio},
                             ignore_index=True)

            # Invoke the plot function
            self.performance_plot(log, self.folder_path + 'comparison')

    @staticmethod
    def create_folder():
        now = datetime.now()
        path_time = now.strftime("%m-%d-%H-%M-%S")
        try:
            root_folder_path = 'ResultLogs/TravelTimeExperiments_' + path_time + '/'
            os.mkdir(root_folder_path)
        finally:
            print('[TravelTimeExpts] Folder exists')
        return root_folder_path

    @staticmethod
    def compute_no_vot_toll(network, users):
        tolls = compute_same_vot_toll(network, users)
        return tolls

    @staticmethod
    def compute_static_vot_toll(network, users):
        tolls = compute_stochastic_program_toll(network, users)
        return tolls

    @staticmethod
    def compute_normalized_violation(violation_vec, t, capacity_array):
        max_violation = max(violation_vec)
        max_index = np.where(violation_vec == max_violation)[0][0]
        average_normalized_violation = max(max_violation / capacity_array[max_index] / t, 0)
        return average_normalized_violation

    @staticmethod
    def performance_plot(log, path):
        fig, axes = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches(18, 8)

        axes[0].plot(log['T'], log['regret_gr_desc'], '*-', c='tab:blue', label='gr_desc')
        axes[0].plot(log['T'], log['regret_stochastic'], '*-', c='tab:orange', label='stochastic')
        axes[0].plot(log['T'], log['regret_no_vot'], '*-', c='tab:green', label='no_vot')
        axes[0].set_xlabel('T')
        axes[0].set_ylabel('Average Normalized Regret')
        axes[0].legend(loc="upper right")

        axes[1].plot(log['T'], log['vio_gr_desc'], '*-', c='tab:blue', label='gr_desc')
        axes[1].plot(log['T'], log['vio_stochastic'], '*-', c='tab:orange', label='stochastic')
        axes[1].plot(log['T'], log['vio_no_vot'], '*-', c='tab:green', label='no_vot')
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('Average Normalized Capacity Violation')
        axes[1].legend(loc="upper right")

        axes[2].plot(log['T'], log['ttt_gr_desc'], '*-', c='tab:blue', label='gr_desc')
        axes[2].plot(log['T'], log['ttt_stochastic'], '*-', c='tab:orange', label='stochastic')
        axes[2].plot(log['T'], log['ttt_no_vot'], '*-', c='tab:green', label='no_vot')
        axes[2].set_xlabel('T')
        axes[2].set_ylabel('Average Travel Time')
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(path + '.png')
        plt.close()

        # save dataframe
        log.to_csv(path + '.csv')