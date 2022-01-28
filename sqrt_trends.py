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


class SqrtTrends:

    def __init__(self):

        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Range of T values for the experiment
        self.Trange = [10, 50, 100, 250, 500]

        # Simulate gradient descent
        print('[sqrt_trends] Simulating gradient descent')
        log_gr_desc = self.simulate_gr_desc()
        self.plot(log_gr_desc, self.folder_path + 'gr_desc')

        # simulate stochastic program
        print('[sqrt_trends] Simulating stochastic program')
        log_stochastic = self.simulate_stochastic_program()
        self.plot(log_stochastic, self.folder_path + 'stochastic')

        # compare performance
        self.comparison_plot(log_gr_desc, log_stochastic, self.folder_path + 'comparison')

    @staticmethod
    def create_folder():
        now = datetime.now()
        path_time = now.strftime("%m-%d-%H-%M-%S")
        try:
            root_folder_path = 'ResultLogs/SqrtTrends_' + path_time + '/'
            os.mkdir(root_folder_path)
        finally:
            print('[JustifyVOT] Folder exists')
        return root_folder_path

    @staticmethod
    def plot(log, path):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 8)

        axes[0].scatter(log['T'], log['regret'])
        axes[0].set_xlabel('T')
        axes[0].set_ylabel('Average Normalized Regret')

        axes[1].scatter(log['T'], log['violation'])
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('Average Normalized Capacity Violation')

        plt.tight_layout()
        plt.savefig(path + '.png')
        plt.close()

    @staticmethod
    def comparison_plot(log_gr, log_sto, path):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 8)

        axes[0].plot(log_gr['T'], log_gr['regret'], '*-', c='tab:blue', label='gr_desc')
        axes[0].plot(log_sto['T'], log_sto['regret'], '*-', c='tab:orange', label='stochastic')
        axes[0].set_xlabel('T')
        axes[0].set_ylabel('Average Normalized Regret')
        axes[0].legend(loc="upper right")

        axes[1].plot(log_gr['T'], log_gr['violation'], '*-', c='blue', label='gr_desc')
        axes[1].plot(log_sto['T'], log_sto['violation'], '*-', c='red', label='stochastic')
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('Average Normalized Capacity Violation')
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(path + '.png')
        plt.close()

    def simulate_gr_desc(self):

        # Initialize network and users
        network = Network(self.city)
        users = Users(self.city)

        # Initialize log
        log = []

        for T in self.Trange:

            print('[sqrt_trends] Running for T=%d' % T)

            # Initialize tolls
            gr_desc_tolls = np.zeros(network.NumEdges)

            # set step size
            step_size = 1e-1/np.sqrt(T)

            # Initialize performance parameters
            obj = 0
            opt_obj = 0
            violation_vec = np.zeros(network.NumEdges)

            for t in range(T):

                # compute user equilibrium
                x, f = user_equilibrium_with_tolls(network, users, gr_desc_tolls)
                x_opt, f_opt = optimal_flow(network, users)

                # Updating gradient descent tolls
                gr_desc_tolls += step_size * (f - np.array(network.capacity))
                gr_desc_tolls[gr_desc_tolls < 0] = 0

                # compute regret
                obj += network.latency_array() @ x @ users.vot_array()
                opt_obj += network.latency_array() @ x_opt @ users.vot_array()

                # compute capacity violation
                violation_vec += f - network.capacity_array()

                # instantiate new realization of VOTs
                users.new_instance()

            # Log parameters for fixed T

            # Normalized regret
            average_normalized_regret = (obj - opt_obj)/opt_obj

            # Normalized capacity
            max_violation = max(violation_vec)
            max_index = int(np.where(violation_vec == max_violation)[0][0])
            average_normalized_violation = max(max_violation / network.capacity_list()[max_index] / T, 0)

            # Adding to log
            log.append([T, average_normalized_regret, average_normalized_violation])

        # clean up log
        log = pd.DataFrame(log, columns=['T', 'regret', 'violation'])

        return log

    def simulate_stochastic_program(self):

        # Initialize network and users
        network = Network(self.city)
        users = Users(self.city)

        # Initialize log
        log = []

        # compute benchmark tolls
        toll = compute_stochastic_program_toll(network, users)

        for T in self.Trange:

            print('[sqrt_trends] Running for T=%d' % T)

            # Initialize performance parameters
            obj = 0
            opt_obj = 0
            violation_vec = np.zeros(network.NumEdges)

            for t in range(T):
                # compute user equilibrium
                x, f = user_equilibrium_with_tolls(network, users, toll)
                x_opt, f_opt = optimal_flow(network, users)

                # compute regret
                obj += network.latency_array() @ x @ users.vot_array()
                opt_obj += network.latency_array() @ x_opt @ users.vot_array()

                # compute capacity violation
                violation_vec += f - network.capacity_array()

                # instantiate new realization of VOTs
                users.new_instance()

            # Log parameters for fixed T

            # Normalized regret
            average_normalized_regret = (obj - opt_obj) / opt_obj

            # Normalized capacity
            max_violation = max(violation_vec)
            max_index = int(np.where(violation_vec == max_violation)[0][0])
            average_normalized_violation = max(max_violation / network.capacity_list()[max_index] / T, 0)

            # Adding to log
            log.append([T, average_normalized_regret, average_normalized_violation])

        # clean up log
        log = pd.DataFrame(log, columns=['T', 'regret', 'violation'])

        return log
