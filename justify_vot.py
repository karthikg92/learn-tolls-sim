import os
import numpy as np
from datetime import datetime
from network import Network
from users import Users
from optimizer import compute_stochastic_program_toll
from optimizer import compute_same_vot_toll
from optimizer import user_equilibrium_with_tolls


class JustifyVOT:

    def __init__(self):

        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Time steps for the experiment
        self.max_steps = 100

        # city network for the experiment
        network = Network(self.city)

        # users for the experiment
        users = Users(self.city)

        # Compute toll without VOT
        no_vot_toll = self.compute_no_vot_toll(network, users)

        # compute static toll with VOT consideration
        static_vot_toll = self.compute_static_vot_toll(network, users)

        # run T time steps and log total travel times
        self.compare(self.max_steps, network, users, no_vot_toll, static_vot_toll)

    @staticmethod
    def create_folder():
        now = datetime.now()
        path_time = now.strftime("%m-%d-%H-%M-%S")
        try:
            root_folder_path = 'ResultLogs/JustifyVOT_' + path_time + '/'
            os.mkdir(root_folder_path)
        finally:
            print('[JustifyVOT] Folder exists')
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
    def compare(max_steps, network, users, no_vot_toll, static_vot_toll):

        ttt_no_vot = []
        ttt_static_vot = []

        for t in range(max_steps):

            _, f = user_equilibrium_with_tolls(network, users, no_vot_toll)
            ttt_no_vot.append(f @ network.edge_latency)

            _, f = user_equilibrium_with_tolls(network, users, static_vot_toll)
            ttt_static_vot.append(f @ network.edge_latency)

            users.new_instance()

        print("Average TTT without considering VOT = %d" % (np.mean(ttt_no_vot)))
        print("Average TTT considering VOT = %d" % (np.mean(ttt_static_vot)))
        #
        # """
        # Compare total travel times when tolls are set with and without vot consideration
        # """
        # time_steps = 100  # total time for the simulation
        #
        # no_vot_toll = None
        # vot_toll = None
        #
        # ttt_no_vot = []
        # ttt_vot = []
        #
        # for t in range(time_steps):
        #     users.vot_realization()
        #
        #     # tolls without VOT consideration
        #     x, f = response(users, no_toll)
        #     ttt.no_toll.append(compute_ttt(x))
        #
        #     # tolls with VOT consideration
        #     x, f = response(users, no_toll)
        #     ttt.no_toll.append(compute_ttt(x))
