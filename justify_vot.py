import os
import numpy as np
from datetime import datetime

from network import Network
from users import Users
from optimization import compute_stochastic_program_toll
from optimization import compute_same_vot_toll
from optimization import user_equilibrium_with_tolls


class JustifyVOT:

    def __init__(self):

        # Create a folder for this experiment (Not needed for now!)
        # self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Time steps for the experiment
        self.max_steps = 1000

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
    def compute_violation(list_vio):
        s = sum(list_vio)
        s[s < 0] = 0
        return np.linalg.norm(s)

    def compare(self, max_steps, network, users, no_vot_toll, static_vot_toll):

        ttt_no_vot = []
        ttt_static_vot = []
        ttt_gr_desc = []

        vio_no_vot = []
        vio_static_vot = []
        vio_gr_desc = []

        gr_desc_tolls = np.zeros(network.NumEdges)
        # gr_desc_tolls = no_vot_toll
        step_size = 1e-1/np.sqrt(max_steps)

        for t in range(max_steps):

            print("[JustifyVOT] Iteration %d" % t)

            _, f = user_equilibrium_with_tolls(network, users, no_vot_toll)
            ttt_no_vot.append(f @ network.edge_latency)
            vio_no_vot.append(f - network.capacity)

            _, f = user_equilibrium_with_tolls(network, users, static_vot_toll)
            ttt_static_vot.append(f @ network.edge_latency)
            vio_static_vot.append(f - network.capacity)

            _, f = user_equilibrium_with_tolls(network, users, gr_desc_tolls)
            ttt_gr_desc.append(f @ network.edge_latency)
            vio_gr_desc.append(f - network.capacity)

            # Updating gradient descent tolls
            gr_desc_tolls += step_size * (f - np.array(network.capacity))
            gr_desc_tolls[gr_desc_tolls < 0] = 0
            print("Total tolls = %d" % np.sum(gr_desc_tolls))

            users.new_instance()

            print("TTT without considering VOT = %d" % ttt_no_vot[-1])
            print("TTT for stochastic program = %d" % ttt_static_vot[-1])
            print("TTT for gradient descent = %d" % ttt_gr_desc[-1])

            print("Capacity violation without considering VOT = %d " % self.compute_violation([vio_no_vot[-1]]))
            print("Capacity violation  for stochastic program = %d " % self.compute_violation([vio_static_vot[-1]]))
            print("Capacity violation  for gradient descent = %d " % self.compute_violation([vio_gr_desc[-1]]))

        print("Average TTT without considering VOT = %d" % (np.mean(ttt_no_vot)))
        print("Average TTT for stochastic program = %d" % (np.mean(ttt_static_vot)))
        print("Average TTT for gradient descent = %d" % (np.mean(ttt_gr_desc)))

        print("Capacity violation without considering VOT = %d " % self.compute_violation(vio_no_vot))
        print("Capacity violation  for stochastic program = %d " % self.compute_violation(vio_static_vot))
        print("Capacity violation  for gradient descent = %d " % self.compute_violation(vio_gr_desc))
