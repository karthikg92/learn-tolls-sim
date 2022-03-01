from map_plot import plot_edge_values
from network import Network
from users import Users
from utils import *


class NetworkPlots:
    def __init__(self):
        # city name for the experiment
        self.city = 'SiouxFalls'

        # Initialize users
        users = Users(self.city)
        print('Number of users: ', users.num_users)

        # Initialize network
        network = Network(self.city)
        physical_links = network.NumEdges
        network.add_outside_option(users)

        plot_edge_values(network.capacity_list()[:physical_links],
                         'ResultLogs/capacity.png',
                         'Edge Capacity (vehicles/hour)')

        vector_to_file('ResultLogs/capacity.csv', network.capacity_list()[:physical_links])

        plot_edge_values(network.latency_list()[:physical_links],
                         'ResultLogs/latency.png',
                         'Edge Latency (hours)')
        vector_to_file('ResultLogs/latency.csv', network.latency_list()[:physical_links])


