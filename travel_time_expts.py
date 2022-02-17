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
from optimization import UserEquilibriumWithTolls
from optimization import OptimalFlow
from map_plot import plot_edge_values


class TravelTimeExperiments:

    def __init__(self):
        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Range of T values for the experiment
        # self.Trange = [5, 25, 50, 100, 250, 500]
        # self.Trange = [10000]
        # self.Trange = [2, 5]
        self.Trange = [5, 25, 50, 100, 250]

        # Initialize networks and users

        # users for the experiment
        users = Users(self.city)

        # city network for the experiment
        network = Network(self.city)
        self.num_physical_edges = network.NumEdges
        network.add_outside_option(users)  # Adding the outside option links

        # Compute toll without VOT
        # no_vot_toll = self.compute_no_vot_toll(network, users)
        # plot_edge_values(no_vot_toll[:self.num_physical_edges],
        #                  self.folder_path+'no_vot_toll.png',
        #                  "Toll ignoring VOT (in $)",
        #                  truncate_flag=False,
        #                  plot_lim=[0, 0.5])

        # Compute toll without population mean VOT
        population_mean_vot_toll = self.compute_population_mean_vot_toll(network, users)
        plot_edge_values(population_mean_vot_toll[:self.num_physical_edges],
                         self.folder_path+'population_mean_toll.png',
                         "Toll with population mean VOT (in $)",
                         truncate_flag=False,
                         plot_lim=[0, 1.5])


        # Compute toll with grouo-specific mean VOT
        group_specific_vot_toll = self.compute_group_specific_mean_vot_toll(network, users)
        plot_edge_values(group_specific_vot_toll[:self.num_physical_edges],
                         self.folder_path+'group_specific_VOT_toll.png',
                         "Toll with group-specific VOT (in $)",
                         truncate_flag=False,
                         plot_lim=[0, 1.5])


        # compute static toll with VOT consideration
        # static_vot_toll = self.compute_static_vot_toll(network, users)
        # print('Stochastic program:')
        # print('min toll: ', min(static_vot_toll))
        # print('max toll: ', max(static_vot_toll))
        # print('average toll: ', np.mean(static_vot_toll))
        # plot_edge_values(static_vot_toll[:self.num_physical_edges],
        #                  self.folder_path+'stochastic_program_toll.png',
        #                  "Fixed toll from stochastic program (in $)",
        #                  truncate_flag=False,
        #                  plot_lim=[0, 0.5])

        # compare these methods with gradient descent
        # run for Trange time steps and log total travel times

        self.compare(network, users, population_mean_vot_toll, group_specific_vot_toll)

    def compare(self, network, users, population_mean_vot_toll, group_specific_vot_toll):

        log = pd.DataFrame(columns=['T', 'regret_gr_desc', 'regret_no_vot', 'regret_stochastic',
                                    'ttt_gr_desc', 'ttt_no_vot', 'ttt_stochastic',
                                    'vio_gr_desc', 'vio_no_vot', 'vio_stochastic'])

        # Initializing user equilibrium solver
        ue_with_tolls = UserEquilibriumWithTolls(network, users, population_mean_vot_toll)

        # initializing optimal flow solver
        opt_solver = OptimalFlow(network, users)

        for T in self.Trange:

            # Initialize performance parameters
            obj_gr_desc = 0
            obj_stochastic = 0
            obj_no_vot = 0
            obj_const_update = 0
            obj_opt = 0

            ttt_gr_desc = 0
            ttt_stochastic = 0
            ttt_const_update = 0
            ttt_no_vot = 0

            vio_gr_desc = np.zeros(network.NumEdges)
            vio_stochastic = np.zeros(network.NumEdges)
            vio_const_update = np.zeros(network.NumEdges)
            vio_no_vot = np.zeros(network.NumEdges)

            # Initialize gr_desc tolls
            gr_desc_tolls = np.zeros(network.NumEdges)

            # Initialize constant update tolls
            const_update_tolls = np.zeros(network.NumEdges)

            # Track evolution of total tolls
            total_toll_gr_desc = [sum(gr_desc_tolls[:network.physical_num_edges])]
            total_virtual_toll_gr_desc = [sum(gr_desc_tolls[network.physical_num_edges:])]

            total_toll_const_update = [sum(const_update_tolls[:network.physical_num_edges])]
            total_virtual_toll_const_update = [sum(const_update_tolls[network.physical_num_edges:])]


            # TODO: check if  convergence will be achieved even without warm start
            # gr_desc_tolls = static_vot_toll

            # step_size = 1e-1 / np.sqrt(T)
            # step_size = 1e-2 / np.sqrt(T)
            step_size = 5e-4 / np.sqrt(T)  # Known to work!
            # step_size = 5e-3 / np.sqrt(T)

            for t in range(T):
                print("[TravelTimeExperiments] Iteration %d of %d" % (t, T))

                # compute optimal flow
                # x_opt, _ = optimal_flow(network, users)
                opt_solver.set_obj(users)
                x_opt, _ = opt_solver.solve()

                obj_opt += network.latency_array() @ x_opt @ users.vot_array()

                # No VOT consideration for toll computation

                # test: comments
                # x, f = user_equilibrium_with_tolls(network, users, no_vot_toll)
                # test: new version
                ue_with_tolls.set_obj(users, population_mean_vot_toll)
                x, f = ue_with_tolls.solve()

                obj_no_vot += network.latency_array() @ x @ users.vot_array()
                ttt_no_vot += f @ network.edge_latency
                vio_no_vot += f - network.capacity

                # Stochastic program tolls

                # test comment
                # x, f = user_equilibrium_with_tolls(network, users, static_vot_toll)

                # replacement:

                # TODO: initially 1e-5, then 1e-3 was better
                # noise_vector = 1e-3 * (np.random.rand(network.NumEdges) - 0.5)
                # noise_vector[noise_vector < 0] = 0
                # noise_vector[network.physical_num_edges:] = 0
                #
                # static_vot_toll_with_noise = static_vot_toll + noise_vector
                #
                # ue_with_tolls.set_obj(users, static_vot_toll_with_noise)
                ue_with_tolls.set_obj(users, group_specific_vot_toll)
                x, f = ue_with_tolls.solve()

                obj_stochastic += network.latency_array() @ x @ users.vot_array()
                ttt_stochastic += f @ network.edge_latency
                vio_stochastic += f - network.capacity

                # Gradient descent algorithm

                # test: comment
                # x, f = user_equilibrium_with_tolls(network, users, gr_desc_tolls)

                # new version test
                ue_with_tolls.set_obj(users, gr_desc_tolls)
                x, f = ue_with_tolls.solve()

                obj_gr_desc += network.latency_array() @ x @ users.vot_array()
                ttt_gr_desc += f @ network.edge_latency
                vio_gr_desc += f - network.capacity

                # Updating gradient descent tolls
                gr_desc_tolls += step_size * (f - np.array(network.capacity))
                gr_desc_tolls[gr_desc_tolls < 0] = 0

                total_toll_gr_desc.append(sum(gr_desc_tolls[:network.physical_num_edges]))
                total_virtual_toll_gr_desc.append(sum(gr_desc_tolls[network.physical_num_edges:]))

                plt.plot(total_toll_gr_desc, label='total toll on real edges')
                plt.plot(total_virtual_toll_gr_desc, label='total toll on virtual edges')
                plt.legend()
                plt.savefig(self.folder_path+'toll_evoultion_gr_desc_' + str(T) + '.png')
                plt.close()

                # plt.hist(gr_desc_tolls[:network.physical_num_edges])
                # plt.savefig(self.folder_path+'current_histogram' + str(T) + '.png')
                # plt.close()

                # Constant Toll Update Policy (courtesy Ed)
                ue_with_tolls.set_obj(users, const_update_tolls)
                x, f = ue_with_tolls.solve()
                const_update_tolls += 0.05 * np.sign(f - network.capacity)
                const_update_tolls[const_update_tolls < 0] = 0

                obj_const_update += network.latency_array() @ x @ users.vot_array()
                vio_const_update += f - network.capacity
                ttt_const_update += f @ network.edge_latency

                total_toll_const_update.append(sum(const_update_tolls[:network.physical_num_edges]))
                total_virtual_toll_const_update.append(sum(const_update_tolls[network.physical_num_edges:]))

                plt.plot(total_toll_const_update, label='total toll on real edges')
                plt.plot(total_virtual_toll_const_update, label='total toll on virtual edges')
                plt.legend()
                plt.savefig(self.folder_path + 'toll_evoultion_const_update_' + str(T) + '.png')
                plt.close()

                # Draw a new user VOT realization for next time step
                users.new_instance()

            # Parameter logging

            # Normalized regret
            no_vot_regret = (obj_no_vot - obj_opt) / obj_opt
            stochastic_regret = (obj_stochastic - obj_opt) / obj_opt
            gr_desc_regret = (obj_gr_desc - obj_opt) / obj_opt
            const_update_regret = (obj_const_update - obj_opt) / obj_opt

            # Normalized capacity
            no_vot_vio = self.compute_normalized_violation(vio_no_vot, T, network.capacity_array())
            stochastic_vio = self.compute_normalized_violation(vio_stochastic, T, network.capacity_array())
            gr_desc_vio = self.compute_normalized_violation(vio_gr_desc, T, network.capacity_array())
            const_update_vio = self.compute_normalized_violation(vio_const_update, T, network.capacity_array())

            # Total travel time
            ttt_no_vot_avg = ttt_no_vot / T
            ttt_gr_desc_avg = ttt_gr_desc / T
            ttt_stochastic_avg = ttt_stochastic / T
            ttt_const_update = ttt_const_update / T

            log = log.append({'T': T,
                              'regret_gr_desc': gr_desc_regret,
                              'regret_no_vot': no_vot_regret,
                              'regret_stochastic': stochastic_regret,
                              'regret_const_update': const_update_regret,
                              'ttt_gr_desc': ttt_gr_desc_avg,
                              'ttt_no_vot': ttt_no_vot_avg,
                              'ttt_stochastic': ttt_stochastic_avg,
                              'ttt_const_update': ttt_const_update,
                              'vio_gr_desc': gr_desc_vio,
                              'vio_no_vot': no_vot_vio,
                              'vio_stochastic': stochastic_vio,
                              'vio_const_update': const_update_vio},
                             ignore_index=True)

            # Invoke the plot function
            self.performance_plot(log, self.folder_path + 'comparison')

            # Plot gradient descent tolls on map
            plot_edge_values(gr_desc_tolls[:network.physical_num_edges],
                             self.folder_path+'tolls_t_' + str(T) + '.png',
                             'Gradient descent tolls after T = ' + str(T),
                             truncate_flag=False,
                             plot_lim=[0, 1.5])

            # Compare tolls in histogram
            plt.hist(gr_desc_tolls[:network.physical_num_edges], alpha=0.3, label='gradient descent')
            plt.hist(group_specific_vot_toll[:network.physical_num_edges],  alpha=0.3, label='group specific mean VOT')
            plt.hist(population_mean_vot_toll[:network.physical_num_edges], alpha=0.3, label='population mean VOT')
            plt.hist(const_update_tolls[:network.physical_num_edges], alpha=0.3, label='constant update')
            plt.legend()
            plt.savefig(self.folder_path+'histogram'+str(T)+'.png', dpi=250)
            plt.close()

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
    def compute_population_mean_vot_toll(network, users):
        tolls = compute_same_vot_toll(network, users, vot=users.population_vot_mean())
        return tolls

    @staticmethod
    def compute_group_specific_mean_vot_toll(network, users):
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

        axes[0].plot(log['T'], log['regret_gr_desc'], '*-', c='tab:blue', label='gradient descent')
        axes[0].plot(log['T'], log['regret_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
        axes[0].plot(log['T'], log['regret_no_vot'], '*-', c='tab:green', label='population mean VOT')
        axes[0].plot(log['T'], log['regret_const_update'], '*-', c='tab:red', label='constant update')
        axes[0].set_xlabel('T')
        axes[0].set_ylabel('Average Normalized Regret')
        axes[0].legend(loc="upper right")

        axes[1].plot(log['T'], log['vio_gr_desc'], '*-', c='tab:blue', label='gradient descent')
        axes[1].plot(log['T'], log['vio_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
        axes[1].plot(log['T'], log['vio_no_vot'], '*-', c='tab:green', label='population mean VOT')
        axes[1].plot(log['T'], log['vio_const_update'], '*-', c='tab:red', label='constant update')
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('Average Normalized Capacity Violation')
        axes[1].legend(loc="upper right")

        axes[2].plot(log['T'], log['ttt_gr_desc'], '*-', c='tab:blue', label='gradient descent')
        axes[2].plot(log['T'], log['ttt_stochastic'], '*-', c='tab:orange', label='group-specific mean VOT')
        axes[2].plot(log['T'], log['ttt_no_vot'], '*-', c='tab:green', label='population mean VOT')
        axes[2].plot(log['T'], log['ttt_const_update'], '*-', c='tab:red', label='constant update')
        axes[2].set_xlabel('T')
        axes[2].set_ylabel('Average Travel Time')
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(path + '.png')
        plt.close()

        # save dataframe
        log.to_csv(path + '.csv')


"""
1. Debug noisy tolls with 2 link example
2. Different means for user groups -- DONE
3. Update NoVOT to use population mean -- DONE
4. Add Ed's Ideas (choose scale carefully) -- DONE

Additional ideas for Appendix:
1. Multiple cities
2. Warm starts
3. Convergence dynamics 
    (i.e., don't need to wait till T=100 for tolls to come within a ball)
4. VCG discussion
"""