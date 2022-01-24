import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from network import Network
from users import Users
from optimizer import Optimizer


class Simulation:
    def __init__(self):
        self.log = []
        self.benchmark_tolls = None
        # self.results = None
        # self.last_run_city = None
        # self.last_run_time = None
        # self.log_steps = None
        # self.log_run = []

    @staticmethod
    def compute_objective(vot, x, latency):
        obj = sum([x[e, u] * vot[u] * latency[e] for e in range(len(latency)) for u in range(len(vot))])

        # regret = (obj - opt)/opt
        # for u in range(len(vot)):
        #     for e in range(len(latency)):
        #         regret += vot[u] * latency[e] * (x[e, u] - x_opt[e, u])
        return obj

    @staticmethod
    def compute_violation(f, capacity):
        violation = f - np.array(capacity)

        # cum_violation = running sum of violations
        return violation

    @staticmethod
    def compute_step_size(index, max_steps, policy=None):

        if policy == 'constant':
            return 1e-1 / np.sqrt(max_steps)

        if policy == 'sqrt':
            return 1e-1 / np.sqrt(index + 1)

        if policy == 'reciprocal':
            return 1e-1 / (index + 1)

        if policy == 'ramp':
            split = 5
            current_phase = np.floor(index/(max_steps/split))
            return 1e-1/(2 ** current_phase)/np.sqrt(index+1 - current_phase*max_steps/split)

    def run(self, n: Network, user: Users, opt: Optimizer, max_steps=10000, step_policy=None, alg=None,
            is_stationary=True):
        """
        Inputs: network, users, optimizer, maximum number of steps, folder path
        for results, step size policy, algorithm

        simulate toll update for a fixed number of max_steps
        """

        toll = np.zeros(n.NumEdges)
        if alg == 'benchmark':
            if self.benchmark_tolls is None:
                print("[simulation] Starting to compute benchmark tolls")
                self.benchmark_tolls = opt.compute_benchmark_toll()
                print("[simulation] Benchmark tolls computed")
            toll = self.benchmark_tolls
            step_size = 'NAN'

        cum_obj = 0
        cum_opt_obj = 0
        cum_violation_vec = np.zeros(n.NumEdges)

        log = []

        print("[simulation] Starting the toll iterations")

        for t in range(max_steps):

            # determine the demand and user config
            if is_stationary is False:
                user.new_instance()
                opt = Optimizer(n, user)
            elif is_stationary is True:
                pass

            # compute the user response to the tolls
            if alg == 'gr_desc':
                x, f = opt.toll_flow(toll, is_feasible=False)
            elif alg == 'feasible_desc':
                x, f = opt.toll_flow(toll, is_feasible=True)
            elif alg == 'benchmark':
                x, f = opt.optimal_toll_response(toll)
            else:
                print("[simulation] Error in alg flag")

            # diagnosis saves
            # np.savetxt('x'+str(t)+'.csv', x, delimiter=',')
            # np.savetxt('f' + str(t) + '.csv', f, delimiter=',')
            # np.savetxt('x_opt' + str(t) + '.csv', opt.x_opt, delimiter=',')

            # check if opt flow only goes through one path!

            # update the tolls
            if alg == 'gr_desc' or alg == 'feasible_desc':
                step_size = self.compute_step_size(t, max_steps, policy=step_policy)
                toll = toll - step_size * (opt.f_opt - f)
                toll[toll < 0] = 0  # cap the toll to be non-negative

            # compute performance
            obj = self.compute_objective(user.vot_list(), x, n.latency_list())
            opt_obj = self.compute_objective(user.vot_list(), opt.x_opt, n.latency_list())

            # print("Latency times travel time", obj)

            cum_obj += obj
            cum_opt_obj += opt_obj

            violation_vec = self.compute_violation(f, n.capacity_list())  # this is a vector
            cum_violation_vec += violation_vec

            max_violation = max(violation_vec)
            # print(np.shape(violation_vec))
            # print(np.shape(cum_violation_vec))
            # print(np.where(violation_vec == max_violation))
            max_index = int(np.where(violation_vec == max_violation)[0][0])
            step_violation = max(max_violation / n.capacity_list()[max_index], 0)

            max_cum_violation = max(cum_violation_vec)
            max_index = int(np.where(cum_violation_vec == max_cum_violation)[0][0])
            cum_violation = max(max_cum_violation / n.capacity_list()[max_index], 0) / (t+1)

            # log parameters ever 100 steps and at the last step
            if t % 100 == 0 or t == max_steps - 1:
                step_log = {'step': t+1,
                            'step_size': step_size,
                            'step_regret': (obj - opt_obj) / opt_obj,
                            'step_violation': step_violation,
                            'cum_regret': (cum_obj - cum_opt_obj) / cum_opt_obj,
                            'cum_violation': cum_violation,
                            'total_toll': sum(toll)}
                log.append(step_log)

        self.log.append(log)

        return None

    # TODO: Dont use/ fixit
    # def save_results(self, fname=None):
    #     if fname is None:
    #         path = 'ResultLogs/results.csv'
    #     else:
    #         path = 'ResultLogs/' + fname + '/results.csv'
    #     self.log.to_csv(path)
    #
    #     # print(self.log.head())
    #     # print(self.log.tail())

    # TODO: Dont use/ fixit
    # def plot_results(self, fname=None):
    #
    #     fig, axes = plt.subplots(nrows=2, ncols=3)
    #     fig.set_size_inches(12, 8)
    #
    #     self.log.plot(kind='line', x='step', y='step_size', ax=axes[0, 0])
    #     self.log.plot(kind='line', x='step', y='step_regret', ax=axes[0, 1])
    #     self.log.plot(kind='line', x='step', y='step_violation', ax=axes[0, 2])
    #     self.log.plot(kind='line', x='step', y='total_toll', ax=axes[1, 0])
    #     self.log.plot(kind='line', x='step', y='cum_regret', ax=axes[1, 1])
    #     self.log.plot(kind='line', x='step', y='cum_violation', ax=axes[1, 2])
    #
    #     plt.tight_layout()
    #
    #     if fname is None:
    #         path = 'ResultLogs/results.png'
    #     else:
    #         path = 'ResultLogs/' + fname + '/results.png'
    #
    #     plt.savefig(path)
    #     plt.close()
    #     # df1.plot(ax=axes[0, 0])
    #     # df2.plot(ax=axes[0, 1])
    #
    #     return None

    def log_sqrt_trends(self, folder=None):

        # create a dataframe of results
        log = []
        for iteration_log in self.log:
            log.append({'T': iteration_log[-1]['step'],
                        'regret': iteration_log[-1]['cum_regret'],
                        'violation': iteration_log[-1]['cum_violation']})

        df = pd.DataFrame(log)
        df.to_csv(folder + 'sqrt_trends_log.csv')

        fig, axes = plt.subplots(nrows=1, ncols=2)
        fig.set_size_inches(12, 8)

        axes[0].scatter(df['T'], df['regret'])
        axes[0].set_xlabel('T')
        axes[0].set_ylabel('Average Normalized Regret')

        axes[1].scatter(df['T'], df['violation'])
        axes[1].set_xlabel('T')
        axes[1].set_ylabel('Average Normalized Capacity Violation')

        plt.tight_layout()
        plt.savefig(folder + 'sqrt_trends.png')
        plt.close()

        return None
