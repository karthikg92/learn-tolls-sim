from network import Network
from users import Users
from optimizer import Optimizer
from simulation import Simulation
from datetime import datetime
import os


class Problem:
    """
    Instantiates a problem
    Sets up the network and associated simulations
    Logs the results
    Possible modes of operation:
        name=sqrt_trends, params=   {
                                    'cities':[city1, city2],
                                    'alg':['gr_desc', 'feasible_desc'],
                                    'stationary':True
                                    }
        name=iteration_dynamics, params=    {
                                            'cities':[city1, city2],
                                            'alg':['gr_desc', 'feasible_desc'],
                                            'stationary':True,
                                            step_policy:['constant', 'sqrt', 'reciprocal', 'ramp']
                                            }
        name=baseline, params={}
        name=why_vot, params={'cities':[city_1, city2, ...]}
    """

    # baseline is a separate thing; no need to iterate at all

    def __init__(self, name=None, params=None):

        if name is None:
            print("[Problem] Problem not defined. Terminating.")

        elif name == 'sqrt_trends':
            """
                Plot the square root T theoretical trends for each city and algorithm
            """
            path_time = self.datetime_path()
            try:
                root_folder_path = 'ResultLogs/' + name + '_' + path_time + '/'
                os.mkdir(root_folder_path)
            finally:
                pass

            for city in params['cities']:

                print('[Problem] Solving sqrt_trends problem for ', city)

                for alg in params['alg']:

                    print('[Problem] Using algorithm ', alg)

                    n = Network(city)
                    users = Users(city)
                    opt = Optimizer(n, users)
                    sim = Simulation()

                    # for T in [5, 10, 20, 100, 200, 500, 1000, 5000, 10000, 20000]:
                    for T in [10, 20, 50, 100, 250, 500]:
                        print("[Problem] Solving for T = ", T)
                        sim.run(n, users, opt, max_steps=T, step_policy='constant', alg=alg,
                                is_stationary=False)
                        sim.log_sqrt_trends(folder=root_folder_path+city+'_'+alg)

        elif name == 'iteration_dynamics':
            pass

        elif name == 'why_vot':
            """
            Compare total travel times when tolls are set with and without vot consideration
            """
            time_steps = 100  # total time for the simulation

            no_vot_toll = None
            vot_toll = None

            ttt_no_vot = []
            ttt_vot = []

            for t in range(time_steps):
                users.vot_realization()

                # tolls without VOT consideration
                x, f = response(users, no_toll)
                ttt.no_toll.append(compute_ttt(x))

                # tolls with VOT consideration
                x, f = response(users, no_toll)
                ttt.no_toll.append(compute_ttt(x))

        else:
            print("[Problem] Problem not defined. Terminating.")

    @staticmethod
    def datetime_path():
        now = datetime.now()
        path_string = now.strftime("%m-%d-%H-%M-%S")
        return path_string
