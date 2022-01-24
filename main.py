import cProfile
import pstats
from problem import Problem
from justify_vot import JustifyVOT


# cp = cProfile.Profile()
# cp.enable()

#Problem(name='sqrt_trends', params={'cities': ['SiouxFalls'], 'alg': ['gr_desc'], 'stationary': False})

JustifyVOT()

# cp.disable()
# stats = pstats.Stats(cp).sort_stats('cumtime')
# stats.print_stats()
