import cProfile
import pstats
from problem import Problem

# cp = cProfile.Profile()
# cp.enable()

#Problem(name='sqrt_trends', params={'cities': ['SiouxFalls'], 'alg': ['gr_desc'], 'stationary': False})

Problem(name='why_vot', params={'cities': ['SiouxFalls']})

# cp.disable()
# stats = pstats.Stats(cp).sort_stats('cumtime')
# stats.print_stats()
