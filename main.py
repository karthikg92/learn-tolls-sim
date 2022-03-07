import cProfile
import pstats
from problem import Problem
from justify_vot import JustifyVOT
from sqrt_trends import SqrtTrends
from sanity_check import StochasticProgramWithConstantVOT
from sanity_check import TestOutsideOption
from travel_time_expts import TravelTimeExperiments
from network_plots import NetworkPlots
from sanity_check import CheckIfNoiseHelpsStochasticProgram
from sanity_check import ComputeOptimalTTT

# cp = cProfile.Profile()
# cp.enable()

'''
Experiments or algorithms we want to run
'''
# stochastic_program_check = StochasticProgramWithConstantVOT()
# SqrtTrends()
# JustifyVOT()
NetworkPlots()
TravelTimeExperiments()
# TestOutsideOption()
# StochasticProgramWithConstantVOT()
# CheckIfNoiseHelpsStochasticProgram()
# ComputeOptimalTTT()


# cp.disable()
# stats = pstats.Stats(cp).sort_stats('cumtime')
# stats.print_stats()



