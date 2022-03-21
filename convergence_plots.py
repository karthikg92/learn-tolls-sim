import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

"""
Plot for sqrt convergence under point-mass OD pair distribution
"""

path = 'ResultLogs/ConvergenceAnalysis/TravelTimeExperiments_03-18-18-57-15/'
result_folder = 'ResultLogs/ConvergenceAnalysis/'
df = pd.read_csv(path + 'comparison.csv')

x = np.array(df['T'])
y1 = df['regret_gr_desc_not_normalized'].to_list()
y2 = df['regret_gr_desc'].to_list()

# dealing with coding bug
y = []
for i in range(len(y1)):
    y += [y1[i] * y2[i] / (1 + y2[i])]
y = np.array(y)


plt.plot(x, y)
plt.savefig(result_folder + 'regret.png')
plt.close()

x = np.log(np.array(df['T']))
y = np.log(np.array(df['vio_gr_desc_not_normalized']))
plt.plot(x, y, '-*')
plt.xlabel('log($T$)')
plt.ylabel('log($V_T$)')

lin_fit = linregress(x[2:], y[2:])
plt.text(2, 11.5, 'R-square = %.3f' % (lin_fit.rvalue)**2)
plt.text(2, 11.3, 'Slope = %.3f' % (lin_fit.slope))

# plt.xscale("log")
# plt.yscale("log")
plt.savefig(result_folder + 'cap_vio.png')
plt.close()
