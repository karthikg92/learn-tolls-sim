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
plt.xlabel('T')
plt.ylabel('Regret')
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

plt.savefig(result_folder + 'cap_vio.png')
plt.close()


""" 
##########
Plot for IID OD pairs
#########
"""

path = 'ResultLogs/ConvergenceAnalysis/IidOdExpts_03-21-15-17-39/'
result_folder = 'ResultLogs/ConvergenceAnalysis/'
df = pd.read_csv(path + 'comparison.csv')


# Capacity violation plots
plt.rcParams['font.size'] = '24'

plt.figure(figsize=(8, 6), dpi=250)

x = np.log(np.array(df['T']))
y = np.log(np.array(df['vio_gr_desc_not_normalized']))

# If you want a linear fit
# lin_fit = linregress(x[:], y[:])
# y_lin = lin_fit.slope * x[:] + lin_fit.intercept
# plt.plot(x[:], y_lin, '--', linewidth=3, label='Linear fit')

# If you want the best fit with slope 0.5, do this instead:
bias_vec = y[:] - 0.5 * x[:]
bias = np.mean(bias_vec)
y_lin = 0.5 * x[:] + bias
plt.plot(x[:], y_lin, '--', linewidth=5, color='C1', label='Theoretical bound')


# plt.text(3.5, 9.4, 'R-square = %.3f' % (lin_fit.rvalue**2))
# plt.text(3.5, 9.1, 'Slope = %.3f' % lin_fit.slope)
rmse = np.sqrt(np.mean((y - y_lin)**2))
# plt.text(3.5, 9.1, 'RMSE = %.3f' % rmse)

plt.plot(x, y, 's', linewidth=3, markersize=10, markeredgecolor='black', markerfacecolor='black', label='Algorithm 1')
plt.xlabel('log(Time Periods)')
plt.ylabel('log(Capacity Violation)')


plt.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(result_folder + 'iid_cap_vio.png')
plt.close()

# regret plot
x = np.array(df['T'])
y = np.array(df['regret_gr_desc_not_normalized'])

plt.figure(figsize=(8, 6.3), dpi=250)
plt.plot(x, y, 's', linewidth=3, markersize=10, markeredgecolor='black', markerfacecolor='black', label='Algorithm 1')
plt.xlabel('Time Periods')
plt.ylabel('Regret')

# plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(result_folder + 'iid_regret.png')
plt.close()

