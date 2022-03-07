import random

import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import tikzplotlib
import numpy as np
from utils import *

import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import contextily as cx


path = 'ResultLogs/TravelTimeExperiments/'

"""
####################
Figure 1
####################
"""

df = pd.read_csv(path + 'comparison.csv')

plt.rcParams['font.size'] = '14'
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 6)

axes[0].plot(df['T'], df['regret_gr_desc'], '*-', c='tab:blue', label='gradient descent')
axes[0].plot(df['T'], df['regret_stochastic'], '*-', c='tab:orange', label='Group VoT')
axes[0].plot(df['T'], df['regret_no_vot'], '*-', c='tab:green', label='Population VoT')
axes[0].plot(df['T'], df['regret_const_update'], '*-', c='tab:red', label='Reactive update')
axes[0].set_xlabel('Number of Time Periods')
axes[0].set_ylabel('Average Normalized Regret')
axes[0].legend(loc="lower right")

axes[1].plot(df['T'], df['vio_gr_desc'], '*-', c='tab:blue', label='gradient descent')
axes[1].plot(df['T'], df['vio_stochastic'], '*-', c='tab:orange', label='Group VoT')
axes[1].plot(df['T'], df['vio_no_vot'], '*-', c='tab:green', label='Population VoT')
axes[1].plot(df['T'], df['vio_const_update'], '*-', c='tab:red', label='Reactive update')
axes[1].set_xlabel('Number of Time Periods')
axes[1].set_ylabel('Average Normalized Capacity Violation')
axes[1].legend(loc="upper right")


plt.tight_layout()
plt.savefig(path + '/figures/sqrtTconvergence.png', dpi=250)

# tikzplotlib.clean_figure()
# tikzplotlib.save(path + '/figures/fig1.tex')

plt.close()


"""
####################
Figure 2
####################
"""

df = pd.read_csv(path + 'comparison.csv')

plt.rcParams['font.size'] = '14'

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 6)

axes.plot(df['T'], 1 + df['ttt_gr_desc'], '*-', c='tab:blue', label='gradient descent')
axes.plot(df['T'], 1 + df['ttt_stochastic'], '*-', c='tab:orange', label='Group VoT')
axes.plot(df['T'], 1 + df['ttt_no_vot'], '*-', c='tab:green', label='Population VoT')
axes.plot(df['T'], 1 + df['ttt_const_update'], '*-', c='tab:red', label='Reactive update')
axes.set_xlabel('Number of Time Periods')
axes.set_ylabel('$\\dfrac{ \\mathrm{Total \;\, Travel \;\, Time}}{ \\mathrm{Optimal \;\, Total \;\, Travel \;\, '
                'Time}}$')
axes.legend(loc="lower right")

plt.tight_layout()
plt.savefig(path + '/figures/totaltraveltime.png', dpi=250)

# tikzplotlib.clean_figure()
# tikzplotlib.save(path + '/figures/fig2.tex')

plt.close()

"""
####################
Figure 3
####################
"""


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_edge_values(value, figpath, truncate_flag=True, plot_lim=None, label=None, text=None):

    # Load vertices
    df_vertices = pd.read_csv('Locations/SiouxFalls/vertices.csv')

    # Load edges
    df_edges = pd.read_csv('Locations/SiouxFalls/edges.csv')

    # Initialize figure
    plt.rcParams['font.size'] = '16'
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 10)

    # Create Nodes for Plotting
    gdf_vertices = gpd.GeoDataFrame(
        df_vertices, geometry=gpd.points_from_xy(df_vertices.xcoord, df_vertices.ycoord), crs="EPSG:4326")
    gdf_vertices = gdf_vertices.to_crs('EPSG:3857')

    # Plot Nodes
    ax = gdf_vertices.plot(color='red', ax=ax)

    # Add basemap
    cx.add_basemap(ax, alpha=0.5)

    # Create edges
    gdf_vertices['lat'] = gdf_vertices.geometry.y
    gdf_vertices['lon'] = gdf_vertices.geometry.x

    merge_df = gdf_vertices.filter(['vert_id', 'lat', 'lon'], axis=1)
    merge_df = merge_df.rename(columns={'lat': 'tail_lat', 'lon': 'tail_lon'})

    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_tail', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    merge_df = merge_df.rename(columns={'tail_lat': 'head_lat', 'tail_lon': 'head_lon'})
    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_head', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    # set colormap
    cm = plt.get_cmap('Oranges')
    if truncate_flag:
        cm = truncate_colormap(cm, minval=0.3, maxval=1, n=100)
    if plot_lim is not None:
        c_norm = colors.Normalize(vmin=plot_lim[0], vmax=plot_lim[1])
    else:
        c_norm = colors.Normalize(vmin=min(value), vmax=max(value))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    df_edges = df_edges.reset_index()  # make sure indexes pair with number of rows

    # print(df_edges)

    for index, row in df_edges.iterrows():
        colorval = scalar_map.to_rgba(value[index])
        a = patches.FancyArrowPatch((row['tail_lon'], row['tail_lat']),
                                    (row['head_lon'], row['head_lat']),
                                    connectionstyle="arc3,rad=.1",
                                    arrowstyle="Fancy, head_length=7, head_width=5",
                                    color=colorval)
        # Adding each edge
        plt.gca().add_patch(a)

    # Adding colorbar
    plt.colorbar(scalar_map, ax=ax, label=label)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if text is not None:
        x_pos = -1.0775e7
        y_pos = 5.405e6
        for line in text:
            # print('adding text')
            plt.text(x_pos, y_pos, line, weight="bold")
            y_pos -= 1e3

    fig.savefig(figpath, dpi=250)

    # Not effective
    # tikzplotlib.save(figpath.split('.')[0] + '.tex')

    plt.close()


# load data
df_capacity = pd.read_csv(path + 'capacity.csv', header=None)
df_latency = pd.read_csv(path + 'latency.csv', header=None)
capacity = (df_capacity[0]/2.4).to_list()  # for a per hour number
latency = (df_latency[0]*60).to_list()  # for units in minutes


# make plots
plot_edge_values(capacity, path + '/figures/capacity.png', truncate_flag=True,
                 label='Capacity (vehicles per hour)')
plot_edge_values(latency, path + '/figures/latency.png', truncate_flag=True,
                 label='Travel time (minutes)')


"""
####################
Figure 4
####################
"""

'''

Need to plot the following on the map:

tolls_gr_desc_t_1000.csv
tolls_const_update_t_1000.csv
group_specific_VOT_toll.csv
population_mean_toll.csv

'''

# load data
gr_desc_toll = pd.read_csv(path + 'tolls_gr_desc_t_1000.csv', header=None)
gr_desc_toll = gr_desc_toll[0].to_list()
# gr_desc_toll = [toll if toll > 0 else 0 for toll in gr_desc_toll]

const_update_toll = pd.read_csv(path + 'tolls_const_update_t_1000.csv', header=None)
const_update_toll = const_update_toll[0].to_list()
# const_update_toll = [toll if toll > 0 else 0 for toll in const_update_toll]

group_specific_toll = pd.read_csv(path + 'group_specific_VOT_toll.csv', header=None)
group_specific_toll = group_specific_toll[0].to_list()
# group_specific_toll = [toll if toll > 0 else 0 for toll in group_specific_toll]

population_mean_toll = pd.read_csv(path + 'population_mean_toll.csv', header=None)
population_mean_toll = population_mean_toll[0].to_list()
# population_mean_toll = [toll if toll > 0 else 0 for toll in population_mean_toll]


plot_edge_values(gr_desc_toll, path + '/figures/toll_gr_desc.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.77',
                       'Max. Toll: 2.54'])

plot_edge_values(const_update_toll, path + '/figures/toll_reactive_update.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.73',
                       'Max. Toll: 2.40'])

plot_edge_values(group_specific_toll, path + '/figures/toll_group_vot.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.80',
                       'Max. Toll: 2.63'])

plot_edge_values(population_mean_toll, path + '/figures/toll_pop_vot.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.45',
                       'Max. Toll: 2.71'])

"""
Push relevant statistics from the map into a table
"""
table_path = path + '/figures/stats_table.csv'
try:
    os.remove(table_path)
except:
    pass

write_row(table_path, ['Algorithm', 'MinToll', 'MaxToll', 'NumOfEdgesWithTolls', 'AverageToll'])

non_zero_tolls = [toll for toll in gr_desc_toll if toll > 1e-2 ]
write_row(table_path, ['Gradient descent',
                       min(gr_desc_toll),
                       max(gr_desc_toll),
                       sum(np.array(gr_desc_toll) > 1e-2),
                       np.mean(non_zero_tolls)])


non_zero_tolls = [toll for toll in const_update_toll if toll > 1e-2 ]
write_row(table_path, ['Reactive update',
                       min(const_update_toll),
                       max(const_update_toll),
                       sum(np.array(const_update_toll) > 1e-2),
                       np.mean(non_zero_tolls)])

non_zero_tolls = [toll for toll in group_specific_toll if toll > 1e-2]
write_row(table_path, ['Group Specific VoT',
                       min(group_specific_toll),
                       max(group_specific_toll),
                       sum(np.array(group_specific_toll) > 1e-2),
                       np.mean(non_zero_tolls)])

non_zero_tolls = [toll for toll in population_mean_toll if toll > 1e-2]
write_row(table_path, ['Population VoT',
                       min(population_mean_toll),
                       max(population_mean_toll),
                       sum(np.array(population_mean_toll) > 1e-2),
                       np.mean(population_mean_toll)])

"""
####################
Figure 5
####################
"""
gr_desc_toll = pd.read_csv(path + 'tolls_gr_desc_t_1000.csv', header=None)
gr_desc_toll = gr_desc_toll[0].to_list()

nbin = 20
count, bins = np.histogram(gr_desc_toll, bins=nbin)

xloc = [(bins[i] + bins[i+1])/2 for i in range(len(bins) -1)]

plt.bar(xloc, count/sum(count)*100, width=bins[1]-bins[0])
plt.xlabel('Tolls (dollars)')
plt.ylabel('Percentage of edges')

plt.tight_layout()
plt.savefig(path + '/figures/tolls_histogram.png')
plt.close()



# count, bins_count = np.histogram(c, bins=500)
#
# plt.hist(c, bins=50, color='tab:grey')
# # pdf = count / sum(count)
# # cdf = np.cumsum(pdf)
# # plt.plot(bins_count[1:], cdf, color="black")


"""
####################
Figure 6
####################
"""
# t100_path = path + 'T_100_log.csv'
t1000_path = path + 'T_1000_log.csv'
#
# df = pd.read_csv(t100_path)
#
# plt.rcParams['font.size'] = '14'
# plt.plot(df['t'], df[' total_const_update'], label='Reactive update', alpha=0.7)
# plt.plot(df['t'], df[' total_gr_desc'], label='gradient descent', alpha=0.7)
# plt.legend(loc='lower right')
# plt.xlabel('Time steps')
# plt.ylabel('Total tolls (dollars)')
#
# plt.savefig(path + '/figures/Fig6_t100.png')
# plt.close()


df = pd.read_csv(t1000_path)

plt.rcParams['font.size'] = '14'
plt.plot(df['t'], df[' total_const_update'], label='Reactive update', alpha=0.7)
plt.plot(df['t'], df[' total_gr_desc'], label='gradient descent', alpha=0.7)
plt.legend(loc='lower right')
plt.xlabel('Number of Time Periods')
plt.ylabel('Total tolls (dollars)')

plt.savefig(path + '/figures/toll_convergence.png')
