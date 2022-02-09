import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import pandas as pd
import geopandas as gpd
import contextily as cx


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_edge_values(value, figpath, figname, truncate_flag=True, plot_lim=None):

    # Load vertices
    df_vertices = pd.read_csv('Locations/SiouxFalls/vertices.csv')

    # Load edges
    df_edges = pd.read_csv('Locations/SiouxFalls/edges.csv')

    # Initialize figure
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
    plt.colorbar(scalar_map, ax=ax)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.title(figname)

    fig.savefig(figpath, dpi=250)
    plt.close()

