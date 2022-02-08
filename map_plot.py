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


df_vertices = pd.read_csv('Locations/SiouxFalls/vertices.csv')

gdf_vertices = gpd.GeoDataFrame(
    df_vertices, geometry=gpd.points_from_xy(df_vertices.xcoord, df_vertices.ycoord), crs="EPSG:4326")
gdf_vertices = gdf_vertices.to_crs('EPSG:3857')

# plot nodes
ax = gdf_vertices.plot(color='red')

# introduce basemap
cx.add_basemap(ax, alpha=0.5)

# read new edge matrix
df_edges = pd.read_csv('Locations/SiouxFalls/edges.csv')

# augment with coordinates
gdf_vertices['lat'] = gdf_vertices.geometry.y
gdf_vertices['lon'] = gdf_vertices.geometry.x

merge_df = gdf_vertices.filter(['vert_id', 'lat', 'lon'], axis=1)
merge_df = merge_df.rename(columns={'lat': 'tail_lat', 'lon': 'tail_lon'})

df_edges = df_edges.merge(merge_df, how='left', left_on='edge_tail', right_on='vert_id')
df_edges = df_edges.drop(columns=['vert_id'])

merge_df = merge_df.rename(columns={'tail_lat': 'head_lat', 'tail_lon': 'head_lon'})
df_edges = df_edges.merge(merge_df, how='left', left_on='edge_head', right_on='vert_id')
df_edges = df_edges.drop(columns=['vert_id'])

# plot edges

# set colormap
cm = plt.get_cmap('Oranges')
cm = truncate_colormap(cm, minval=0.3, maxval=1, n=100)

cNorm = colors.Normalize(vmin=0, vmax=25900)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

df_edges = df_edges.reset_index()  # make sure indexes pair with number of rows

patch_collection=[]
for index, row in df_edges.iterrows():
    colorval = scalarMap.to_rgba(row['capacity'])
    a = patches.FancyArrowPatch((row['tail_lon'], row['tail_lat']),
                                (row['head_lon'], row['head_lat']),
                                connectionstyle="arc3,rad=.1",
                                arrowstyle="Fancy, head_length=7, head_width=5",
                                color=colorval)

    plt.gca().add_patch(a)

    patch_collection.append(a)


plt.colorbar(scalarMap, ax=ax)

# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
