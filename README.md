# Online Learning for Traffic Routing under Unknown Preferences #

This repository contains the official implementation for the paper [Online Learning for Traffic Routing under Unknown Preferences](https://arxiv.org/abs/2203.17150) by [Devansh Jalota](https://stanfordasl.github.io//people/devansh-jalota/), [Karthik Gopalakrishhah](https://karthikg92.github.io), [Navid Azizan](https://azizan.mit.edu), [Ramesh Johari](https://web.stanford.edu/~rjohari/), and [Marco Pavone](https://profiles.stanford.edu/marco-pavone), published in [AISTATS'23](http://aistats.org/aistats2023/accepted.html).



## Data sources ##
The traffic network, user flows, and road capacities are obtained from the [TNTP dataset](https://github.com/bstabler/TransportationNetworks)

## Requirements ##

This code uses the following packages
- [Gurobi](https://www.gurobi.com/products/gurobi-optimizer/) for the optimization solvers
- [Geopandas](https://geopandas.org/en/stable/) for manipulating geospatial data
- [Contextily](https://contextily.readthedocs.io/en/latest/) for loading basemap for plots

## Running the code ##

- `main.py` to run simulations
- `plots.py` to generate the plots 
