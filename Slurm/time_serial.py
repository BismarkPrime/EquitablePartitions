#!/usr/bin/env python3
#SBATCH --job-name=bn-human-BNUserial
#SBATCH --time=3-00:00:00   # walltime
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024G
#SBATCH --tasks-per-node=1
#SBATCH --qos=normal


import os, sys, json, io
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
import numpy as np
import timing as tim
import ep_utils
import subprocess
from time import perf_counter as pc
import scipy.sparse as sp

CUST_COM="#"
CUST_DEL=" "
# this is currently not being changed, maybe in the future when our algorithm can handle weighted graphs.
WEIGHTED=False

if __name__ == "__main__":
    # get graph we are doing analysis on and parameters relating to that
    real = 'real' in sys.argv
    directed = True if 'True' in sys.argv else False
    graph_type = None
    graph_path = sys.argv[1]
    print(f'\n\n\n{graph_path}\n\n\n')
    data_fn = graph_path.split('/')[-1].split('.')[0]
    os.environ['GRAPH_PATH'] = graph_path

    # try loading the graph in
    try: 
        # commented code on trial. I don't think we need it now that we have the oneGraphtoRuleThemAll function
        # if real:
        G = graphs.oneGraphToRuleThemAll(graph_path, suppress=True,directed=directed, cust_del=CUST_DEL,cust_com=CUST_COM, weighted=WEIGHTED); graph_type = 'sparse'
        # else:
        #     try: G = nx.read_graphml(graph_path); graph_type = 'graphml'
        #     except: G = sp.load_npz(graph_path); graph_type = 'sparse'; 
    except Exception as e:
        print(f"Error occured: {e}")

    # run the timing
    # if graph_type == 'graphml':
    #     G_size = G.number_of_nodes()
    #     t = tim.time_this(nx.adjacency_spectrum,[G],
    #                         label="get_eigvals",store_in='./' + data_fn + '_serial.txt')
    if graph_type == 'sparse':
        G_size = G.shape[0]
        t = tim.time_this(sp.linalg.eigs,[G,G_size-2],
                            label="get_eigvals",store_in='./' + data_fn + '_serial.txt')

    with open('./' + data_fn + '_serial.txt','a') as f:
        f.write(f"Size: {G_size}")
    print("Finished!")


