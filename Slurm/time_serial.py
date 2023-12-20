#!/usr/bin/env python3
#SBATCH --job-name=bertha121_getEPs
#SBATCH --time=00:05:00   # walltime
##SBATCH --output=output.%A.%a.out
##SBATCH --error=error.%A.%a.err
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024M
#SBATCH --tasks-per-node=1
#SBATCH --qos=normal

#import ep_utils
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

if __name__ == "__main__":
    # get graph we are doing analysis on
    graph_type = None
    graph_path = sys.argv[1]
    data_fn = graph_path.split('/')[-1].split('.')[0]
    os.environ['GRAPH_PATH'] = graph_path
    try: 
        try: G = nx.read_graphml(graph_path); graph_type = 'graphml'
        except: G = sp.load_npz(graph_path); graph_type = 'sparse'; 
    except: print("You need to give the file path to graph you want to run")
    if graph_type == 'graphml':
        t = tim.time_this(nx.adjacency_spectrum,[G],
                            label="get_eigvals",store_in='./' + data_fn + '_serial.txt')
    if graph_type == 'sprase':
        t = tim.time_this(sp.linalg.eigs,[G,g.shape[0]-2],
                            label="get_eigvals",store_in='./' + data_fn + '_serial.txt')
    print("Finished!")


