#!/usr/bin/env python3
#SBATCH --job-name=bio-celegans-udserial
#SBATCH --time=00:10:00   # walltime
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --tasks-per-node=1
#SBATCH --qos=test


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
    # get graph we are doing analysis on and parameters relating to that
    real = 'real' in sys.argv
    directed = True if 'True' in sys.argv else False
    graph_type = None
    graph_path = sys.argv[1]
    print(f'\n\n\n{graph_path}\n\n\n')
    data_fn = graph_path.split('/')[-1].split('.')[0]
    os.environ['GRAPH_PATH'] = graph_path

    print("DATA FILE LOOK LIKE THIS:")
    output = os.popen(f"head {file_name}").read()
    print(output)
    custom = h.parse_input("Custom Delimiter/comment symbol? (yes/no): ")
    if custom:
        cust_del = input("\tWhat is the delimiter: ")
        cust_com = input("\tWhat is the comment symbol: ")
    else:
        cust_del = None
        cust_com = "#"

    # try loading the graph in
    try: 
        if real:
            G = graphs.oneGraphToRuleThemAll(graph_path, suppress=True,cust_del=cust_del,cust_com=cust_com); graph_type = 'sparse'
        else:
            try: G = nx.read_graphml(graph_path); graph_type = 'graphml'
            except: G = sp.load_npz(graph_path); graph_type = 'sparse'; 
    except Exception as e:
        print(f"Error occured: {e}")

    # run the timing
    if graph_type == 'graphml':
        t = tim.time_this(nx.adjacency_spectrum,[G],
                            label="get_eigvals",store_in='./' + data_fn + '_serial.txt')
    if graph_type == 'sparse':
        t = tim.time_this(sp.linalg.eigs,[G,G.shape[0]-2],
                            label="get_eigvals",store_in='./' + data_fn + '_serial.txt')
    print("Finished!")


