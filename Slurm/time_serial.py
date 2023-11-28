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
import os, sys, json
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
import numpy as np
import timing as tim
import ep_utils
import subprocess
from time import perf_counter as pc

if __name__ == "__main__":
    # get graph we are doing analysis on
    graph_path = sys.argv[1]
    data_fn = graph_path.split('/')[-1].split('.')[0]
    os.environ['GRAPH_PATH'] = graph_path
    try: G = nx.read_graphml(graph_path)
    except: print("You need to give the file path to graph you want to run")
    # run how long it takes to get equitable partitions
    out,t = tim.time_this(ep_utils.getEquitablePartitions,[G],ret_output=True,
                            label="get_eps",store_in='./' + data_fn + '_parallel.txt')
    ep,lep_list = out
    # get the eps and leps from that operation
    os.environ['EP'] = tim.serialize(ep)
    os.environ['LEPs'] = tim.serialize({i:list(lep) for i,lep in enumerate(lep_list)})
    num_nodes = min(30,int(np.sqrt(len(lep_list))))
    print(str(num_nodes))
    os.environ['NUM_NODES'] = str(num_nodes)

    # divy up the leps to each of the nodes in a second script
    slurm_command = [
        'sbatch',
        '--nodes=1',
        '--ntasks-per-node=1',
        '--time=00:05:00',
        '--mem-per-cpu=1024M',
        '--job-name=' + 'bertha121_LEPs',
        '--array=1-' + f'{num_nodes}',
        '--qos=normal',
        '_LEP_slurm.py'
    ]   
    subprocess.run(slurm_command)


