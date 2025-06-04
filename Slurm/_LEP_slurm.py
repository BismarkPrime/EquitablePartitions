#!/usr/bin/env python3
#SBATCH --job-name=bn-human-BNU_1_0025890_session_1_LEPs
#SBATCH --time=3-00:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024G
#SBATCH --qos=normal
#SBATCH --array=1-30


import os, sys, json
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
from time import perf_counter as pc
import timing as tim
import ep_utils
import numpy as np
import scipy.sparse as sp

# NOTE: could speed up by finding a way to not calculate and scater the singletons.

if __name__ == "__main__":
    start = pc()
    graph_path = os.environ.get('GRAPH_PATH')
    CUST_COM = os.environ.get('CUST_COM')
    CUST_DEL = os.environ.get('CUST_DEL')
    WEIGHTED = bool(os.environ.get('WEIGHTED'))
    try:
        G = graphs.oneGraphToRuleThemAll(graph_path, suppress=True,cust_del=CUST_DEL,cust_com=CUST_COM, weighted=WEIGHTED)
        G_csc = G.tocsc()
        G_csr = G.tocsr()
    except Exception as e:
        print(f"Error occured in _Lep_slurm.py. Error was: {e}") 
    end = pc()
    data_fn = graph_path.split('/')[-1].split('.')[0]
    build_time = end-start

    # get environment variables
    start = pc()

    with open("EpLep.txt","r") as f:
        data = f.read()
    data = data.split('==')
    ep = data[1]    
    ep = tim.deserialize(ep)
    ep = {int(key):val for key,val in ep.items()}
    lep_list = data[3]
    lep_list = tim.deserialize(lep_list)
    lep_list = [[int(l) for l in lep] for lep in lep_list.values()]
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_nodes = int(os.environ.get("NUM_NODES"))
    leps_prepped, _ = tim.prep_scatter(lep_list,num_nodes)

    end = pc()

    # write the extra time in the txt file
    with open(data_fn + '_parallel.txt','a') as f:
        f.write(f"build_graph{task_id}:" + str(build_time) + '\n')
        f.write(f"communication{task_id}:" + str(end-start) + '\n') # the colon will help recognize where the parallelization starts

    # actually run the algorithm for this node process
    include_globals = task_id == num_nodes
    part_spec,t = tim.time_this(ep_utils._getEigenvaluesSparseFromPartialLeps,[G_csc,G_csr,ep,leps_prepped[int(task_id)-1],include_globals],
                            ret_output=True,store_in='./' + data_fn + '_parallel.txt',label=f"lep_time{task_id}")
    globals, locals = part_spec

    # TODO: make this so it'ss easier to work with post the fact.
    if include_globals: 
        with open(data_fn + '_spec_total.txt', 'a') as f:
            f.write(str(len(globals)) + '\n')
        # writing not json is giving errors
        json.dump(str(globals) + '\n',open(data_fn + '_spec.txt','a'))
    else: 
        with open(data_fn + '_spec_total.txt', 'a') as f:
            f.write(str(len(locals)) + '\n')
        json.dump(str(locals) + '\n',open(data_fn + '_spec.txt','a'))
        
    print("Finished!")
    