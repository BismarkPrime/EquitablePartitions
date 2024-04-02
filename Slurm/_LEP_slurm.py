#!/usr/bin/env python3
#SBATCH --job-name=bertha_262144_LEPs
#SBATCH --time=3-00:00:00   # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=256G
#SBATCH --qos=normal
#SBATCH --array=1-30


import os, sys, json
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
from time import perf_counter as pc
import timing as tim
import ep_utils
import scipy.sparse as sp

if __name__ == "__main__":
    start = pc()
    graph_path = os.environ.get('GRAPH_PATH')
    try:
        G = nx.read_graphml(graph_path)
    except: 
        G = sp.load_npz(graph_path)
    #nodes, edges = os.environ.get('GRAPH_NODES'), os.environ.get('GRAPH_EDGES')
    #G = nx.Graph()
    #G.add_nodes_from(nodes)
    #G.add_edges_from(edges)
    #G = nx.parse_edgelist(serialized_graph,nodetype=int)
    #print(type(G))
    end = pc()
    data_fn = graph_path.split('/')[-1].split('.')[0]
    build_time = end-start
    # get environment variables
    start = pc()
    with open("EpLep.txt","r") as f:
        data = f.read()
    data = data.split('==')
    ep = data[1]    
    #ep = os.environ.get('EP') # failed after 8100 with this method trying read/writing files
    ep = tim.deserialize(ep)
    ep = {int(key):val for key,val in ep.items()}
    #lep_list = os.environ.get('LEPs')
    lep_list = data[3]
    lep_list = tim.deserialize(lep_list)
    lep_list = [set(lep) for lep in lep_list.values()]
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_nodes = int(os.environ.get("NUM_NODES"))
    leps_prepped, _ = tim.prep_scatter(lep_list,num_nodes)
    end = pc()
    # write the extra time in the txt file
    with open(data_fn + '_parallel.txt','a') as f:
        f.write(f"build_graph{task_id}:" + str(build_time) + '\n')
        f.write(f"communication{task_id}:" + str(end-start) + '\n') # the colon will help recognize where the parallelization starts
    part_spec,t = tim.time_this(ep_utils.GetSpectrumFromLEPs,[G,[ep,leps_prepped[int(task_id)-1]]],
                                ret_output=True,store_in='./' + data_fn + '_parallel.txt',label=f"lep_time{task_id}")
    json.dump(str(part_spec) + '\n',open(data_fn + '_spec.txt','a'))
    print("Finished!")
    