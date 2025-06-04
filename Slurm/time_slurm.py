#!/usr/bin/env python3
#SBATCH --job-name=bn-human-BNUslurm
#SBATCH --time=3-00:00:00   # walltime
##SBATCH --ntasks-per-node=1
##SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024G
#SBATCH --tasks-per-node=1
#SBATCH --qos=normal


import os, sys, json
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
import numpy as np
import timing as tim
# import ep_utils"""
import subprocess
import ep_finder, lep_finder
from time import perf_counter as pc
import scipy.sparse as sp
import slurm_helper as h

CUST_COM="#"
CUST_DEL=" "
# this is currently not being changed, maybe in the future when our algorithm can handle weighted graphs.
WEIGHTED=False

if __name__ == "__main__":
    start_script = '/home/jrhmc1/Desktop/EquitablePartitions/Slurm/_LEP_slurm.py'
    # get graph we are doing analysis on and relevant information for later.
    real = 'real' in sys.argv
    directed = True if 'True' in sys.argv else False
    print(f"\n\n\nThis is a check to see if the directed argument is working right. directed is: {directed}\n\n\n")
    graph_path = sys.argv[1]
    data_fn = graph_path.split('/')[-1].split('.')[0]

    # environment variables
    os.environ['GRAPH_PATH'] = graph_path
    os.environ['CUST_COM'] = "#"
    os.environ['CUST_DEL'] = " "
    os.environ['WEIGHTED'] = str(WEIGHTED)

    # try to load in graph
    try: 
        G = graphs.oneGraphToRuleThemAll(graph_path, suppress=True,cust_del=CUST_DEL,cust_com=CUST_COM, weighted=WEIGHTED)
    except Exception as e: 
        print(f"Error occured: {e}")
    # out,t = tim.time_this(ep_utils.getEquitablePartitions,[G],ret_output=True,
    #                         label="get_eps",store_in='./' + data_fn + '_parallel.txt')"""
    ep,t = tim.time_this(ep_finder.getEquitablePartition,[ep_finder.initFromSparse(G)],ret_output=True,
                            label="get_eps",store_in='./' + data_fn + '_parallel.txt')
    lep_list,t = tim.time_this(lep_finder.getLocalEquitablePartitions,[lep_finder.initFromSparse(G),ep],ret_output=True,
                            label="get_leps",store_in='./' + data_fn + '_parallel.txt')

    print(lep_list)
    # get the eps and leps from that operation
    with open("EpLep.txt","w") as f:
        f.write("EPs==" + tim.serialize(ep))
        f.write("==LEPs==" + tim.serialize({i:list(lep) for i,lep in enumerate(lep_list)}))
        
    # Get meta data and store in in a meta.txt file
    with open("Meta.txt","w") as f:
        f.write(f"Graph size: {G.shape[0]}\n")
        f.write(f"Nontrivial ep perc: {sum([len(l) for l in ep.values() if len(l) > 1]) / G.shape[0]}\n")
        f.write(f"Nontivial lep perc: {sum([len(l) for l in lep_list if len(l) > 1]) / max(ep.keys())}")

    total_tasks = int(np.sqrt(len(lep_list)))
    num_nodes = min(30,int(np.sqrt(len(lep_list))))
    print(f"number of nodes to be used calculated as: {total_tasks}\nsubmitting with: {num_nodes}")
    os.environ['NUM_NODES'] = str(num_nodes)

    # get the slurm parameters
    param_dict = tim.CheckSlurmParameters(skip_check=True)
    node_val,ntasks_val,time_val,mem_val,qos_val = [pair[1] for pair in param_dict.values()]

    # divy up the leps to each of the nodes in a second script
    # get string to paste into python file.
    slurm_paste = f"""#!/usr/bin/env python3
#SBATCH --job-name={data_fn}_LEPs
#SBATCH --time={time_val}   # walltime
#SBATCH --ntasks={ntasks_val}
#SBATCH --nodes={node_val}
#SBATCH --mem-per-cpu={mem_val}
#SBATCH --qos={qos_val}
#SBATCH --array=1-{num_nodes}"""
    # prep and run the script
    h.PrepSlurmScript(slurm_paste,start_script)
    slurm_command = [
        'sbatch',
        f'{start_script}',]
    # slurm_command = [
    #     'sbatch',
    #     f'--nodes={node_val}',
    #     f'--ntasks-per-node={ntasks_val}',
    #     f'--time={time_val}',
    #     f'--mem-per-cpu={mem_val}',
    #     '--job-name=' + f'{data_fn}_LEPs',
    #     '--array=1-' + f'{num_nodes}',
    #     f'--qos={qos_val}',
    #     '/home/jrhmc1/Desktop/EquitablePartitions/Slurm/_LEP_slurm.py'
    # ]   
    print(slurm_command)
    subprocess.run(slurm_command)


