#!/usr/bin/env python3
#SBATCH --job-name=bertha_369664slurm
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
import ep_utils
import subprocess
from time import perf_counter as pc
import scipy.sparse as sp
import helper as h

if __name__ == "__main__":
    start_script = '/home/jrhmc1/Desktop/EquitablePartitions/Slurm/_LEP_slurm.py'
    # get graph we are doing analysis on
    graph_path = sys.argv[1]
    data_fn = graph_path.split('/')[-1].split('.')[0]
    os.environ['GRAPH_PATH'] = graph_path
    try: 
        try: G = nx.read_graphml(graph_path)
        except: G = sp.load_npz(graph_path)
    except: print("You need to give the file path to graph you want to run")
    out,t = tim.time_this(ep_utils.getEquitablePartitions,[G],ret_output=True,
                            label="get_eps",store_in='./' + data_fn + '_parallel.txt')
    ep,lep_list = out
    # get the eps and leps from that operation
    with open("EpLep.txt","w") as f:
        f.write("EPs==" + tim.serialize(ep))
        f.write("==LEPs==" + tim.serialize({i:list(lep) for i,lep in enumerate(lep_list)}))
        
    #os.environ['EP'] = tim.serialize(ep)
    #os.environ['LEPs'] = tim.serialize({i:list(lep) for i,lep in enumerate(lep_list)})
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


