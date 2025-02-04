#!/usr/bin/env python3
#SBATCH --job-name=build_bertha
#SBATCH --time=3-00:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024G
#SBATCH --tasks-per-node=1
#SBATCH --array=1-4
#SBATCH --qos=normal

#import ep_utils
import os, sys
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import ep_utils as epu

def birthBertha(size_id):
    sizes = np.array([2**i for i in [18.5,19,19.5,20]])#range(19,20)])
    sizes = np.floor(np.sqrt(sizes))**2
    sizes = sizes.astype(int)
    
    print(f"building Bertha of size {sizes[size_id]}")
    bertha = graphs.GenBerthaSparse(sizes[size_id],parallel=True)
    #bertha = graphs.GenBertha(sizes[size_id])
    bertha = nx.from_scipy_sparse_array(bertha)
    # code to check ep, lep struture and look at graph
    #ep,lep=epu.getEquitablePartitions(bertha)
    #print(ep,lep)
    #nx.draw(bertha)
    #plt.savefig('/home/jrhmc1/Desktop/EquitablePartitions/Networks/bertha_check.png')
    nx.write_graphml(bertha,f"/home/jrhmc1/Desktop/EquitablePartitions/Networks/bertha_{sizes[size_id]}.graphml")
    print("Finished.")


if __name__ == "__main__":
    import os
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    birthBertha(task_id-1)
