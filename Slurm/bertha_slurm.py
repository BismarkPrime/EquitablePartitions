#!/usr/bin/env python3
#SBATCH --job-name=build_bertha
#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=256G
#SBATCH --tasks-per-node=1
#SBATCH --array=1-2
#SBATCH --qos=normal

#import ep_utils
import os, sys
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx
import numpy as np

def birthBertha(size_id):
    sizes = np.array([2**i for i in range(19,21)])#[2**i for i in range(17,19)])
    sizes = np.floor(np.sqrt(sizes))**2
    sizes = sizes.astype(int)
    bertha = graphs.GenBertha(sizes[size_id])
    nx.write_graphml(bertha,f"/home/jrhmc1/Desktop/EquitablePartitions/Networks/bertha_{sizes[size_id]}.graphml")


if __name__ == "__main__":
    import os
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    birthBertha(task_id-1)
