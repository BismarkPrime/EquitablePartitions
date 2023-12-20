#!/usr/bin/env python3
#SBATCH --job-name=build_bertha
#SBATCH --time=01:00:00   # walltime
##SBATCH --output=output.%A.%a.out
##SBATCH --error=error.%A.%a.err
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10G
#SBATCH --tasks-per-node=1
#SBATCH --array=1-18
#SBATCH --qos=normal

#import ep_utils
import os, sys
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import graphs
import networkx as nx

def birthBertha(size_id):
    sizes = [i**2 for i in range(130,301,10)]
    bertha = graphs.GenBertha(sizes[size_id])
    nx.write_graphml(bertha,f"/home/jrhmc1/Desktop/EquitablePartitions/Networks/bertha_{sizes[size_id]}.graphml")


if __name__ == "__main__":
    import os
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    birthBertha(task_id-1)
