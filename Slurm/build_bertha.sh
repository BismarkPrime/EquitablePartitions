#!/bin/bash --login

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH -job-name=bertha_build   # job name
#SBATCH --array=1
##SBATCH --mail-user=jrhmc1@byu.edu   # email address
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
##SBATCH --qos=test


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#module load python/3.11
#python -m pip install mpi4py
#pip3 install alive_progress
#pip3 install networkx
#pip3 install scipy
#pip3 install numpy
#pip3 install alive_progress
#pip3 install matplotlib
#pip3 install tqdm
mamba activate advisor
echo "Running on: "${hostname}
which python3
#srun python3 ./test_mpi.py True
mpirun -n 4 python3 ./test_mpi.py
#python3 ./speed_test.py slurm top_power 10
