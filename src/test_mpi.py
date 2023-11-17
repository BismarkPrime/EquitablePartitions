import time
import math
import sys,os
import numpy as np
from multiprocessing import Pool
from time import perf_counter as pc
import ep_utils

def cube(x):
    return x**3, x**(1/3)

def multi_vs_serial(N,nproc=None):
    # using multiprocessing
    if nproc is not None:
        pool = Pool(processes = nproc)
    else:
        pool = Pool()
    start = pc()
    result = pool.map(cube,range(10,N))
    #print(f"number of processes: {pool._processes}")
    end = pc()
    print(f"Multiprocessing time: {end-start}\n---")
    # using serial computation
    start = pc()
    result2 = []
    for n in range(10,N):
        result2.append(cube(n))
    end = pc()
    print(f"Serial time: {end-start}")
    print(result[:10])
    print(final)

def mVs_network(N):
    """Test multiprocessing versus serial processing for finding the leps of a random geometric network"""
    graph = nx.random_geometric_graph(N,.1)
    ep, lep = ep_utils.getEquitablePartitions(graph)

    pool = Pool(processors=len(lep))


    ## PROBLEM: right now we only have a method to get the spectrum from the graph using the whole method. To test multiprocessing we'll probably
    # need to write a method that will get the spectrum just when handed the LEPs and the graph itself. 

if __name__ == "__main__":
    N = int(sys.argv[1])
    try:
        nproc = int(sys.argv[2])
    except:
        nproc = None
    multi_vs_serial(N,nproc)

    # OLD CODE FOR MPI4PY TESTING
    #verbose = sys.argv[1]
    #test_grouping(verbose)
    # testing the ways to split a list (enter mpirun -n4 python3 test_mpi.py top, skip, verbose)
    #top,skip,verbose = int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
    #method_test(top,skip,verbose)
    # See if a random graph lep_list will scatter properly
    #to_run = ["verbose = sys.argv[1]","test_grouping(verbose)"]
    #bouncer(to_run)

###################################################### CODE RETIREMENT #############################################################################
#################################### THIS CODE WAS USED FOR mpi4py WHICH IS INCOMPATIBLE WITH ######################################################
####################################     SLURM ON MARLYLOU, SO I AM NO LONGER USING IT.       ######################################################

def prep_scatter(data_list,tot_threads,verbose=False):
    """prepares a list to be scattered among a certain number of threads 
    for parallelization"""
    # create lists with both methods
    num_in_each = int(len(data_list)/tot_threads)
    scatter_list1 = [data_list[j*num_in_each:(j+1)*num_in_each] if j != (tot_threads-1) else data_list[j*num_in_each:] for j in range(tot_threads)]
    num_in_each = math.ceil(len(data_list)/tot_threads)
    scatter_list2 = [data_list[j*num_in_each:(j+1)*num_in_each] if j != (tot_threads-1) else data_list[j*num_in_each:] for j in range(tot_threads)]
    # check which is better
    diff1,diff2 = abs(len(scatter_list1[-1])-len(scatter_list1[-2])), abs(len(scatter_list2[-1]) - len(scatter_list2[-2]))
    if verbose: print(f"diff1: {diff1}\ndiff2: {diff2}")
    if diff1 <= diff2:
        if verbose: print("using full remainder method")
        return scatter_list1, True
    else:
        if verbose: print("using remainder distribution method")
        return scatter_list2, False

def method_test(top,skip,verbose=False):
    comm = MPI.COMM_WORLD
    total_threads =  comm.Get_size()
    current_thread = comm.Get_rank()
    counter = {"full remainder":0,"remainder distribution":0}
    for j in range(10,top,skip):
        testL = [i for i in range(j)]
        testL, full_rem = prep_scatter(testL,total_threads,verbose=verbose)
        if full_rem:
            counter["full remainder"] += 1
        else:
            counter["remainder distribution"] += 1

    print(counter)

def test_grouping(verbose=False):
    import ep_utils
    import networkx as nx
    # initialize mpi information
    comm = MPI.COMM_WORLD
    total_threads =  comm.Get_size()
    current_thread = comm.Get_rank()

    # make grandom geometric graph to test
    if current_thread == 0:
        graph = nx.random_geometric_graph(100,0.08)
        ep_dict,lep_list = ep_utils.getEquitablePartitions(graph)
        lep_list += [{'div'}]  # adding divisor matrix

        # scatter the lep list
        to_scatter, full_rem_used = prep_scatter(lep_list,total_threads)
        if current_thread == 0:
            print(f"num that will be in each thread is: {[len(piece) for piece in to_scatter]}")
        # scatter the list
        for thread,piece in [[i+1,part] for i,part in enumerate(to_scatter[1:])]:
            comm.send(piece,dest=thread,tag=0)
            #scattered = comm.scatter(to_scatter,root=0) # old, when scattering all at once
        scattered = to_scatter[0]
    else:
        print(f"current thread is: {current_thread}")
        scattered = comm.recv(source=0,tag=0)
    if verbose: print(f"thread {current_thread} has: {scattered}")
    print("Done\n")

def bouncer(to_run):
    automated = input("Do you want package installation to be automated? (yes/no): ")
    if automated=='no': automated = False
    else: automated = True

    try:
        for code in to_run:
            exec(code)
    except Exception as e:
        print(f"Exception encountered of type: {type(e)}")
        if type(e) is ModuleNotFoundError:
            mod_needed = str(e).split('\'')[-2]
            if automated:
                install=True
            else:
                install = input(f"module needed is {mod_needed}. Install it with pip? (yes/no): ")
            if install:
                print(f"Installing {mod_needed}")
                os.system(f"pip install {mod_needed}")
                os.system("mpirun -n 4 python test_mpi.py True")