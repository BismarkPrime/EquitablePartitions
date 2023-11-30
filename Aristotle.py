"""Meant to be used as the python script that will do organize the submissions to the supercomputer.
Aristotle will:
    1) Create a folder in the Results directory that is called graphName_res
    2) it will put the output spectrum and time files inside that folder
    3) it will create a subdirectory called slurm_output and put all the slurm files there
    4) it will record the times of the most recent run into the master dictionary and resave it as the 
        Master_times.json file.
"""

import numpy as np
import sys, os, subprocess, json
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import timing as tim
from time import perf_counter as pc
import networkx as nx
import helper as h

def ChangeToCorrectFolder(graph):
    os.chdir('./Results')
    # get into the right folder
    name = graph.split('.')[0] + '_results'
    if name in os.listdir(): # if there delete it (we are assuming the previous run information was saved in the master time json)
        subprocess.run(f'rm -r {name}',shell=True)
    # then create it and change to that directory
    os.makedirs(name,exist_ok=True)
    os.chdir(f'./{name}')
        

def TimeGraph(graph):
    """Times the graph both serially and in parallel and records the times in the master_dict"""
    # make sure params are what you want
    param_dict = tim.CheckSlurmParameters()
    # Get the values
    node_val,ntasks_val,time_val,mem_val,qos_val = [pair[1] for pair in param_dict.values()]
    job_name = graph.split('.')[0]
    slurm_command = [
        'sbatch',
        f'--nodes={node_val}',
        f'--ntasks-per-node={ntasks_val}',
        f'--time={time_val}',
        f'--mem-per-cpu={mem_val}',
        f'--job-name={job_name}_getEPs',
        f'--qos={qos_val}',
        '/home/jrhmc1/Desktop/EquitablePartitions/Slurm/time_slurm.py',
        f'/home/jrhmc1/Desktop/EquitablePartitions/Networks/{graph}'
    ]   
    subprocess.run(slurm_command)

if __name__ == '__main__':
    try: mandato = sys.argv[1]
    except: 
        print("\n\nyou must choose a command for Aristotle\n\trun -> run a timing\n\tclean -> compile timing results\n\n")
        
    if mandato == 'run':
        # show possible graphs to run time tests on
        print("Available graphs:\n")
        graphml_list = [file for file in filter(lambda x: 'graphml' in x or 'csr' in x or 'coo' in x,os.listdir("./Networks/"))]
        for i,graph in enumerate(graphml_list):
            print(f"{i}: {graph}")
        
        # ask which ones you want to run the test on
        decision = input("Choose which graph to run a speed test on:\n\tsingle number = only that graph"
                        "\n\trange of numbers (#:#) = will test that range (ending index graph included)"
                        "\n\tlist of numbers with spaces between (# # # #...) = will grab only those #'s"
                        "\nYour choice is: ")

        # if given a range
        if ':' in decision:
            first,last = decision.split(':')
            first = int(first)
            last = int(last)
            for graph in graphml_list[first:last+1]:
                ChangeToCorrectFolder(graph)
                TimeGraph(graph)
                os.chdir('../..')
        elif len(decision) == 1:
            ChangeToCorrectFolder(graphml_list[int(decision)])
            TimeGraph(graphml_list[int(decision)])
        else:
            for ind in decision.strip().split(' '):
                ChangeToCorrectFolder(graphml_list[int(ind)])
                TimeGraph(graphml_list[int(ind)])
                os.chdir('../..')
    
    elif mandato == 'clean':
        print("Please still implement me.")
        
