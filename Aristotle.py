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
import pandas as pd
import re

def GetRunInfo(results_fn):
    """records the time taken to do a run in the master_keeper.csv file. This function must
    be called in the results folder of the run you want to save."""
    # get the graph format
    g_name = results_fn.split('_')[0] + results_fn.split('_')[1]
    if 'csr' in results_fn: g_format = 'csr'
    elif 'coo' in results_fn: g_format = 'coo'
    elif 'lil_matrix' in results_fn: g_format = 'lil_matrix'
    else: g_format = 'graphml'

    # get the timing file
    timing_file = [file for file in os.listdir('./' + results_fn) if 'parallel' in file or 'serial' in file]
    spectrum_file = [file for file in os.listdir('./' + results_fn) if 'spec' in file]

    # if parallel run get all the details
    if 'parallel' in timing_file[0]: 
        if not timing_file:
            raise NameError("parallel or serial timing file not present.")
        if not spectrum_file:
            raise NameError("spectrum file not present. timing did not run to completion.")
        print(f"THE RESULTS FILE IS: {results_fn}")
        with open(results_fn + '/' + timing_file[0],'r') as f:
            data = f.read()
        total,longest_lep,longest_com,longest_g = 0,np.inf,np.inf,np.inf # initialize times

        # get the times we'll record
        for t_record in data.strip().split('\n'):
            if len(t_record.split(':')) == 1: continue
            label,t = t_record.split(':')
            t = float(t)
            # only take the longest time since that will be what counts the most
            if 'lep_time' in label: 
                if t < longest_lep: longest_lep = t
            elif 'communication' in label: 
                if t < longest_com: longest_com = t
            elif 'build_graph' in label: 
                if t < longest_g: longest_g = t
            elif 'eps' in label: total += t
        total += longest_com + longest_lep + longest_g
        total_noGraph = total - longest_g
        time_pieces = {'longest_lep_time':longest_lep,'longest_comm_time':longest_com,'longest_graphBuild':longest_g}
        return [g_name,g_format,'parallel',time_pieces,total_noGraph,total]
    
    elif 'serial' in timing_file[0]: # if serial run there's only one time recorded
        if not timing_file:  # only check for timing file
            raise NameError("parallel or serial timing file not present.")
        with open(results_fn + '/' + timing_file[0],'r') as f:
            data = f.read()
        total = float(data.strip().split(':')[1])
        return [g_name, g_format,'serial',None,None,total]
    
    else: # otherwise something was wrong
        print('Something went wrong, couldn\'t find whether it was serial or parallel')


def EnsureRecord(results_fn):
    # may need to change this once slurm files get put into a folder
    slurm_list = [file for file in os.listdir('./' + results_fn) if file.startswith('slurm')]
    slurm_id = slurm_list[0].split('-')[1].split('_')[0].split('.')[0]
    df = pd.read_csv('/home/jrhmc1/Desktop/EquitablePartitions/Results/Master_keeper.csv',index_col="SlurmID")
    try: df.loc[slurm_id] # do nothing if it exists
    except: 
        stats = GetRunInfo(results_fn)
        df.loc[slurm_id] = stats
        df.to_csv('/home/jrhmc1/Desktop/EquitablePartitions/Results/Master_keeper.csv',index_label="SlurmID")

def ChangeToCorrectFolder(graph,serial=None):
    os.chdir('./Results')
    if serial is None:
        print("Need to specify serial")
    else:
        if serial: tag="_serial"
        if not serial: tag="_slurm"
    # get into the right folder
    name = graph.split('.')[0] + '_results' + tag
    if name in os.listdir(): # make sure the run has been save and then recreate the folder.
        try: 
            EnsureRecord(name) # put try because errors mean there is no data to record so I want to just delete it anyway, might not be the best fix though JRH
            subprocess.run(f'rm -r {name}',shell=True)
        except: 
            print(f"\nPROBLEM WITH {name}, changing name to reflect this...")
            subprocess.run(f'mv {name} {name}_prob',shell=True)
            os.chdir('..')
            return False
    # then create it and change to that directory
    os.makedirs(name,exist_ok=True)
    os.chdir(f'./{name}')
    return True

def TimeGraph(graph: str,serial=False) -> None:
    """Times the graph both serially and in parallel and records the times in the master_dict"""
    if serial: start_script= '/home/jrhmc1/Desktop/EquitablePartitions/Slurm/time_serial.py'
    else: start_script='/home/jrhmc1/Desktop/EquitablePartitions/Slurm/time_slurm.py'
    tag = start_script.split('_')[1].split('.')[0]
    # make sure params are what you want
    print(f"\n\nSTARTING RUN FOR: {graph}\n\n")
    print(os.getcwd())
    param_dict = tim.CheckSlurmParameters()
    # Get the values
    node_val,ntasks_val,time_val,mem_val,qos_val = [pair[1] for pair in param_dict.values()]
    job_name = graph.split('.')[0]
    # get string to paste into python file.
    slurm_paste = f"""#!/usr/bin/env python3
#SBATCH --job-name={job_name+tag}
#SBATCH --time={time_val}   # walltime
##SBATCH --ntasks-per-node={ntasks_val}
##SBATCH --nodes={node_val}
#SBATCH --mem-per-cpu={mem_val}
#SBATCH --tasks-per-node=1
#SBATCH --qos={qos_val}"""
    # prep and run the script
    h.PrepSlurmScript(slurm_paste,start_script)
    slurm_command = [
        'sbatch',
        f'/{start_script}',
        f'/home/jrhmc1/Desktop/EquitablePartitions/Networks/{graph}'
    ]  
    # slurm_command = [
    #     'sbatch',
    #     f'--nodes={node_val}',
    #     f'--ntasks-per-node={ntasks_val}',
    #     f'--time={time_val}',
    #     f'--mem-per-cpu={mem_val}',
    #     f'--job-name={job_name+tag}',
    #     f'--qos={qos_val}',
    #     f'/home/jrhmc1/Desktop/EquitablePartitions/Slurm/{start_script}',
    #     f'/home/jrhmc1/Desktop/EquitablePartitions/Networks/{graph}'
    # ]   
    subprocess.run(slurm_command)

def ProcessDecision(decision,graph_list,serial):
    """takes the input of the user to tell what graphs to time and times those graphs"""
    pattern = re.compile("\w+=\S+")
    digit_catcher = re.compile("\d+")
    commands = re.findall(pattern,decision)

    if commands:
        for command in commands:
            if 'type' in command: # is you want to run all graphs of one graph type
                graph_list = [file for file in filter(lambda x: command.split('=')[1] in x, graph_list)]
            if 'range' in command: # then we only want to run graphs of a specific size
                low,high = command.split('=')[1].split('-') # getting top and bottom
                low = int(low)
                if high: high = int(high)
                else: high = np.inf
                print(high,low)
                # filter by graph size
                graph_list = [f for f in filter(lambda x: low <= int(re.findall(digit_catcher,x)[0]) and int(re.findall(digit_catcher,x)[0]) < high, graph_list)]

        print("Available graphs:\n")
        for i,graph in enumerate(graph_list):
            print(f"{i}: {graph}")
        print()    
    
        # ask which ones you want to run the test on
        decision = input("Choose which graph to run a speed test on:\n\tsingle number = only that graph"
                    "\n\trange of numbers (#:#) = will test that range (ending index graph included)"
                    "\n\tlist of numbers with spaces between (# # # #...) = will grab only those #'s"
                    "\nYour choice is: ")
    if ':' in decision: # if given a range of indices
        first,last = decision.split(':')
        first = int(first)
        last = int(last)
        for graph in graph_list[first:last+1]:
            ready = ChangeToCorrectFolder(graph,serial=serial)             # create results folder (built into this is recording the last run)
            if ready:
                TimeGraph(graph,serial=serial)               # Time the algorithim on the graph
                os.chdir('../..')                            # go back to Aristotles directory
            else:
                continue
        return
    elif len(decision) == 1: # if only one index given
        ChangeToCorrectFolder(graph_list[int(decision)],serial=serial)
        TimeGraph(graph_list[int(decision)],serial=serial)
        return
    elif ' ' in decision: # if multiple indices that aren't a range was given
        for ind in decision.strip().split(' '):
            ChangeToCorrectFolder(graph_list[int(ind)],serial=serial)
            TimeGraph(graph_list[int(ind)],serial=serial)
            os.chdir('../..')
        return

    # run for filtered graphs based on commands
    for graph in graph_list:
        ChangeToCorrectFolder(graph,serial=serial)
        TimeGraph(graph,serial=serial)
        os.chdir('../..')
    

def RunChoice():
    """shows the list of graphs and asks you which you want to run
    """
    # get run type (serial or slurm)
    run_type = h.parse_input("Is this a serial or slurm run? (serial/slurm): ")
    if run_type == 'serial': serial=True
    else: serial=False

    # show possible graphs to run time tests on
    print("Available graphs:\n")
    graph_list = [file for file in filter(lambda x: 'graphml' in x or 'csr' in x or 'coo' in x,os.listdir("./Networks/"))]
    for i,graph in enumerate(graph_list):
        print(f"{i}: {graph}")
    
    # ask which ones you want to run the test on
    decision = input("Choose which graph to run a speed test on:\n\tsingle number = only that graph"
                    "\n\trange of numbers (#:#) = will test that range (ending index graph included)"
                    "\n\tlist of numbers with spaces between (# # # #...) = will grab only those #'s"
                    "\n\nYOU MAY FILTER YOUR CHOICES WITH:"
                    "\n\ttype=network_type (type=graphml) = will run on all graphs of that type"
                    "\n\trange=#-# or #-  (range=1200- or range=1400-1900)"
                    "\n\nYour choice is: ")
    
    return decision,graph_list,serial

if __name__ == '__main__':
    try: mandato = sys.argv[1]
    except: 
        print("\n\nyou must choose a command for Aristotle\n\trun -> run a timing\n\trecord -> record timing results in master keeper csv file\n\n")
        
    if mandato == 'run':
        # gets the graphs you want to run
        decision,graph_list,serial = RunChoice()
        # runs the graphs that you chose
        ProcessDecision(decision,graph_list,serial)        
    
    elif mandato == 'record':
        os.chdir('/home/jrhmc1/Desktop/EquitablePartitions/Results') 
        for results_folder in os.listdir():
            if '_results' in results_folder:
                if 'recorded' in results_folder or 'prob' in results_folder:
                    continue
                else:
                    try: 
                        EnsureRecord(results_folder)
                        print(f"{results_folder} RUN RECORDED!")
                        subprocess.run(f"mv {results_folder} {results_folder}_recorded",shell=True)
                    except Exception as e:
                        print(f"PROBLEM WITH RESULTS: {results_folder}\nGAVE ERROR: {e}\nchanging name to reflect problem")
                        subprocess.run(f"mv {results_folder} {results_folder}_prob",shell=True)      
                        

            # TODO: problem with the cleaning code!

