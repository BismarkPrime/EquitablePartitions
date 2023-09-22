import os,sys
import graphs
import ep_utils
from time import perf_counter as pc
import networkx as nx
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
#import profile
#import pstats
from importlib import reload

def create_berthas(top_power2=13):
    print("Creating Graphs")
    #sizes = list(set([int(2**i) for i in np.linspace(1,14,100)]))
    sizes = 2**np.arange(5,top_power2+1)
    prog = tqdm(sizes,total = len(sizes))
    berthas = {}

    for i in prog:
        
        try:
            G = graphs.GenBertha(i)
            berthas[i] = G
        except:
            print(f"failed for size {i}")
        prog.update()
    return berthas,sizes

def DuelOfMethods(bertha,verbose=False):
    """compares networkx's normal eigenvalue catching method to our method
    PARAMETERS:
        size (int): how big of graph you want to compare with
    RETURNS:
        our_time (float): how long our method took
        their_time (float): how long their method took
        accurate (bool): if we matched their spectrum
    """
    # normal method
    start = pc()
    their_spec = nx.adjacency_spectrum(bertha)
    end = pc()
    their_time = end-start
    
    # our method
    start = pc()
    our_spec = ep_utils.GetSpectrumFromLEPs(bertha,verbose=verbose)
    end = pc()
    our_time = end-start

    
    # check if our spectrums are the same
    accurate = Counter(np.round(np.array(their_spec),2)) == Counter(np.round(our_spec,2))
    
    return our_time, their_time, accurate

def test_speed(bertha_dict,sizes):

    our_times = []
    their_times = []
    acc_count = []

    progress_bar = tqdm(sizes,total=len(sizes))

    for size in progress_bar:
        print(f"For Bertha of size: {size}")
        us, them, peter, acc = 0, 0, 0, 0
        iters = 2
        for i in range(iters):
            us_i, them_i, acc_i = DuelOfMethods(bertha_dict[size])
            us, them, acc = us+us_i, them+them_i, acc+acc_i
        us, them, acc = us/iters, them/iters, acc/iters
        our_times.append(us)
        their_times.append(them)
        acc_count.append(acc)
        progress_bar.update()

    return our_times, their_times, acc_count

def plot_result(our_times,their_times,acc_count,sizes,fname="speed_comp.png"):
    plt.loglog(list(sizes),our_times,'-',color='forestgreen',label="Our Method")
    plt.loglog(list(sizes),their_times,'-',color='maroon',label='Naiive Method')
    #plt.loglog(sizes,p(size_list),'-',color='midnightblue',label='Polynomial Regression (squared)')
    #plt.loglog(sizes,p1(size_list),'-',color='chartreuse',label='Their Polynomial Regression (cubed)')

    plt.title(f"Method Comparison\nAccuracy: {np.sum(acc_count)/len(acc_count)}")
    plt.legend()
    plt.savefig(f'Speed_Graphs/{fname}')
    plt.show()

if __name__ == "__main__":
    graph_fname = input("Input the name of the speed test file to be saved (default: 'speed_comp.png): ")
    if graph_fname == '': graph_fname = 'speed_comp.png'       
    top_power2 = input("Input the highest power of 2 to go to when generating Bertha's sizes (default: 13): ")
    if top_power2 == '': top_power2 = 13
    else: top_power2 = int(top_power2)
    berthas, sizes = create_berthas(top_power2)
    us,them,acc = test_speed(berthas,sizes)
    plot_result(us,them,acc,sizes)