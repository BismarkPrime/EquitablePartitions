import os,sys
import graphs
import ep_utils
from time import perf_counter as pc
import networkx as nx
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
#import profile
#import pstats

def create_berthas(top_power2=13):
    print("Creating Graphs")
    #sizes = list(set([int(2**i) for i in np.linspace(1,14,100)]))
    sizes = 2**np.arange(5,top_power2+1)
    prog = tqdm(sizes,total = len(sizes))
    with open("bertha_storage.pkl",'rb') as infile:
        berthas = pickle.load(infile)

    for n in prog:
        if n in berthas.keys():
            print(f"Already created for size {n}")
            prog.update()
            continue
        try:
            print(f"Generating for size {n}")
            G = graphs.GenBertha(n)
            berthas[n] = G
        except:
            print(f"failed for size {n}")
        prog.update()
    with open("bertha_storage.pkl","wb") as outfile:
        pickle.dump(berthas,outfile)
    return berthas,sizes

def DuelOfMethods(bertha,verbose=False,fake_parallel=False):
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
    our_spec = ep_utils.GetSpectrumFromLEPs(bertha,verbose=verbose,fake_parallel=fake_parallel)
    end = pc()
    our_time = end-start

    
    # check if our spectrums are the same
    accurate = Counter(np.round(np.array(their_spec),2)) == Counter(np.round(our_spec,2))
    
    return our_time, their_time, accurate

def test_speed(bertha_dict,sizes,fake_parallel=False):
    # get times if already recorded for certain sizes
    if 'times.pkl' in os.listdir():
        print("loading previously calculated times...")
        try:
            with open('times.pkl','rb') as infile:
                time_dict = pickle.load(infile)
                our_times = time_dict['us']
                their_times = time_dict['them']
                sizes_to_skip = time_dict['sizes']
                acc_count = [1.0]*len(our_times)
        except:
            our_times = []
            their_times = []
            sizes_to_skip = []
            acc_count = [] 
    else:
        our_times = []
        their_times = []
        sizes_to_skip = []
        acc_count = []

    # eliminate sizes we already have times for
    print(f"We already have times recorded for sizes: {sizes_to_skip}")
    sizes = list(set(sizes) - set(sizes_to_skip))
    if sizes == []:
        print("no times for new sizes requested.")
        return our_times, their_times, acc_count
    else:
        print(f"Testing speed for n = {sizes}")
    progress_bar = tqdm(sizes,total=len(sizes))

    for size in progress_bar:
        print(f"For Bertha of size: {size}")
        us, them, peter, acc = 0, 0, 0, 0
        iters = 2
        for i in range(iters):
            us_i, them_i, acc_i = DuelOfMethods(bertha_dict[size],fake_parallel=fake_parallel)
            us, them, acc = us+us_i, them+them_i, acc+acc_i
        us, them, acc = us/iters, them/iters, acc/iters
        our_times.append(us)
        their_times.append(them)
        acc_count.append(acc)
        progress_bar.update()

    # save the new recorded times
    with open('times.pkl','wb') as out:
        time_dict = dict(zip(['us','them','sizes'],[our_times,their_times,sizes_to_skip+sizes]))
        pickle.dump(time_dict,out)
    # save times recorded
    return our_times, their_times, acc_count

def plot_result(our_times,their_times,acc_count,sizes,fname="speed_comp.png"):
    sizes = np.array(list(sizes))
    naive = lambda x: x**(2.1) * (their_times[0]/(sizes[0]**(2.1)))
    optimized = lambda x: x**(3/2) * (our_times[0]/(sizes[0]**(3/2)))
    plt.loglog(sizes,our_times,'.-',color='forestgreen',label="Our Method Time")
    plt.loglog(sizes,their_times,'.-',color='maroon',label='Naiive Method Time')
    plt.loglog(sizes,optimized(sizes),'--',color='forestgreen',label="Theoretical Time (Ours)",alpha=.8)
    plt.loglog(sizes,naive(sizes),'--',color='maroon',label='Theoretical Time (Naiive)',alpha=.8)
    #plt.loglog(sizes,p(size_list),'-',color='midnightblue',label='Polynomial Regression (squared)')
    #plt.loglog(sizes,p1(size_list),'-',color='chartreuse',label='Their Polynomial Regression (cubed)')

    plt.title(f"Method Comparison\nAccuracy: {np.sum(acc_count)/len(acc_count)}")
    plt.legend()
    plt.savefig(f'Speed_Graphs/{fname}')
    plt.show()

if __name__ == "__main__":
    fake_parallel = False
    try:
        sys.argv[1]
        print("Conducting fake parallelized Speed Test")
        fake_parallel = True
    except:
        print("Conducting normal Speed Test")
    # get parameters of test
    graph_fname = input("Input the name of the speed test file to be saved (default: 'speed_comp.png): ")
    if graph_fname == '': graph_fname = 'speed_comp.png'       
    top_power2 = input("Input the highest power of 2 to go to when generating Bertha's sizes (default: 13): ")
    if top_power2 == '': top_power2 = 13
    else: top_power2 = int(top_power2)
    # conduct test
    berthas, sizes = create_berthas(top_power2)
    us,them,acc = test_speed(berthas,sizes)
    plot_result(us,them,acc,sizes,fname=graph_fname)