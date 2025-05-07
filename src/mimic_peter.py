import sys
sys.path.append("/home/jrhmc1/Desktop/EquitablePartitions/src/")
import time
import networkx as nx
import graphs
import ep_utils
from matplotlib import pyplot as plt
import scipy.sparse as sp
from functools import wraps
import os,json
import pandas as pd

# I made this to try to learn how to use Peter's decorator method. So this doesn't need to be committed.


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@timer_decorator
def load_graphml(path):
    G = nx.read_graphml(path)
    return int(path.split('.')[0].split('_')[1])

@timer_decorator
def get_npz(path):
    G = sp.load_npz(path)
    return int(path.split('.')[0].split('_')[1])

@timer_decorator
def graphml_alg(path):
    ext = int(path.split('.')[0].split('_')[1])
    G = nx.read_graphml(path)
    ep_utils.getEquitablePartitions(G,progress_bars=False)
    return ext

@timer_decorator
def npz_alg(path):
    ext = int(path.split('.')[0].split('_')[1])
    G = sp.load_npz(path)
    ep_utils.getEquitablePartitions(G,progress_bars=False)
    return ext
"""
sizes = set()
for file in os.listdir():
    extension = file.split('.')[-1]
    if extension == 'graphml' or extension == 'npz':
        sizes.add(int(file.split('.')[0].split('_')[1]))

sizes = list(sizes)

print("loading graphmls...", end='\n')
graphml_sizes, graphml_time = zip(*[load_graphml(path) for path in os.listdir() if path.split('.')[-1] == 'graphml'])
print("loading npzs...", end='\n')
npz_sizes, npz_time = zip(*[get_npz(path) for path in os.listdir() if path.split('.')[-1] == 'npz'])
print("testing graphml algorithm...", end='\n')
_, graphml_alg_time = zip(*[graphml_alg(path) for path in os.listdir() if path.split('.')[-1] == 'graphml'])
print("testing npz algorithm...", end='\n')
_, npz_alg_time = zip(*[npz_alg(path) for path in os.listdir() if path.split('.')[-1] == 'npz'])

all_times = {size:{'graphml_load':time_load,'graphml_alg':time_alg,'npz_load':None,'npz_alg':None} for size,time_load,time_alg in zip(graphml_sizes,graphml_time,graphml_alg_time)}
for size, time_load, time_alg in zip(npz_sizes,npz_time,npz_alg_time):
    all_times[size]['npz_load'] = time_load
    all_times[size]['npz_alg'] = time_alg

# print("saving times...", end='\r')
# with open('all_times.json','w') as file:
#     json.dump(all_times,file)

print("Plotting Results...", end ='\r')
plt.scatter(graphml_sizes,graphml_time)
plt.scatter(npz_sizes,npz_time)
plt.xlabel("graph sizes (nodes)")
plt.ylabel("Time (seconds)")
plt.yscale('log')
plt.legend(("graphml times", "npz times"))
plt.savefig('loadIn_plot.png')

plt.clf()

plt.scatter(graphml_sizes,graphml_alg_time)
plt.scatter(npz_sizes,npz_alg_time)
plt.xlabel("graph sizes (nodes)")
plt.ylabel("Time (seconds)")
plt.yscale('log')
plt.legend(("graphml alg times", "npz alg times"))
plt.savefig('alg_plot.png')
"""

def decorator_timing(dec_list,save_fig=None):
    """ have a list of decorator functions that it will run one at a time and then
    plot the results. dec list is list of lists where inner list has the function to time
    and then the keyword for the list comprehension. save_fig is the name of the figure you 
    want to save if that's what you want to do
    """
    for func, keyword in dec_list:
        print(f"timing function:\n\t{func}\nfor keyword:\n\t{keyword}")
        graph_sizes, time = zip(*[func(path) for path in os.listdir() if keyword in path.split('_')[-1]])
        plt.scatter(graph_sizes,time,label=keyword)
        
    plt.xlabel("graph sizes (nodes)")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.legend()

    if save_fig is not None:
        plt.savefig(save_fig)

dec_list = [[graphml_alg,'graphml'],[npz_alg,'coo'],[npz_alg,'csr']]
decorator_timing(dec_list,save_fig='nxVcooVcsr.png')