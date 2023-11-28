
import time
import networkx as nx
from matplotlib import pyplot as plt
from functools import wraps, partial
from typing import List, Tuple
from sys import getsizeof
from timeit import Timer
import numpy as np

from sizing import getsize

import graphs
import ep_utils
import ep_finder2

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

# @timer_decorator
def getLEParDEigenvaluesNx(G: nx.Graph | nx.DiGraph):
    return ep_utils.getEigenvaluesNx(G)

# @timer_decorator
def getTradEigenvaluesNx(G: nx.Graph | nx.DiGraph):
    return nx.adjacency_spectrum(G)

@timer_decorator
def getBertha(n):
    return graphs.GenBertha(n)

@timer_decorator
def getBertha2(n):
    return nx.from_scipy_sparse_array(graphs.GenBerthaSparse(n))

# testing relative sizes of data structures if N is stored as a list or a dict
def sizeTest(Gs: nx.Graph | nx.DiGraph) -> None:
    status("Computing Sizes...")
    dict_sizes = list()
    list_sizes = list()
    for G in Gs:
        N = ep_finder2.initFromNx(G)
        # using getsizeof (to see size of dict vs size of list) here demosntrates large difference between sizes
        # however, this difference is made almost inconsequential by the size of objects contained therein
        # TLDR: we should probably eventually use a list instead of a dict, but performance gains will likely still be
        #   minimal. Not low-hanging fruit
        dict_sizes.append(getsize(N))
        list_sizes.append(getsize(list(N.values())))

    status("Plotting Results...")
    plt.plot([len(G) for G in Gs], list(zip(dict_sizes, list_sizes)))
    plt.xlabel("Size of Bertha")
    plt.ylabel("Size of N (bytes)")
    plt.legend(("Using Dict", "Using List"))
    plt.show()

# testing to see the fastest way to check if matrix is symmetric for initializing ep_finder2 with sparse matrix
def equalitySpeedTest(G1, G2):
    t1 = Timer(lambda: (G1 != G2).nnz == 0)

    r1, c1 = G1.nonzero()
    def computeEquality(a1, a2):
        # r1, c1 = a1.nonzero()
        r2, c2 = a2.nonzero()
        return all((np.array_equal(r1, r2),
                np.array_equal(c1, c2),
                np.array_equal(a1.data, a2.data)))

    t2 = Timer(lambda: computeEquality(G1, G2))
    print(f"Speed of ( != ).nnz(): {min(t1.repeat(10, 10))}")
    print(f"Speed of np conditions: {min(t2.repeat(10, 10))}")
    
@timer_decorator
def initEPFromSparse(mat):
    return ep_finder2.initFromSparse(mat)

@timer_decorator
def initEPFromNx(mat):
    return ep_finder2.initFromNx(mat)

def status(msg: str, prevlen:List[int]=[0]) -> None:
    print(msg, ' ' * max(0, prevlen[0] - len(msg)), end='\r')
    prevlen[0] = len(msg)

# comparing LEParD algorithm to traditional methods
def epFinderInitSpeedTest(sizes):
    # status = partial(print, end='\r')
    status("Generating Sparse Berthas...")
    sparseBerthas = [graphs.GenBerthaSparse(size) for size in sizes]
    status("Generating Nx Berthas...")
    nxBerthas = [nx.from_scipy_sparse_array(mat) for mat in sparseBerthas]
    status("Initializing EP Finder with Sparse Berthas...")
    sparseNs, sparseInitTimes = zip(*[initEPFromSparse(mat) for mat in sparseBerthas])
    status("Initializing EP Finder with Nx Berthas...")
    nxNs, nxInitTimes = zip(*[initEPFromNx(G) for G in nxBerthas])
    status("Comparing Init Results for Equality...")
    def n_eq(n1, n2):
        return all((n1.label == n2.label,
                   n1.old_color == n2.old_color,
                   n1.new_color == n2.new_color,
                   np.all(n1.predecessors == n2.predecessors),
                   np.all(n1.successors == n2.successors),
                   n1.in_edge_count == n2.in_edge_count,
                   n1.out_edge_count == n2.out_edge_count))
    # for i, (sparseN, nxN) in enumerate(zip(sparseNs, nxNs)):
    #     for key in sparseN.keys():
    #         assert n_eq(sparseN[key], nxN[key]), f"Initialization is different for bertha with size {sizes[i]}"
    
    print("Plotting Results...", end ='\r')
    times = (sparseInitTimes, nxInitTimes)
    legend = ("EP Finder Init from Sparse", "EP Finder Init from Nx")

    plt.plot(sizes, list(zip(*times)))
    plt.xlabel("Size of Bertha")
    plt.ylabel("Time (seconds)")
    plt.legend(legend)
    plt.show()
    

def speedTest(Gs: List[nx.Graph | nx.DiGraph], build_times: List[float]=None) -> None:
    # note: this method has not been tested with DiGraphs
    status("Running LEParD Algorithm...")
    our_time = zip(*[min(Timer(lambda: getLEParDEigenvaluesNx(G)).repeat(1, 5)) for G in Gs])
    status("Running Traditional Algorithm...")
    their_time = zip(*[min(Timer(getTradEigenvaluesNx(G)).repeat(1, 5)) for G in Gs])

    # status("Verifying Correctness...")
    # result_diffs = [ep_utils.getSymmetricDifference(ours, theirs) for ours, theirs in zip(our_res, their_res)]
    # for i, (ours, theirs) in enumerate(result_diffs):
    #     assert len(ours) == 0, \
    #         f"ERROR: For size {len(Gs[i])}, LEParD Algorithm produced the following eigenvalues not in traditional results: {ours}"
    #     assert len(theirs) == 0, \
    #         f"ERROR: For size {len(Gs[i])}, traditional algorithm produced the following eigenvalues not in LEParD results: {theirs}"

    status("Plotting Results...")
    times = (our_time, their_time)
    legend = ("LEParD Algorithm", "Traditional Algorithm")
    if build_times is not None:
        times = build_times, *times
        legend = "Building Graphs",*legend
    plt.plot([len(G) for G in Gs], list(zip(*times)))
    plt.xlabel("Size of Bertha")
    plt.ylabel("Time (seconds)")
    plt.yscale('log')
    plt.legend(legend)
    plt.show()

def getBerthas(sizes: List[int]) -> Tuple[List[nx.Graph | nx.DiGraph], List[float]]:
    status("Generating Berthas...")
    berthas, building_time = zip(*[getBertha2(size) for size in sizes])
    return berthas, building_time

def berthaSpeedTest(sizes: List[int]) -> None:
    berthas, building_time = getBerthas(sizes)
    speedTest(berthas, building_time)

def berthaSizeTest(sizes: List[int]) -> None:
    berthas, _ = getBerthas(sizes)
    sizeTest(berthas)

if __name__ == "__main__":
    sizes = list(range(500, 5500, 500))
    berthaSpeedTest(sizes)
