import time
import networkx as nx
import graphs
import ep_utils
from matplotlib import pyplot as plt
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@timer_decorator
def getLEParDEigenvalues(G: nx.Graph | nx.DiGraph):
    return ep_utils.getEigenvaluesNx(G)

@timer_decorator
def getNxEigenvalues(G: nx.Graph | nx.DiGraph):
    return nx.adjacency_spectrum(G)

@timer_decorator
def getBertha(n):
    return graphs.GenBertha(n)

sizes = list(range(250, 2750, 250))
print("Generating Berthas...", end='\r')
Gs, building_time = zip(*[getBertha(size) for size in sizes])
print("Running LEParD Algorithm...", end='\r')
our_res, our_time = zip(*[getLEParDEigenvalues(G) for G in Gs])
print("Running Traditional Algorithm...", end='\r')
their_res, their_time = zip(*[getNxEigenvalues(G) for G in Gs])

print("Verifying Correctness...", end='\r')
result_diffs = [ep_utils.getSymmetricDifference(ours, theirs) for ours, theirs in zip(our_res, their_res)]
for i, (ours, theirs) in enumerate(result_diffs):
    assert len(ours) == 0, \
        f"ERROR: For size {sizes[i]}, LEParD Algorithm produced the following eigenvalues not in traditional results: {ours}"
    assert len(theirs) == 0, \
        f"ERROR: For size {sizes[i]}, traditional algorithm produced the following eigenvalues not in LEParD results: {theirs}"

print("Plotting Results...", end ='\r')
plt.plot(sizes, list(zip(building_time, our_time, their_time)))
plt.xlabel("Size of Bertha")
plt.ylabel("Time (seconds)")
plt.yscale('log')
plt.legend(("Building Bertha", "Our Algorithm", "Traditional Algorithm"))
plt.show()