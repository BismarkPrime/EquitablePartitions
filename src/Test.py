import graphs
import networkx as nx
from matplotlib import pyplot as plt
import time

sizes = [50 * 2 ** i for i in range(8)]

def runTime(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return end - start

labels = ["diameter", "radius", "averageShortestPathLength"]#, "edgeConnectivity", "nodeConnectivity"]

print("Generating Berthas...")
berthas = [graphs.genBertha(size) for size in sizes]

# diameter, radius, and average_shortest_path_length took about 1.2 seconds for 750 nodes and 120 seconds for 6000 nodes
print("Calculating diameters...")
diameterTimes = [runTime(nx.diameter, bertha) for bertha in berthas]
print("Calculating radii...")
radiusTimes = [runTime(nx.radius, bertha) for bertha in berthas]
print("Calculating average shortest path lengths...")
averageShortestPathLengthTimes = [runTime(nx.average_shortest_path_length, bertha) for bertha in berthas]
# print("Calculating edge connectivities...")
# edgeConnectivityTimes = [runTime(nx.edge_connectivity, bertha) for bertha in berthas]
# print("Calculating node connectivities...")
# nodeConnectivityTimes = [runTime(nx.node_connectivity, bertha) for bertha in berthas]

for label, times in zip(labels, [diameterTimes, radiusTimes, averageShortestPathLengthTimes]):#, edgeConnectivityTimes, nodeConnectivityTimes]):
    plt.plot(sizes, times, label=label)
    plt.legend()
    plt.show()

