import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import reduce
import sys

import ep_finder

# TODO: update naming to match paper

def getEquitablePartitions(G, timed = True, progress_bars = True):
    """Finds the coarsest equitable partition and local equitable partitions of a graph.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
        timed : boolean, optional
            Whether to also return the time taken to compute the EP and LEPs
    
    RETURNS:
        The equitable partition (dict; int -> set), local equitable partition (list of sets
            of partition elements grouped together), and computation time (when applicable)
    """
    start_time = time.time()
    C, N = ep_finder.initialize(G)
    ep, N = ep_finder.equitablePartition(C, N, progress_bar=progress_bars)
    coarsest = time.time() - start_time
    start_time = time.time()
    N_G = initialize(G)
    leps = getLocalEquitablePartitions(N_G, ep, progress_bar=progress_bars)
    local = time.time() - start_time
    if timed:
        return ep, leps, coarsest + local
    return ep, leps

def plotEquitablePartition(G, pi, pos_dict = None):
    """Plots the equitable partition of a graph, with each element in its own color.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to be plotted
        pi : dict
            The equitable partition of the graph, as returned by ep_finder
    """
    # stores the color for each node
    color_list = [0 for _ in range(G.number_of_nodes())]
    # iterator over equidistant colors on the color spectrum
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(pi))))
    # assign all vertices in the same partition element to the same color
    for V_i in pi.values():
        c = next(color)
        for vertex in V_i:
            color_list[vertex] = c
    
    nx.draw_networkx(G, pos=pos_dict, node_color=color_list)
    plt.show()

def printStats(G):
    # get coarsest EP and list of local EPs
    ep, leps, time = getEquitablePartitions(G)
    # keep non-trivial parts of EP and LEPs
    f_ep = list(filter(lambda i: len(i) != 1, ep.values()))
    # here, non-trivial just means that there are multiple nodes in the LEP
    f_leps = list(filter(lambda i: len(i) != 1 or len(ep[list(i)[0]]) != 1, leps.values()))
    # calculate how much is non-trivial
    partitionSize = lambda part_el: len(ep[part_el])
    # calculate number of non-trivial nodes
    nt_nodes = reduce(
        lambda part_sum, curr: part_sum + sum([partitionSize(i) for i in curr]),
        f_leps, 0)
    nt_nodes2 = reduce(
        lambda part_sum, curr: part_sum + len(curr), f_ep, 0)
    print("\nnt1 = {}\nnt2 = {}\n\n".format(nt_nodes, nt_nodes2))
    # percentage of nodes that are non-trivial
    nt_percent = nt_nodes * 100 / G.number_of_nodes()

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    general = "Nodes: {}, Edges: {}, Edges/Node: {}".format(
        num_nodes, num_edges, num_edges / num_nodes)
    computational = "EPs: {}, LEPs: {}, Time: {}".format(
        len(ep), len(leps), time)
    dist_template = "{} - ({}, {}): {}"
    distribution = dist_template.format("DATA", "MIN", "MAX", "AVG")
    # calculate some basic stats about non-trivial parts
    ep_distribution = dist_template.format("\nEP", *__getEPStats(f_ep))
    lep_distribution = dist_template.format("\nLEP", *__getEPStats(f_leps))
    printWithLabel("GENERAL COMPUTATION", '=', general + '\n' + computational)
    printWithLabel("DISTRIBUTIONS", '*', distribution + ep_distribution + lep_distribution)
    printWithLabel("PERCENT NON-TRIVIAL", '#', "{} %".format(nt_percent))

def __getEPStats(set_list):
    minSize = lambda min, curr: min if min < len(curr) else len(curr)
    maxSize = lambda max, curr: max if max > len(curr) else len(curr)
    sumSize = lambda part_sum, curr: part_sum + len(curr)
    min_len = reduce(minSize, set_list, sys.maxsize)
    max_len = reduce(maxSize, set_list, 0)
    avg_len = reduce(sumSize, set_list, 0) / max(len(list(set_list)), 1)
    return min_len, max_len, avg_len

def printWithLabel(label, delim, item):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item))