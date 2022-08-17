import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import reduce
import sys
import math
from ep_finder import Node
from alive_progress import alive_bar

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

def initialize(G):
    """Initializes the inverted neighbor dictionary required to compute leps.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyzed
    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """

    g_rev = G.reverse() if G.is_directed() else G

    # NOTE: N stores the in-edge neighbors, i.e. N[v] returns all nodes w with an edge w -> v.
    #    Thus, it is different than just calling G.neighbors(v); (hence, we use G.reverse())
    N = { node:set(g_rev.neighbors(node)) for node in G.nodes() }
    return N

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

def getLocalEquitablePartitions(N, ep, progress_bar = True):
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        N : dict
            A dictionary containing nodes as keys with their in-edge neighbors as values
        ep : dict
            The equitable partition of the graph, as returned by ep_finder
        progress_bar : boolean
            whether to show realtime progress bar (enabled by default)
    
    RETURNS:
        A list of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """
    retval = None
    # if progress_bar:
    #     title = "COMPUTING LEPS"
    #     print("{0}\n{1}".format(title, '=' * len(title)))
    with alive_bar(3 * len(ep) + 1, title="COMPUTING LEPS\t", disable=not progress_bar) as bar:
        for i in __computeLocalEquitablePartitions(N, ep):
            bar()
            retval = i
    return retval

def __computeLocalEquitablePartitions(N, pi):
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        N : dict
            A dictionary containing nodes as keys with their in-edge neighbors as values
        pi : dict
            The equitable partition of the graph, as returned by ep_finder
    
    RETURNS:
        A list of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """

    # array that maps nodes (indices) to their partition element
    partition_dict = np.empty(len(N), int)
    for (element, nodes) in pi.items():
        for node in nodes:
            partition_dict[node] = element
        yield

    # keeps track of which partition elements are stuck together by internal cohesion,
    #   with partition element index as key and internally cohesive elements as values
    lep_network = dict()

    for (index, V) in pi.items():
        common_neighbors = set(N[V[0]])
        for v in V:
            common_neighbors.intersection_update(set(N[v]))
        yield
        for v in V:
            for unique_neighbor in set(N[v]).difference(common_neighbors):
                __link(index, partition_dict[unique_neighbor], lep_network)
        yield

    leps = __extractSCCs(lep_network, len(pi))
    yield leps

def __link(i, j, edge_dict):
    if i not in edge_dict:
        edge_dict.update({i: set()})
    edge_dict.get(i).add(j)

    if j not in edge_dict:
        edge_dict.update({j: set()})
    edge_dict.get(j).add(i)

def __extractSCCs(edge_dict, num_nodes):
    visited = set()
    scc_list = []
    for i in range(num_nodes):
        if i not in visited:
            scc = set()
            scc.add(i)
            visited.add(i)
            if i in edge_dict:
                neighbors = edge_dict.get(i)
                while len(neighbors) > 0:
                    j = neighbors.pop()
                    scc.add(j)
                    visited.add(j)
                    neighbors.update(edge_dict.get(j).difference(scc))
            scc_list.append(scc)
    return scc_list