import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import reduce
import sys

import ep_finder

# TODO: update naming to match paper

def getEquitablePartitions(G, timed = True):
    """Finds the coarsest equitable partition and local equitable partitions of a graph.
   
    ARGUMENTS:
        G: the NetworkX graph
        timed (opt): whether to also return the time taken to compute the EP and LEPs
    
    RETURNS:
        The equitable partition (dict; int -> set), local equitable partition (list of sets
            of partition elements grouped together), and computation time (when applicable)
    """
    start_time = time.time()
    C, N = ep_finder.initialize(G)
    ep, N = ep_finder.equitablePartition(C, N)
    coarsest = time.time() - start_time
    start_time = time.time()
    leps = getLocalEquitablePartitions(ep, G)
    local = time.time() - start_time
    if timed:
        return ep, leps, coarsest + local
    return ep, leps

def plotEquitablePartition(G, pi):
    """Plots the equitable partition of a graph, with each element in its own color.
   
    ARGUMENTS:
        G: the NetworkX graph
        pi: the equitable partition of the graph, as returned by ep_finder
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
    
    nx.draw_networkx(G, node_color=color_list)
    plt.show()

def getLocalEquitablePartitions(G, ep, progress_bar = True):
    """This function finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        G: the NetworkX graph
        ep: the equitable partition of the graph, as returned by ep_finder
        progress_bar (opt): whether to show realtime progress bar
            (enabled by default)
    
    RETURNS:
        A list of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """

    progress = 0
    if progress_bar:
        print("FINDING LEPS...")
        updateLoadingBar(progress)

    # NOTE: perhaps using N (list of nodes and their neighbors) could more efficiently create edge_partition?

    # start_time = time.time()
    # array that maps nodes (indices) to their partition element
    partition_dict = np.empty(G.number_of_nodes(), int)
    for (element, nodes) in ep.items():
        for node in nodes:
            partition_dict[node] = element

    # the preceding portion of the code generally takes about 2% of the total LEP runtime
    progress = 2
    if progress_bar:
        updateLoadingBar(progress)
    
    # 2d array mapping partition elements to the edges connecting them; i.e.,
    #   edge_partition[0][3] should be a set of edges connecting partition elements
    #   0 and 3
    edge_partition = {}

    # the following loop takes about 76 percent of the LEP algorithm's runtime, so we should update the progress
    #   bar 76 times during the 
    num_edges_per_percent = G.number_of_edges() // 76
    edge_num = 0

    # populate "top right" half (i.e., i < j in key (i, j)) of edge_partition (bottom half is redundant for undirected graphs)
    for (i, j) in G.edges:
        edge_num += 1
        if progress_bar and num_edges_per_percent != 0 \
                and edge_num % num_edges_per_percent == 0:
            updateLoadingBar(progress + edge_num // num_edges_per_percent)
        part_i = partition_dict[i]
        part_j = partition_dict[j]
        key = (part_i, part_j) if part_i < part_j else (part_j, part_i)
        if key not in edge_partition:
            edge_partition.update({key: set()})
        edge_partition[key].add((i, j))
    
    progress = 78
    if progress_bar:
        updateLoadingBar(progress)

    # keeps track of which partition elements are stuck together by internal cohesion
    #   i.e., if the values at two indices are the same, the partition elements at those 
    #   two indices are stuck together in the adjacency matrix (AKA they are not externally
    #   consistent AKA their sub-adjacency matrix is not all 1s or all 0s)
    int_cohesion_list = list(ep.keys())

    # find and record which submatrices are internally cohesive
    # (we need only check the top right half of the matrix, since it is symmetric)

    edge_partition_num = 0
    edge_partition_el_per_percent = len(edge_partition) // 18

    # i and j are indices of the two partition elements in question
    for ((i, j), edge_set) in edge_partition.items():
        edge_partition_num += 1
        if progress_bar and edge_partition_el_per_percent \
                and edge_partition_num % edge_partition_el_per_percent == 0:
            updateLoadingBar(progress + edge_partition_num // edge_partition_el_per_percent)
        num_edges = len(edge_set)
        total_possible_edges = len(ep[i]) * len(ep[j])
        # if two partition elements are not externally consistent with one another
        #    (e.g., if they are internally cohesive)
        if num_edges != total_possible_edges:
            partition_element = min(int_cohesion_list[i], int_cohesion_list[j])
            curr = j if int_cohesion_list[i] < int_cohesion_list[j] else i
            # update back pointers RTL
            next = int_cohesion_list[curr]
            while next != curr:
                int_cohesion_list[curr] = partition_element
                curr = next
                next = int_cohesion_list[curr]
            # one more update needed once we have reached the leftmost partition element (when next == curr)
            int_cohesion_list[curr] = partition_element
    
    progress = 96
    if progress_bar:
        updateLoadingBar(progress)

    # consolidate pointers to make the implicit tree structure in internalCohesionList one level deep at most
    #   (in other words, update back pointers LTR)
    for i in range(len(int_cohesion_list)):
        int_cohesion_list[i] = int_cohesion_list[int_cohesion_list[i]]
    
    progress = 98
    if progress_bar:
        updateLoadingBar(progress)

    # this list sorts the partitions by their internal cohesion groups, while 
    #   preserving the indices to determine which parititon elements are together
    lep_list = enumerate(int_cohesion_list)
    lep_dict = dict()
    for (node, part_el) in lep_list:
        if part_el not in lep_dict:
            lep_dict.update({part_el: set()})
        lep_dict.get(part_el).add(node)
    
    progress = 100
    if progress_bar:
        updateLoadingBar(progress)
        print()

    return lep_dict.values()

def printStats(G):
    # get coarsest EP and list of local EPs
    ep, leps, time = getEquitablePartitions(G)
    # keep non-trivial parts of EP and LEPs
    f_ep = list(filter(lambda i: len(i) != 1, ep.values()))
    # here, non-trivial just means that there are multiple nodes in the LEP
    f_leps = list(filter(lambda i: len(i) != 1 or len(ep[list(i)[0]]) != 1, leps))
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
    ep_distribution = dist_template.format("\nEP", *getEPStats(f_ep))
    lep_distribution = dist_template.format("\nLEP", *getEPStats(f_leps))
    printWithLabel("GENERAL COMPUTATION", '=', general + '\n' + computational)
    printWithLabel("DISTRIBUTIONS", '*', distribution + ep_distribution + lep_distribution)
    printWithLabel("PERCENT NON-TRIVIAL", '#', "{} %".format(nt_percent))

def getEPStats(set_list):
    minSize = lambda min, curr: min if min < len(curr) else len(curr)
    maxSize = lambda max, curr: max if max > len(curr) else len(curr)
    sumSize = lambda part_sum, curr: part_sum + len(curr)
    min_len = reduce(minSize, set_list, sys.maxsize)
    max_len = reduce(maxSize, set_list, 0)
    avg_len = reduce(sumSize, set_list, 0) / max(len(list(set_list)), 1)
    return min_len, max_len, avg_len

def printWithLabel(label, delim, item):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item))

def updateLoadingBar(percent):
    print("\r [{0}] {1}%".format('#' * percent + ' ' * (100 - percent), percent), end='')