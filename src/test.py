import math
from pprint import pprint
from timeit import Timer
from typing import Iterable
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import json
import random
import timeit
from functools import partial
import pdb

import ep_finder
import ep_finder2
import lep_finder
import graphs
import ep_utils
from graphs import GetLocalSpec

# TODO: update naming to match paper
# TODO: add LEP verification tests

import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def testSlots():
    ns = list(range(250, 2750, 250))
    print("Generating Graphs...", end='\r')
    Gs = [graphs.GenBertha(n) for n in ns]
    print("Computing Coarsest EP w/Dict...", end='\r')
    dict_times = [min(timeit.Timer(partial(runEP, G)).repeat(repeat=10, number=20)) / 10 for G in Gs]
    print("Computing Coarsest EP w/Slots...", end='\r')
    slot_times = [min(timeit.Timer(partial(runEPSlots, G)).repeat(repeat=10, number=20)) / 10 for G in Gs]
    print("Plotting Results...", end='\r')
    plt.plot(ns, list(zip(dict_times, slot_times)))
    plt.xlabel("Size of Bertha")
    plt.ylabel("Time to Compute Coarsest EP")
    plt.title("Slots vs Dicts Runtime in EP Finder")
    plt.legend(("Dictionary", "Slots"))
    plt.show()

def runEP(G):
    ep_finder.equitablePartition(*ep_finder.initialize(G), False)

def runEPSlots(G):
    # for testing, ep_finder1 had slots and ep_finder didn't. Since slots are faster, they have now been added to ep_finder
    # ep_finder1.equitablePartition(*ep_finder1.initialize(G), False)
    pass

def genDivGraph(G,ep_dict,retMat=False):
    """calculates and returns the divisor graph of the input graph
    INPUTS:
        G (networkx graph): the graph whose divisor graph you want
        ep_dict (dict): the dictionary defining the equitable partition on G
        retMat (bool): whether or not the function returns the divisor graph as a matrix
        
    RETURNS:
        divMat (array): the divisor matrix
        OR
        nx.Graph(divMat): divisor matrix as nx graph object"""
    # helper function to get key given a value
    def get_key(val,my_dict):
        for key, value_list in list(my_dict.items()):
            if val in value_list:
                return key
        print("they key does not exist")
        
    # create empty divisor to fill
    n = len(ep_dict)
    divMat = np.zeros((n,n))
    
    # cycle through one node in each partition element to get connections for the Divisor Graph
    for curPartElInd, node_list in enumerate(ep_dict.values()):
        # always get the first one since the partition is equitable and all connection 
        # will be the same within the partition elements.
        node = node_list[0]
        #print(f"this is the input node: {node}") # debugging statement
        # count connection to partitions and update divisor matrix accordingly
        for connection in G.edges(node):
            connNode = connection[1]
            #print(curPartElInd,get_key(connNode,ep_dict)) # debugging statement
            divMat[curPartElInd][get_key(connNode,ep_dict)]+=1
    # return matrix if desired
    if retMat:
        return divMat
    # otherwise return the divisor graph as a networkx object
    return nx.DiGraph(divMat)

def GetLocalSpec(G,ep_dict,lep_list):
    """return the spectrum of each LEP along with its divisor spectrum in a dictionary"""
    orig_spec = np.round(nx.adjacency_spectrum(G),3)
    GDiv = genDivGraph(G,ep_dict)
    GDivSpec = np.round(nx.adjacency_spectrum(GDiv),3)
    spec_dict = {} # dictionary to hold spectrums
    
    for lep in list(lep_list): # cycle through each lep
        node_list = []   # place to get all nodes in lep
        temp_ep_dict = {} # make a place for the original ep partitions
        
        # iterate through each partition element in that lep.
        for partElInd, partEl in enumerate(lep):
            node_list += ep_dict[partEl] # after this loop node_list has all nodes in the lep
            temp_ep_dict[partElInd] = ep_dict[partEl] # make the temporary ep_dict
                        
        subgraph = nx.subgraph(G,node_list) # make a subgraph
        spec_dict[f"{lep}"] = np.round(nx.adjacency_spectrum(subgraph),3)  # store its spectrum in dictionary
       
        spec_dict[f"{lep} Divisor"] = np.round(nx.adjacency_spectrum(genDivGraph(subgraph,temp_ep_dict)),3)
    # spec_dict["Original Graph Divisor"] = GDivSpec
        
    return spec_dict, GDivSpec, orig_spec

# test code
def test(p=.04, iters=500, nodes=40):
    for nodes in range(20, 160, 20):
        print(f"\nChecking graphs with {nodes} nodes")
        for i in range(iters):
            print(f'\r{i}', end='')
            G = nx.random_internet_as_graph(nodes, seed=i)
            if not ep_utils.compareEigenvalues(G):
                print("ERROR")
                if (input() == 'v'):
                    print(nx.adjacency_matrix(G, dtype=int).todense())
                    ep_dict = ep_utils.getTransceivingEP(G)
                    ep_utils.plotEquitablePartition(G, ep_dict)
                    input()

    # G = nx.random_geometric_graph(40, .15)
    # pi, leps = lep_finder.getEquitablePartitions(G, False, False)
    # lep_finder.plotEquitablePartition(G, pi, nx.get_node_attributes(G, "pos"))

def testCorrectness(p=.04, iters=3000, nodes=40):
    for nodes in range(20, 80, 20):
        print(f"\nChecking graphs with {nodes} nodes")
        for i in range(iters):
            print(f'\r{i}', end='')
            G = nx.erdos_renyi_graph(nodes, 3.2 / nodes, directed=True, seed=i)
            pi1 = list(ep_utils.getTransceivingEP(G)[0].values())
            pi2 = list(ep_utils.getTransceivingEP2(G)[0].values())
            for l in pi1:
                l.sort()
            for l in pi2:
                l.sort()
            pi1.sort()
            pi3 = sorted(pi2)
            if pi1 != pi3:
                print("ERROR")
                if (input() == 'v'):
                    print(nx.adjacency_matrix(G, dtype=int).todense())
                    pprint(f"EP from ep_finder:\n{pi1}")
                    pprint(f"EP from ep_finder2:\n{pi3}")
                    pprint(f"EP2 orig: {pi2}")
                    input()

# function to test runtime complexity
def complexityTest():
    num_nodes = list()
    ep_comp_time = list()
    ep2_comp_time = list()
    for nodes in range(200, 4000, 200):
        print(f"\rComputing iteration w/ {nodes} nodes.", end='')
        num_nodes.append(nodes)
        # G = nx.erdos_renyi_graph(nodes, 2.4 / nodes, directed=True)
        G = nx.random_internet_as_graph(nodes)
        func = lambda: ep_utils.getTransceivingEP(G)
        func2 = lambda: ep_utils.getTransceivingEP2(G)
        t = Timer(func)
        t2 = Timer(func2)
        ep_comp_time.append(t.timeit(15))
        ep2_comp_time.append(t2.timeit(15))

    
    plt.scatter(num_nodes, ep_comp_time, color='b', label="ep_finder")
    plt.scatter(num_nodes, ep2_comp_time, color='r', label="ep_finder2")
    plt.title("EP vs EP2 Computation Time")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time')
    plt.legend(loc="upper left")
    plt.show()

def compareEPEigenvalues(G: nx.Graph | nx.DiGraph, pi, leps) -> None:
    spec_dict, div_spec, orig_spec = GetLocalSpec(G, pi, leps)
    print("EP Eigenvalues:")
    # pprint(getEigenvalues(G))
    pprint(orig_spec)

    print("LEP Eigenvalues:")
    lep_eigs = []
    for key, value in list(spec_dict.items()):
        if "Divisor" not in key:
            lep_eigs.append(value)
    pprint(lep_eigs)
    print("Divisor Eigenvalues:")
    pprint(div_spec)
    # pprint(GetLocalSpec(G, pi, leps)[0])
    # pprint(list(filter(lambda x: len(x) > 1, getEPEigenvalues(G).values())))


def getEPEigenvalues(G: nx.Graph | nx.DiGraph) -> Iterable[numpy.ndarray]:
    ep, leps = ep_utils.getEquitablePartitions(G, progress_bars=False)
    for lep in leps:
        if len(lep) > 1:
            lep = list(lep)
            for i in lep[1:]:
                ep[lep[0]] += ep[i]
                del ep[i]
    eigen_dict = {}
    for index, partition_element in ep.items():
        eigen_dict[index] = getEigenvalues(G.subgraph(partition_element))
    return eigen_dict

def getEigenvalues(G: nx.Graph | nx.DiGraph) -> Iterable[numpy.ndarray]:
    L = nx.to_numpy_array(G)
    return numpy.linalg.eigvals(L)

def findInterestingGraphs():
    while True:
        G = nx.gnp_random_graph(20, .1, directed=True)
        vals = test.getEPEigenvalues(G).values()
        for l in vals:
                if np.any(l):
                        pprint(vals)
                        input()
                        break


def hasAllNodesOnce(G, start, end):
    for i in range(start, end):
        if not G.has_node(i):
            print("Missing node {}".format(i))
            return False
    return True
    
def graphIsUndirected(G):
    for (i, j) in G.edges:
        if not (j, i) in G.edges:
            print("Missing edge ({}, {})".format(j, i))
            return False
    return True

def plotIfInvalid(G):
    ep, leps = lep_finder.getEquitablePartitions(G)
    if not isPartition(ep, G) or not isEquitable(ep, G):
        print(ep)
        lep_finder.plotEquitablePartition(G, ep)


# generates random geometric graphs until it finds ep_finder problems
def findBadEPs():
    i = 0
    while (True):
        G = nx.gnp_random_graph(20, .14, seed=i, directed=False)
        G = graphs.randomRelabel(G)
        if not validEpResults(G):
            print("Current seed: {}".format(i))
            print("Press <Enter> to continue...")
            input()
        i += 1
        print("\r{}".format(i), end='')

def validEpResults(G):
    pi, leps = ep_utils.getEquitablePartitions(G, False, False)
    if not isPartition(pi, G):
        printWithLabel("PI IS NOT A PARTITION!!!", '=', pi)
        return False
    if not isEquitable(pi, G):
        print("PI IS NOT EQUITABLE!!!")
        # pos_dict = {}
        # for node in G.nodes:
        #     pos_dict.update({ node: node.pos })
        print(pi)
        ep_utils.plotEquitablePartition(G, pi)
        return False
    return True

def isPartition(pi, G):
    # vertex_count = np.ones(G.number_of_nodes())
    vertices = set()
    # verify that each vertex shows up exactly once in pi
    for V_i in pi.values():
        for vertex in V_i:
            if vertex in vertices:
                return False
            vertices.add(vertex)
    
    return len(vertices) == G.number_of_nodes()

def isEquitable(pi, G):
    # create table for fast node-to-partition lookups
    partition_dict = dict() #np.empty(G.number_of_nodes(), int)
    for (element, nodes) in pi.items():
        for node in nodes:
            partition_dict[node] = element

    g_rev = G.reverse() if G.is_directed() else G
    
    for V_i in pi.values():
        if len(V_i) > 1:
            # dict of partition_element -> number of connections
            rule = {}
            for i, vertex in enumerate(V_i):
                # construct rule
                if i == 0:
                    rule = getPartitionNeighbors(vertex, G, partition_dict, g_rev)
                # test other vertices against the rule
                else:
                    conns = getPartitionNeighbors(vertex, G, partition_dict, g_rev)
                    if conns != rule:
                        print(V_i)
                        print("last call{}getPartitionNeighbors({}, {}, part)".format('-'*30, vertex, G))
                        print("RULE: {}\nCONNS: {}".format(rule, conns))
                        return False
    return True

# specifically, we are getting the in-edge neighbors
def getPartitionNeighbors(vertex, G, partition_dict, g_rev):
    conns = {}
    for neighbor in g_rev.neighbors(vertex):
        part_el = partition_dict[neighbor]
        if part_el not in conns:
            conns.update({ part_el: 0 })
        conns.update({ part_el: conns[part_el] + 1 })
    return conns

# FIXME: bad function name
def areLEPs(leps, G, pi):
    
    return True

def iterationsTest():
    num_nodes = list()
    num_iters = list()
    for i in range(2000, 50000, 2000):
        num_nodes.append(i)
        G = nx.random_internet_as_graph(i)
        # func = lambda: ep_finder.getIters(*ep_finder.initialize(G))
        # t = Timer(func)
        # computation_time.append(t.timeit(1))
        iters = ep_finder.getIters(*ep_finder.initialize(G))
        num_iters.append(iters)
    
    plt.scatter(num_nodes, num_iters, color='b', label="LEP Algorithm")
    plt.title("Input Size vs. Iterations Computation Time")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of Iterations')
    plt.legend(loc="upper left")
    plt.show()

# OLD CODE:

# duplicated here for usability (also found in lep_finder)
def printWithLabel(label, delim, item):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item))

def main(nodes = 1000, print_results = False):

    G = nx.random_internet_as_graph(nodes)

    C, N = ep_finder.initialize(G)
    ep, N = ep_finder.equitablePartition(C, N)

    if print_results:
        printWithLabel("COARSEST EQUITABLE PARTITION", '=', ep)
    
    leps = lep_finder.getLocalEquitablePartitionsFromGraph(ep, G)

    if print_results:
        printWithLabel("LOCAL EQUITABLE PARTITIONS", '=', leps)

    print("(Nodes, Partition Elements, LEPs): ({}, {}, {})".format(nodes, len(ep), len(leps)))

    lep_finder.plotEquitablePartition(G, ep)

def graphWithColoredPartEl(adj_mat, ep):
    """draws the nx graph with color coded partition elements for the coursest EP
    if given a permuted adjacency matrix
    
    ARGUMENTS
    =========
    ep (dict): containts the nodes in each partition element"""
    hexColors = list(matplotlib.colors.cnames.values())  #get hex colors
    colorArr = [0 for i in range(max(max(ep.values()))+1)] #create place to store node colors
    index = 0    #start node index as 9
    for partEl in ep.values():   #cycle through each partition element
        color = random.randint(0,148)   #get a random color to assign to this partition element
        for i in range(len(partEl)):   #cycle through all nodes in partition element
            colorArr[index] = hexColors[color]   #assign each node that color
            index+=1   #use this to get different colorArr position each time
    
    nx.draw_networkx(nx.from_numpy_array(adj_mat),node_color=colorArr)    #graph it with colors

def getLocalEquitablePartitions(ep, adjMat, print_subgraphs = False, verbose = False):
    """This function finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        ep: the equitable partition of the graph, as returned by ep_finder
        adjMat: the adjacency matrix of the graph, permuted so as to place each
            partition element in a contiguous block in the matrix; the order
            must match the order or partition elements in ep
        printSubgraphs (opt): whether to print the submatrices of connections
            between partition elements; false by default
        verbose (opt): whether to print data structures at various stages of the
            algorithm; false by default
     
    RETURNS:
        A lit of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """

    # keeps track of which partition elements are stuck together by internal cohesion
    #   i.e., if the values at two indices are the same, the partition elements at those 
    #   two indices are stuck together in the adjacency matrix (AKA they are not externally
    #   consistent AKA their sub-adjacency matrix is not all 1s or all 0s)
    int_cohesion_list = list(ep.keys())

    # cells, AKA partition elements
    cells = list(ep.items())

    # find and record which submatrices are internally cohesive
    # (we need only check the top right half of the matrix, since it is symmetric)
    end1 = 0
    for i in range(len(cells)):
        start1 = end1
        end1 = start1 + len(cells[i][1])
        end2 = end1
        for j in range(i + 1, len(cells)):
            start2 = end2
            end2 = start2 + len(cells[j][1])

            # isolate submatrix of connections between two partition elements
            submatrix = adjMat[start1:end1,start2:end2]
            # for undirected weighted:
            # externally_consistent = np.all(submatrix == submatrix[0,0]) # len(set(submatrix)) == 1 # submatrix.all([submatrix[0, 0]])
            all_ones = submatrix.all()
            all_zeros = not submatrix.any()

            # if it is internally cohesive, place both partition elements in the same LEP
            if not all_ones and not all_zeros:
                partition_element = min(cells[i][0], cells[j][0])
                curr = j if cells[i][0] < cells[j][0] else i
                # update back pointers RTL
                next = int_cohesion_list[curr]
                while next != curr:
                    int_cohesion_list[curr] = partition_element
                    curr = next
                    next = int_cohesion_list[curr]
                # one more update needed once we have reached the leftmost partition element (when next == curr)
                int_cohesion_list[curr] = partition_element

            # if print_subgraphs:
                # report = np.array2string(submatrix) + "\nAll 1s: {}\nAll 0s: {}".format(allOnes, allZeros)
                # printWithLabel("SUBGRAPH OF PARTITIONS {} AND {}".format(cells[i][0], cells[j][0]), '*', report)

    if verbose:
        printWithLabel("INTERNAL COHESION LIST", '#', int_cohesion_list)

    # consolidate pointers to make the implicit tree structure in internalCohesionList one level deep at most
    #   (in other words, update back pointers LTR)
    for i in range(len(int_cohesion_list)):
        int_cohesion_list[i] = int_cohesion_list[int_cohesion_list[i]]

    if verbose:
        printWithLabel("CONDENSED INTERNAL COHESION LIST", '#', int_cohesion_list)

    # this list sorts the partitions by their internal cohesion groups, while 
    #   preserving the indices to determine which parititon elements are together
    #   (sorting is O(n log n), whereas finding groupings without sorting is O(n^2), worst case)
    lep_list = sorted(enumerate(int_cohesion_list), key=lambda x: x[1])

    if verbose:
        printWithLabel("SORTED PARTITION ELEMENT GROUPINGS", '#', lep_list)
    
    leps = []

    # combine lep_list elements into their proper groupings
    i = 0
    while i < len(lep_list):
        j = i + 1
        while j < len(lep_list) and lep_list[i][1] == lep_list[j][1]:
            j += 1
        # add each partition element number to an lep set
        leps.append(set([item[0] for item in lep_list[i:j]]))
        i = j
    
    return leps

def PartitionAristotle(part_dict):
    """Aristotle was the great organizer, and fittingly, so is this function.
    It creates the permutation matrices necessary to group the partitions of the
    coarsest equitable partition in the adjacency matrix
   
    ARGUMENTS:
        part_dict (dict): contains lists of the vertices belonging to each
        partition element
     
    RETURNS:
        P (array): the right bound permutation matrix
        Pinv (array): the left bound permutation matrix
    """
    # get the size of the permutation matrices by
    # checking how many nodes are in the graph
    Psize = max([max(partElement) for partElement in part_dict.values()]) + 1

    P = np.zeros((Psize, Psize))    # create the zero matrix to fill    
    I = np.eye(Psize, Psize)        # create identity as your pantry to pull from
                                    # to fill the columns of the zero matrix
                           
    col = 0
    for part_element in part_dict.values():     # itertate through partition elements
        for vertex in part_element:             # iterate through vertices in partition elements
            P[:,col] = I[:,vertex].copy()       # replace next column with appropriate part of I
            col += 1                            # make sure you're getting the next column in the next iteration
       
    Pinv = np.linalg.inv(P)   # create P inverse
    return Pinv, P

def old_main():
    adj_mat = graphs.getDolores()

    G, ep = getEquitablePartition(adj_mat)
    p, p_inv = PartitionAristotle(ep)
    permuted_adj_mat = np.matmul(p, np.matmul(adj_mat, p_inv))

    leps = getLocalEquitablePartitions(ep, permuted_adj_mat)

    print()
    printWithLabel("ADJACENCY MATRIX", '=', adj_mat)
    printWithLabel("COARSEST EQUITABLE PARTITION", '=', ep)
    printWithLabel("PERMUTED ADJACENCY MATRIX", '=', permuted_adj_mat)
    printWithLabel("LOCAL EQUITABLE PARTITIONS", '=', leps)

    graphWithColoredPartEl(permuted_adj_mat,ep)

def getEquitablePartition(adjacency_matrix):
    """This function gets the coarsest equitable partition of a graph.
   
    ARGUMENTS:
        adjacency_matrix: the adjacency matrix of the graph in question.
     
    RETURNS:
        A dictionary with numbered partition elements mapped to sets of
            the verticies included in each partition element.
    """
    G = nx.Graph(adjacency_matrix)
    C, N = ep_finder.initialize(G)
    ep, N = ep_finder.equitablePartition(C, N)
    return G, ep

# LEP Finder before we added support for directed graphs:

def getLocalEquitablePartitions(G, ep, progress_bar = True):
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyzed
        ep : dict
            The equitable partition of the graph, as returned by ep_finder
        progress_bar : boolean
            whether to show realtime progress bar (enabled by default)
    
    RETURNS:
        A list of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """

    rev_g = G.reverse() if G.is_directed() else G

    N = [ep_finder.Node(node, 0, list(rev_g.neighbors(node))) for node in G.nodes()]

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
    
    # "2d array" mapping partition elements to the edges connecting them; i.e.,
    #   edge_partition[0][3] should be a set of edges connecting partition elements
    #   0 and 3; implementation using dictionary with (i, j) pair key to reduce temporal complexity
    edge_partition = {}

    # the following loop takes about 76 percent of the LEP algorithm's runtime, so we should update the progress
    #   bar 76 times during the 
    num_edges_per_percent = G.number_of_edges() / 76

    # populate edge_partition
    for edge_num, (i, j) in enumerate(G.edges):
        if progress_bar and num_edges_per_percent != 0 \
                and edge_num % math.ceil(num_edges_per_percent) == 0:
            updateLoadingBar(progress + edge_num / num_edges_per_percent)
        part_i = partition_dict[i]
        part_j = partition_dict[j]
        if G.is_directed():
            key = (part_i, part_j)
        # in the undirected case, fill the "top right" half (i.e., i < j in key (i, j)) of edge_partition
        #   (bottom half is redundant for undirected graphs)
        else:
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

    edge_partition_num = 0
    edge_partition_el_per_percent = len(edge_partition) / 18

    # i and j are indices of the two partition elements in question
    for ((i, j), edge_set) in edge_partition.items():
        edge_partition_num += 1
        if progress_bar and edge_partition_el_per_percent \
                and edge_partition_num % math.ceil(edge_partition_el_per_percent) == 0:
            updateLoadingBar(progress + edge_partition_num / edge_partition_el_per_percent)
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
    

def updateLoadingBar(progress):
    pass

def compute():
    # a = 0
    # for i in range(10000):
    #     a += 1
    #     yield i
    # # print(a)
    # yield -3
    yield 1
    yield 2
    yield 3

def yieldTest():
    for _ in range(3):
        for i in compute():
            print('{0}'.format(i), end=' ')
    print(compute())

               

def updateCohesionGroups(i, int_cohesion_list):
    # find current root of tree of which i is a leaf
    curr = i
    next = int_cohesion_list[curr]
    node_stack = []
    while curr != next:
        node_stack.append(next)
        curr = next
        next = int_cohesion_list[curr]
    # now curr is the root/tail of the linked list
    while node_stack.count() > 0:
        int_cohesion_list[node_stack.pop()] = curr
    int_cohesion_list[i] = curr
            


def __computeLocalEquitablePartitions_old(N, pi):
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

    # keeps track of which partition elements are stuck together by internal cohesion
    #   i.e., if the values at two indices are the same, the partition elements at those 
    #   two indices are stuck together in the adjacency matrix (AKA they are not externally
    #   consistent)
    int_cohesion_list = list(pi.keys())

    for (index, V) in pi.items():
        common_neighbors = set(N[V[0]])
        for v in V:
            common_neighbors.intersection_update(set(N[v]))
        yield
        for v in V:
            for unique_neighbor in set(N[v]).difference(common_neighbors):
                i = index
                j = partition_dict[unique_neighbor]
                # updates int_cohesion_list so that i and j point directly to the root of the tree,
                #   thereby preventing the possibility of rightward-pointing links
                #   (rightward links are possible if i or j are out of date)
                # updateCohesionGroups(i, int_cohesion_list)
                # updateCohesionGroups(j, int_cohesion_list)
                # only coalesce cohesion groups if they are different
                if int_cohesion_list[i] != int_cohesion_list[j]:
                    # partition_element = min(int_cohesion_list[i], int_cohesion_list[j])
                    # curr = j if int_cohesion_list[i] < int_cohesion_list[j] else i
                    cohesion_group = int_cohesion_list[i] # cohesion_group = i would also work, but may be less efficient
                    curr = j
                    # update back pointers RTL
                    next = int_cohesion_list[curr]
                    while next != curr:
                        int_cohesion_list[curr] = cohesion_group
                        curr = next
                        next = int_cohesion_list[curr]
                    # one more update needed once we have reached the leftmost partition element (when next == curr)
                    int_cohesion_list[curr] = cohesion_group
        yield

    # consolidate pointers to make the implicit tree structure in internalCohesionList one level deep at most
    #   (in other words, update back pointers LTR)
    for i in range(len(int_cohesion_list)):
        int_cohesion_list[i] = int_cohesion_list[int_cohesion_list[i]]

    # this list sorts the partitions by their internal cohesion groups, while 
    #   preserving the indices to determine which parititon elements are together
    lep_list = enumerate(int_cohesion_list)
    lep_dict = dict()
    for (node, part_el) in lep_list:
        # add len(N) to key value so as to avoid clashing keys when relabeling (below)
        if part_el not in lep_dict:
            lep_dict.update({part_el + len(N): set()})
        lep_dict.get(part_el + len(N)).add(node)
        yield

    # for convenience, relabel keys to integers [0, len(int_cohesion_list) - 1]
    for (index, key) in enumerate(lep_dict.keys()):
        lep_dict[index] = lep_dict.pop(key)

    yield lep_dict

# TODO: verify correctness (at least run on a few more graphs)
# consider initializing LEP finder as well to make algorithm implementation independent of NetworkX
# verify time complexity (again, because algorithm has been changed)
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

    progress = 0
    if progress_bar:
        print("FINDING LEPS...")
        updateLoadingBar(progress)


    # array that maps nodes (indices) to their partition element
    partition_dict = np.empty(len(N), int)
    for (element, nodes) in ep.items():
        for node in nodes:
            partition_dict[node] = element

    # the preceding portion of the code generally takes about 2% of the total LEP runtime
    progress = 2
    if progress_bar:
        updateLoadingBar(progress)

    # keeps track of which partition elements are stuck together by internal cohesion
    #   i.e., if the values at two indices are the same, the partition elements at those 
    #   two indices are stuck together in the adjacency matrix (AKA they are not externally
    #   consistent)
    int_cohesion_list = list(ep.keys())

    edge_partition_num = 0
    edge_partition_el_per_percent = len(ep) / 94

    for (index, V) in ep.items():
        if progress_bar and edge_partition_el_per_percent \
                and edge_partition_num % math.ceil(edge_partition_el_per_percent) == 0:
            updateLoadingBar(progress + edge_partition_num / edge_partition_el_per_percent)

        common_neighbors = set(N[V[0]])
        for v in V:
            common_neighbors.intersection_update(set(N[v]))
        for v in V:
            for unique_neighbor in set(N[v]).difference(common_neighbors):
                i = index
                j = partition_dict[unique_neighbor]
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

    return lep_dict






REV_LABEL_ATTR = 'relabel_reverse_mapping'

def reverseRelabel(G, ep_dict=None, lep_list=None):
    
    if hasattr(G, REV_LABEL_ATTR):
        unmapping = getattr(G, REV_LABEL_ATTR)
        nx.relabel_nodes(G, unmapping, copy=False)
        if ep_dict is not None:
            for i in ep_dict.keys():
                ep_dict[i] = {unmapping[node] for node in ep_dict[i]}
        if lep_list is not None:
            for i in range(len(lep_list)):
                lep_list[i] = {unmapping[node] for node in lep_list[i]}
        
    else:
        raise Exception("Graph must be relabeled before being reverseRelabeled")
    

def relabel(G):
    # unmapping = {index: label for index, label in enumerate(G.nodes())}
    # setattr(G, "relabel_reverse_mapping", unmapping)
    # if mapping is None:
    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    nx.relabel_nodes(G, mapping, copy=False)





