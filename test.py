import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import random

import ep_finder
import lep_finder
import graphs

# TODO: update naming to match paper
# TODO: add LEP verification tests

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

# test code
def test():
    G = nx.random_geometric_graph(40, .15)
    pi, leps = lep_finder.getEquitablePartitions(G, False, False)
    lep_finder.plotEquitablePartition(G, pi, nx.get_node_attributes(G, "pos"))

# generates random geometric graphs until it finds ep_finder problems
def findBadEPs():
    i = 0
    while (True):
        G = nx.random_geometric_graph(40, .15, seed=i)
        if not validEpResults(G):
            print("Current seed: " + i)
            print("Press <Enter> to continue...")
            input()
        i += 1
        print("\r{}".format(i), end='')

def validEpResults(G):
    pi, leps = lep_finder.getEquitablePartitions(G, False, False)
    if not isPartition(pi, G):
        printWithLabel("PI IS NOT A PARTITION!!!", '=', pi)
        return False
    if not isEquitable(pi, G):
        print("PI IS NOT EQUITABLE!!!")
        pos_dict = {}
        for node in G.nodes:
            pos_dict.update({ node: node.pos })
        lep_finder.plotEquitablePartition(G, pi)
        return False
    return True

def isPartition(pi, G):
    vertex_count = np.ones(G.number_of_nodes())
    # verify that each vertex shows up exactly once in pi
    for V_i in pi.values():
        for vertex in V_i:
            vertex_count[vertex] -= 1
    
    return not np.any(vertex_count)

def isEquitable(pi, G):
    # create table for fast node-to-partition lookups
    partition_dict = np.empty(G.number_of_nodes(), int)
    for (element, nodes) in pi.items():
        for node in nodes:
            partition_dict[node] = element
    
    for V_i in pi.values():
        if len(V_i) > 1:
            # dict of partition_element -> number of connections
            rule = {}
            for i, vertex in enumerate(V_i):
                # construct rule
                if i == 0:
                    rule = getPartitionNeighbors(vertex, G, partition_dict)
                # test other vertices against the rule
                else:
                    conns = getPartitionNeighbors(vertex, G, partition_dict)
                    if conns != rule:
                        print(V_i)
                        print("last call{}getPartitionNeighbors({}, {}, part)".format('-'*30, vertex, G))
                        print("RULE: {}\nCONNS: {}".format(rule, conns))
                        return False
    return True

def getPartitionNeighbors(vertex, G, partition_dict):
    conns = {}
    for neighbor in G.neighbors(vertex):
        part_el = partition_dict[neighbor]
        if part_el not in conns:
            conns.update({ part_el: 0 })
        conns.update({ part_el: conns[part_el] + 1 })
    return conns

# FIXME: bad function name
def areLEPs(leps, G, pi):
    
    return True

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
    colorArr = [0 for i in range(max(ep.values())[0]+1)] #create place to store node colors
    index = 0    #start node index as 9
    for partEl in ep.values():   #cycle through each partition element
        color = random.randint(0,148)   #get a random color to assign to this partition element
        for i in range(len(partEl)):   #cycle through all nodes in partition element
            colorArr[index] = hexColors[color]   #assign each node that color
            index+=1   #use this to get different colorArr position each time
    
    nx.draw_networkx(nx.from_numpy_array(adj_mat),node_color=colorArr)    #graph it with colors

# outdated function to test runtime complexity
def complexityTest():
    num_nodes = list()
    ep_comp_time = list()
    lep_comp_time = list()
    for i in range(2000, 50000, 2000):
        num_nodes.append(i)
        G = nx.random_internet_as_graph(i)
        # func = lambda: getEquitablePartitions(G)
        # t = Timer(func)
        # computation_time.append(t.timeit(1))
        coarsest, local = lep_finder.getEquitablePartitions(G)
        ep_comp_time.append(coarsest)
        lep_comp_time.append(local)

    
    plt.scatter(num_nodes, ep_comp_time, color='b', label="ep_finder")
    plt.scatter(num_nodes, lep_comp_time, color='r', label="LEP Algorithm")
    plt.title("EP vs LEP Computation Time")
    plt.xlabel('Number of Nodes')
    plt.ylabel('Computation Time')
    plt.legend(loc="upper left")
    plt.show()

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