import numpy as np
import networkx as nx
import ep_finder
import matplotlib.pyplot as plt

def __main__():
    adj_mat = getDolores()

    G, ep = getEquitablePartition(adj_mat)
    
    p, p_inv = PartitionAristotle(ep)

    print()
    printWithLabel("ADJACENCY MATRIX", '=', adj_mat)
    printWithLabel("COARSEST EQUITABLE PARTITION", '=', ep)

    plotEquitablePartition(G, ep)

    permuted_adj_mat = np.matmul(p, np.matmul(adj_mat, p_inv))

    leps = getLocalEquitablePartitions(ep, permuted_adj_mat)

    printWithLabel("PERMUTED ADJACENCY MATRIX", '=', permuted_adj_mat)
    printWithLabel("LOCAL EQUITABLE PARTITIONS", '=', leps)

def printWithLabel(label, delim, item):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item))

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

def plotEquitablePartition(G, ep):
    nx.draw_networkx(G)
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
            allOnes = submatrix.all()
            allZeros = not submatrix.any()

            # if it is internally cohesive, place both partition elements in the same LEP
            if not allOnes and not allZeros:
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

            report = np.array2string(submatrix) + "\nAll 1s: {}\nAll 0s: {}".format(allOnes, allZeros)
            if print_subgraphs:
                printWithLabel("SUBGRAPH OF PARTITIONS {} AND {}".format(cells[i][0], cells[j][0]), '*', report)

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

def getDolores():
    #                 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    return np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # 1
                     [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], # 2
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 3
                     [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 4
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 5
                     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 6
                     [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 7
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 8
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 9
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 10
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # 11
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], # 12
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], # 13
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 14
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 15
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 16
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 17
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1], # 18
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], # 19
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # 20
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], # 21
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # 22
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], # 23
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]])# 24

def getDogbone():
    return np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
                     [0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                     [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]])