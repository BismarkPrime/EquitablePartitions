import cmath
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import reduce
import sys
import gc
from scipy import sparse
from scipy.sparse import linalg
from typing import Dict, List, Set, Tuple

from collections import Counter
import ep_finder, lep_finder
import ep_finder2
import graphs

# TODO: update naming to match paper
# TODO: use child processes for finding EP and LEP to release memory after computation.

EPSILON = 1e-6

# maybe start with NetworkX graphs for simplicity, then convert to adjacency dict
def getDivisorMatrixNx(G: nx.Graph | nx.DiGraph, pi: Dict[int, Set[int]]) -> sparse.base.spmatrix: # [int, set[int]]:
    """Finds the divisor matrix of a graph with respect to its coarsest equitable partition.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
        pi : dict
            The coarsest equitable partition of the graph, as returned by ep_finder
        leps : list
            The local equitable partitions of the graph, as returned by lep_finder
    
    RETURNS:
        The divisor matrix with weights representing the number of connections between LEPs (sparse)
    """
    # remember: relabeling ep nodes to be consecutive integers is necessary for the divisor matrix to be square
    #   but indexing into the divisor matrix with original ep numbers will no longer work
    #   so we need to keep track of the mapping between original ep numbers and new ep numbers
    # div_to_ep = []
    ep_to_div = {ep_id: div_index for div_index, ep_id in enumerate(pi.keys())}
    # label each node with its ep
    # for ep_id, ep in pi.items():
        # div_to_ep.append(ep_id)
        # ep_to_div[ep_id] = len(div_to_ep) - 1
    # create divisor matrix
    div_mat = np.zeros((len(pi), len(pi)), dtype=int)
        # sparse.dok_matrix((len(pi), len(pi)), dtype=int)
    # populate divisor matrix by sampling one node from each ep and counting 
    #   the number of connections between the sampled node and its neighbors
    for ep_id, ep in pi.items():
        node = ep[0]
        for neighbor in G.neighbors(node):
            neighbor_ep = G.nodes[neighbor]['ep']
            div_mat[ep_to_div[ep_id], ep_to_div[neighbor_ep]] += 1
    return div_mat

def getEigenStuffsNx(G: nx.Graph) -> Tuple[List[complex], Dict[int, List[complex]], Dict[int, List[complex]]]:
    """Finds the eigenvalues of the divisor matrix of a graph with respect to its coarsest equitable partition.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
        pi : dict
            The coarsest equitable partition of the graph, as returned by ep_finder
        leps : list
            The local equitable partitions of the graph, as returned by lep_finder
    
    RETURNS:
        A three-tuple of the (global) eigenvalues of the divisor matrix (list[complex]]), the eigenvalues 
            of the LEP submatrices (dict[int -> list[complex]]), and the eigenvalues of the divisor 
            matrices of the LEPs (dict[int -> list[complex]])
    """
    # hint: use numpy.linalg.eigvals
    # hint: use getDivisorMatrix to get the divisor matrix of the graph and of each LEP

    # np.linalg.eigvals(nx.adjacency_matrix(div))
    # PROBLEM: getDivisorMatrix returns a sparse scipy dok matrix, not sure which function to use to get eigenvalues
    # so that we can concatenate numpy arrays of complex eigenvalues

    # NOTE: using Counters here instead of lists for efficiency, but complex 
    #   eigenvalues can't be compared as easily this way
    # TODO:
    # 1. get EP and LEPs
    pi, leps = getEquitablePartitions(G, progress_bars=False)
    # label nodes with their ep element
    for ep_id, ep in pi.items():
        for node in ep:
            G.nodes[node]['ep'] = ep_id
    # 2. get divisor matrix of graph
    div = getDivisorMatrixNx(G, pi)
    # 3. get eigenvalues of divisor matrix of graph
    globals = list(np.linalg.eigvals(div))

    # 4. get divisor matrix of each LEP
    lep_globals = [] # dict()
    lep_locals = [] # dict()
    for i, lep in enumerate(leps):
        # 4a. induce subgraph of G on each LEP
        subgraph_nodeset = set() # reduce(lambda sum, x: sum.union(pi[x]), lep, set())
        pi_i = dict()
        for ep_el in lep:
            subgraph_nodeset.update(pi[ep_el])
            pi_i[ep_el] = pi[ep_el]
        subgraph = G.subgraph(subgraph_nodeset)
        # faster (supposedly) than the line below, because it doesn't have
        #   to iterate over all nodes in pi
        # pi_i = {k: v for k, v in pi.items() if k in lep}
        # 4b. get divisor matrix of induced subgraph
        subgraph_div = getDivisorMatrixNx(subgraph, pi_i)
        # 5. get eigenvalues of divisor matrix of each LEP
        lep_globals += list(np.linalg.eigvals(subgraph_div))
        # 6. get eigenvalues of LEP submatrices
        # print(nx.adjacency_matrix(subgraph))
        # lep_locals += list(np.linalg.eigvals(nx.adjacency_matrix(subgraph).todense()))
        # the line below should be identical; both present to verify that assumption
        lep_locals += list(nx.adjacency_spectrum(subgraph))
    # 7. return all three
    return globals, lep_locals, lep_globals

def getEigenvaluesNx(G: nx.Graph | nx.DiGraph) -> List[complex]:
    '''
    Extracts eigenvalues from graph using the complete equitable partitions method.
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze

    RETURNS:
        A list of the eigenvalues of the graph
    '''
    lifting_counter, lep_eigs, lep_globals = getEigenStuffsNx(G)
    # lifting_counter = Counter(lifting_eigs)
    # for i in lep_eigs.keys():
    lifting_counter += lep_eigs
    # use getSymmetricDifferenceMatching here if complex
    lifting_counter, _ = getSymmetricDifference(lifting_counter, lep_globals)
    return lifting_counter
    




    # NOTE: this version of the function accounts for complex eigenvalues, with the caveat that 
    #   it is not as efficient as the version above. It is left here for reference.
    # 1. get eigenvalues of divisor matrix of graph, LEP submatrices, and divisor matrices of LEPs
    lifting_eigs, lep_eigs, lep_globals = getEigenStuffsNx(G)
    lifting_eigs = np.sort_complex(lifting_eigs)
    for i in lep_eigs.keys():
        lep_eigs[i] = np.sort_complex(lep_eigs[i])
        lep_globals[i] = np.sort_complex(lep_globals[i])

    # print(f"Globals: {lifting_eigs}")
    # print(f"LEP locals: {lep_locals}")
    # print(f"LEP globals: {lep_globals}")

    # 2. remove eigenvalues from LEP divisor matrices from eigenvalues of LEP submatrices
    for i in lep_eigs.keys():
        lep_locals = lep_eigs[i]
        # iterate over each eigenvalue of the LEP divisor matrix
        # using a monotonic index to keep track of the index in the LEP submatrix
        #  works because the eigenvalues are sorted

        # TODO: optimize by keeping track of values and indicies in a dict
        local_index = 0
        for lep_global in lep_globals[i]:
            # find the corresponding eigenvalue from the LEP submatrix
            # this will throw an error if there is no corresponding eigenvalue in lep_locals,
            #   but that should never happen. If it does, we need to see and fix the bug.
            # check for equality within a small tolerance
            while not cmath.isclose(lep_locals[local_index], lep_global, abs_tol=EPSILON):
                local_index += 1
            # remove if equal
            lep_locals = np.delete(lep_locals, local_index)
            # decrement index because we just removed an element
            local_index -= 1
        # 3. concatenate all eigenvalues into one list
        lifting_eigs = np.concatenate((lifting_eigs, lep_locals))
    return lifting_eigs

def getSymmetricDifference(list1: List[complex], list2: List[complex]) -> Tuple[List, List]:
    '''
        Gets the symmetric difference of two lists. Returns two lists: list1 - list2,
        and list2 - list1
    '''
    skipIndices1 = set()
    skipIndices2 = set()
    res1 = []
    res2 = []
    for i, cnum1 in enumerate(list1):
        for j, cnum2 in enumerate(list2):
            if j in skipIndices2:
                continue
            if cmath.isclose(cnum1, cnum2, abs_tol=EPSILON):
                skipIndices1.add(i)
                skipIndices2.add(j)
                break

    for i, cnum in enumerate(list1):
        if i not in skipIndices1:
            res1.append(cnum)
    
    for j, cnum in enumerate(list2):
        if j not in skipIndices2:
            res2.append(cnum)
    return res1, res2

def getSymmetricDifferenceMatching(list1: List[complex], list2: List[complex]) -> Tuple[List, List]:
    '''
        Gets the symmetric difference of two lists of complex numbers using a bipartite 
        matching algorithm to find the maximum number of complex pairs (c1, c2), where 
        c1 is from list1 and c2 from list2, such that c1 and c2 are sufficiently close to
        be considered equal complex numbers given some floating point tolerance.
        Returns two lists: list1 - list2, and list2 - list1
    '''
    # 1. create a bipartite graph with edges between each complex number in list1 and all 
    #   complex numbers in list2 that are sufficiently close to be considered equal
    #   (within some floating point tolerance)
    sparse_graph = sparse.dok_matrix((len(list1), len(list2)), dtype=bool)
    for i, cnum1 in enumerate(list1):
        for j, cnum2 in enumerate(list2):
            if cmath.isclose(cnum1, cnum2, abs_tol=EPSILON):
                sparse_graph[i, j] = True
    # 2. find the maximum matching of the graph
    matches = sparse.csgraph.maximum_bipartite_matching(sparse_graph.tocsr(), perm_type='column')
    # 3. return the symmetric difference of the two lists, where the elements in the
    #   matching are removed from the lists
    skipIndices1 = set()
    skipIndices2 = set()
    for i, j in enumerate(matches):
        if j != -1:
            skipIndices1.add(i)
            skipIndices2.add(j)
    res1 = []
    res2 = []
    for i, cnum in enumerate(list1):
        if i not in skipIndices1:
            res1.append(cnum)
    for j, cnum in enumerate(list2):
        if j not in skipIndices2:
            res2.append(cnum)
    return res1, res2

def compareEigenvalues(G: nx.Graph) -> None:
    '''
    Compares the eigenvalues of the divisor matrix of a graph to the eigenvalues
    generated by equitable partition analysis. Prints the results to the console.

    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
    '''
    # TODO:
    # 1. get eigenvalues of the graph using the complete equitable partitions method
    cep_eigs = getEigenvaluesNx(G)
    # 2. get eigenvalues of the graph using networkx
    np_eigs = np.linalg.eigvals(nx.adjacency_matrix(G).toarray())

    # use getSymmetricDifferenceMatching for complex eigs
    rem_cep, rem_np = getSymmetricDifference(cep_eigs, np_eigs)

    if len(rem_cep) != 0 or len(rem_np) != 0:
        print(f"Eigenvalues do not match!\nUnique to CEP: \n{np.asarray(rem_cep)}\nUnique to NP: \n{np.asarray(rem_np)}")
        print(f"Eigenvalues do not match!\nCEP: \n{np.asarray(cep_eigs)}\nNP: \n{np_eigs}")
        return False
    return True
    # print(f"CEP eigenvalues: {cep_eigs}")
    # print(f"NX eigenvalues: {nx_eigs}")
    # 3. compare the two lists of eigenvalues within a small tolerance
    diff = False
    for i in range(len(cep_eigs)):
        if not cmath.isclose(cep_eigs[i], np_eigs[i], abs_tol=EPSILON):
            # return False
            print(f"Eigenvalues do not match! CEP: {cep_eigs[i]}, NX: {np_eigs[i]}")
            diff = True
    # if not diff:
    #     print("Eigenvalues match!")
    return not diff
    # return True

def getDivisorMatrix(N: dict[int, set[int]], pi: Dict[int, Set[int]]) -> sparse.base.spmatrix: # [int, set[int]]:
    """Finds the divisor matrix of a graph with respect to its coarsest equitable partition.
   
    ARGUMENTS:
        N : dict
            The graph to analyze, with nodes as keys and a set of their neighbors as values.
            (The inverse of N as returned by lep_finder.initialize)
        pi : dict
            The coarsest equitable partition of the graph, as returned by ep_finder
        leps : list
            The local equitable partitions of the graph, as returned by lep_finder
    
    RETURNS:
        The divisor matrix with weights representing the number of connections between LEPs (sparse)
    """
    pass

def getEigenvalues(N: dict[int, set[int]], pi: dict[int, set[int]], leps: list[set[int]]) \
        -> tuple[list[complex], dict[int, list[complex]], dict[int, list[complex]]]:
    """Finds the eigenvalues of the divisor matrix of a graph with respect to its coarsest equitable partition.
   
    ARGUMENTS:
        N : dict
            The graph to analyze, with nodes as keys and a set of their neighbors as values.
            (The inverse of N as returned by lep_finder.initialize)
        pi : dict
            The coarsest equitable partition of the graph, as returned by ep_finder
        leps : list
            The local equitable partitions of the graph, as returned by lep_finder
    
    RETURNS:
        A three-tuple of the (global) eigenvalues of the divisor matrix (list[complex]]), the eigenvalues 
            of the LEP submatrices (dict[int -> list[complex]]), and the eigenvalues of the divisor 
            matrices of the LEPs (dict[int -> list[complex]])
    """
    # hint: use numpy.linalg.eigvals
    # hint: use getDivisorMatrix to get the divisor matrix of the graph and of each LEP
    pass

def getTransceivingEP(G: nx.Graph | nx.DiGraph) -> dict[int, set[int]]:
    """
    Finds the transceiving equitable partition of a graph.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
    
    RETURNS:
        The transceiving equitable partition (dict; int -> set)
    """
    # find the transmitting EP; use it as an initial coloring when finding the receiving;
    #   use the resulting coloring for finding a transmitting EP; continue until stable
    # NOTE: this method increases complexity of finding directed (transceiving) EPs as 
    #   opposed to the undireced case. Perhaps ep_finder can be modified to account for 
    #   both trasceiving directed cases better?
    C1, C2 = None, None
    ep1, ep2 = None, None
    G_inv = G.reverse() if nx.is_directed(G) else G
    while True:
        # 1. get transmitting equitable partition
        C1, N1 = ep_finder.initialize(G, C2)
        ep1, N1 = ep_finder.equitablePartition(C1, N1, progress_bar=False)
        # 2. get receiving equitable partition
        # if graph is undirected, the transceiving equitable partition is the same as the transmitting
        if not nx.is_directed(G):
            return ep1, N1
        if ep1 == ep2:
            return ep1, N1
        C2, N2 = ep_finder.initialize(G_inv, C1)
        ep2, N2 = ep_finder.equitablePartition(C2, N2, progress_bar=False)
        if ep1 == ep2:
            return ep1, N1

def getTransceivingEP2(G: nx.Graph | nx.DiGraph) -> dict[int, set[int]]:
    """
    Finds the transceiving equitable partition of a graph.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
    
    RETURNS:
        The transceiving equitable partition (dict; int -> set)
    """

    while True:
        # 1. get transmitting equitable partition
        N = ep_finder2.initFromNx(G)
        ep = ep_finder2.equitablePartition(N, progress_bar=False)
        # 2. get receiving equitable partition
        # if graph is undirected, the transceiving equitable partition is the same as the transmitting
        return ep, N

def getEquitablePartitions(G, progress_bars = True, ret_adj_dict = False, rev = False,plot_graph=False):
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
    G = G if not rev else G.reverse()
    # start_time = time.time()
    # C, N = ep_finder.initialize(G2)
    # ep, N = ep_finder.equitablePartition(C, N, progress_bar=progress_bars)
    ep, N = getTransceivingEP2(G)
    # coarsest = time.time() - start_time
    # start_time = time.time()
    N_G = lep_finder.initialize(G)
    leps = lep_finder.getLocalEquitablePartitions(N_G, ep, progress_bar=progress_bars)
    # local = time.time() - start_time
    if ret_adj_dict:
        # this may no longer be applicable with a tranceiving equitable partition
        return ep, leps, {node: N[node].successors for node in N}
    if plot_graph:
        plotEquitablePartition(G,ep)
    return ep, leps

def getEquitablePartitionsFromFile(file_path, num_nodes=None, delim=',', comments='#', directed=False, progress_bars=True, ret_adj_dict=False, rev=False):
    """Finds the coarsest equitable partition and local equitable partitions of a graph.
   
    ARGUMENTS:
        file_path : the path to the file storing edge data of the graph to be analyzed
        num_nodes : the total number of nodes; only necessary if the file at file_path
            does not contain all nodes (i.e., if there are nodes with no edges between them)
        delim : the delimiter between source and destination nodes for each edge in the
            file at file_path; uses ',' by default
        comments : a character used to denote a comment, or line to ignore; uses '#' by default
        directed : a boolean indicating whether the graph is directed or not; uses False by default
    
    RETURNS:
        The equitable partition (dict; int -> set), local equitable partition (list of sets
            of partition elements grouped together), and computation time (when applicable)
    """
    C, N = ep_finder.initFromFile(file_path, num_nodes=num_nodes, delim=delim, comments=comments, directed=directed, rev=rev)
    ep, N = ep_finder.equitablePartition(C, N, progress_bar=progress_bars)
    
    N_G = lep_finder.initFromFile(file_path, num_nodes=num_nodes, delim=delim, comments=comments, directed=directed, rev=rev)
    leps = lep_finder.getLocalEquitablePartitions(N_G, ep, progress_bar=progress_bars)
    if ret_adj_dict:
        return ep, leps, {node: N[node].neighbors for node in N}
    return ep, leps

def plotEquitablePartition(G, pi, pos_dict = None):
    """Plots the equitable partition of a graph, with each element in its own color.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to be plotted
        pi : dict
            The equitable partition of the graph, as returned by ep_finder
        pos_dict : dict (optional)
            A dictionary mapping nodes to their x,y coordinates. Only used when a such
            values are available and meaningful (such as a random geometric graph).
    """
    # iterator over equidistant colors on the color spectrum
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(pi) + 1)))
    # stores the color for each node
    default_color = next(color)
    color_list = [default_color for _ in range(G.number_of_nodes())]
    # assign all vertices in the same partition element to the same color
    for V_i in pi.values():
        c = next(color)
        for vertex in V_i:
            color_list[vertex] = c
    
    # set plot as non-blocking
    plt.ion()

    nx.draw_networkx(G, pos=pos_dict, node_color=color_list)
    plt.show()
    # need to pause briefly because GUI events (e.g., drawing) happen when the main loop sleeps
    plt.pause(.001)

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

def printWithLabel(label, delim, item, file=sys.stdout):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item), file=file)

def GetSpectrumFromLEPs(G,progress_bars=False,verbose=False,fake_parallel=False):
    """Gets the spectrum of a graph using the decomposition by leps method"""
    total_spec = []
    div_specs = []

    ep_dict, lep_dict = getEquitablePartitions(G,progress_bars = progress_bars)
    

    if fake_parallel:lep_list = list(lep_dict[0])
    else: lep_list = list(lep_dict)

    for i, lep in enumerate(lep_list): # cycle through each lep                         ## COMPLEXITY: L, times for leps
        if i%1000==0:
            if verbose: print(f"{i} out of {len(list(lep_dict))}")
        node_list = []   # place to get all nodes in lep
        temp_ep_dict = {} # make a place for the original ep partitions
        
        # iterate through each partition element in that lep.
        for partElInd, partEl in enumerate(lep):                                ## COMPLEXITY: will sum to k, eventually will hit all partitions elements
            node_list += ep_dict[partEl] # after this loop node_list has all nodes in the lep
            temp_ep_dict[partElInd] = ep_dict[partEl] # make the temporary ep_dict   ## POSSIBLE ERROR: SHOULD BE ADDING TO DICTIONARY
                        
        subgraph = nx.subgraph(G,node_list)
        total_spec += list(nx.adjacency_spectrum(subgraph)) # gets subgraph spectrum            ## COMPLEXITY <= n(l-1)F
        div_specs += list(nx.adjacency_spectrum(graphs.genDivGraph(subgraph,temp_ep_dict))) # get div spec of those subgraphs

    # collect everything that could be in the spectrum
    if verbose: print('now getting total divisor spectrum')
    total_spec += list(nx.adjacency_spectrum(graphs.genDivGraph(G,ep_dict)))
    # place to store the actual spectum
    actual_spec = []
    if verbose: print('creating counter')
    # account for everything in both including repeats
    total_count = Counter(total_spec)
    if verbose: print('subtracting counter')
    total_count.subtract(div_specs)
    if verbose: print('returning spectrum')
    return list(total_count.elements())

def ValidateMethod(G):
    """runs our eigenvalue catching method and then makes sure that it matches the other method"""
    our_spec = np.round(np.array(GetSpectrumFromLEPs(G)),2)
    their_spec = np.round(nx.adjacency_spectrum(G),2)

    return Counter(our_spec) == Counter(their_spec)


