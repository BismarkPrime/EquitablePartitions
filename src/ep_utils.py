import cmath
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce, wraps
import sys
import gc
import scipy.linalg
from scipy import sparse
import itertools
from typing import Dict, List, Set, Tuple
import graphs

from collections import Counter
import ep_finder, lep_finder
import graphs

# TODO: update naming to match paper
# TODO: use child processes for finding EP and LEP to release memory after computation.

EPSILON = 1e-4


@profile
def getEigenvaluesSparse(mat: sparse.sparray) -> List[float | complex]:
    # NOTE: despite using sparse matrices where possible, eigenvalue calculations 
    #   are still performed on dense matrices. If possible, it would be good to hook 
    #   into a library designed for finding eigenvalues of sparse matrices specifically
    # see https://scicomp.stackexchange.com/questions/7369/what-is-the-fastest-way-to-compute-all-eigenvalues-of-a-very-big-and-sparse-adja

    csr = mat.tocsr()
    csc = mat.tocsc()

    # 1. Find Coarsest Equitable Partition
    pi = ep_finder.getEquitablePartition(ep_finder.initFromSparse(csr))

    # 2. Find Global Eigenvalues
    #       a. Compute Divisor Matrix
    #       b. Calculate spectrum
    divisor_matrix = getDivisorMatrixSparse(csc, pi)
    globals = np.linalg.eigvals(divisor_matrix) # if hermetian, use eigvalsh here instead (and below)
    # print(f"{divisor_matrix.todense() = }")

    # 2. Find Monad LEP Set
    L = lep_finder.getLocalEquitablePartitions(lep_finder.initFromSparse(csc), pi)

    # 3. Find Local Eigenvalues
    #    For each LEP:
    #       a. Create subgraph
    #       b. Compute divisor graph of subgraph
    #       c. Calculate spectrum of subgraph, divisor graph
    #       d. Compute difference eigs(SG) - eigs(DG)
    locals = []
    for lep in L:
        nodes = []
        for V in lep:
            nodes.extend(pi[V])
        # skip iterations for which globals = locals
        if len(nodes) < 2:
            continue
        #TODO: consider if this logic can be optimized (is it faster to convert to csc if we only perform one column-slice?)
        subgraph = csr[nodes,:].tocsc()[:,nodes]
        divisor_submatrix = divisor_matrix[lep,:][:,lep]

        subgraph_globals = scipy.linalg.eigvals(divisor_submatrix)
        subgraph_locals = scipy.linalg.eigvals(subgraph.todense())

        locals.append(getSymmetricDifference(subgraph_locals, subgraph_globals)[0])
    
    spectrum = list(itertools.chain.from_iterable((globals, *locals)))
    # TODO: compare runtime with:
    # spectrum = []
    # map(spectrum.extend, (globals, *locals))

    return spectrum

@profile
def getDivisorMatrixSparse(mat_csc: sparse.csc_array, pi: Dict[int, List[int]]) -> sparse.sparray:

    node2ep = { node: i for i, V in pi.items() for node in V }
    div_mat = np.zeros((len(pi), len(pi)), dtype=int)

    for i, V in pi.items():
        node = V[0]
        neighbors = mat_csc.indices[mat_csc.indptr[node]:mat_csc.indptr[node + 1]]
        for neighbor in neighbors:
            div_mat[i, node2ep[neighbor]] += 1 # perhaps += weight for weighted graphs...
    
    return div_mat

@profile
def getDivisorMatrixNx(G: nx.Graph | nx.DiGraph, pi: Dict[int, Set[int]]) -> sparse.sparray: # [int, set[int]]:
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
    ep_to_div = {ep_id: div_index for div_index, ep_id in enumerate(pi.keys())}
    # create divisor matrix
    div_mat = np.zeros((len(pi), len(pi)), dtype=int)
    
    # label nodes with their ep element
    for ep_id, ep in pi.items():
        for node in ep:
            G.nodes[node]['ep'] = ep_id

    # populate divisor matrix by sampling one node from each ep and counting 
    #   the number of connections between the sampled node and its neighbors
    for ep_id, ep in pi.items():
        node = ep[0]
        predecessors = G.predecessors(node) if G.is_directed() else G.neighbors(node)
        for neighbor in predecessors:
            neighbor_ep = G.nodes[neighbor]['ep']
            div_mat[ep_to_div[ep_id], ep_to_div[neighbor_ep]] += 1
    return div_mat

@profile
def getEigenStuffsNx(G: nx.Graph | nx.DiGraph) -> Tuple[List[complex], Dict[int, List[complex]], Dict[int, List[complex]]]:
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

    # STEPS:
    # 1. get EP and LEPs
    pi, leps = getEquitablePartitions(G, progress_bars=False)
    
    # 2. get divisor matrix of graph
    div = getDivisorMatrixNx(G, pi)
    # 3. get eigenvalues of divisor matrix of graph
    globals = list(np.linalg.eigvals(div))

    # 4. get divisor matrix of each LEP
    lep_globals = []
    lep_locals = []
    for lep in leps:
        # 4a. induce subgraph of G on each LEP
        subgraph_nodeset = set() # reduce(lambda sum, x: sum.union(pi[x]), lep, set())
        for ep_el in lep:
            subgraph_nodeset.update(pi[ep_el])
        subgraph = G.subgraph(subgraph_nodeset)

        # 4b. get divisor matrix of induced subgraph
        if len(lep) < 2:
            # globals = locals for 1x1 lep, so skip this iteration
            continue

        subgraph_div = np.empty((len(lep), len(lep)), dtype=int)
        for i in range(len(lep)):
            for j in range(len(lep)):
                subgraph_div[i,j] = div[lep[i], lep[j]]

        # 5. get eigenvalues of divisor matrix of each LEP
        lep_globals += list(np.linalg.eigvals(subgraph_div))
        
        # 6. get eigenvalues of LEP submatrices

        # lep_locals += list(np.linalg.eigvals(nx.adjacency_matrix(subgraph).todense()))
        # the line below should be identical; both present to verify that assumption
        lep_locals += list(nx.adjacency_spectrum(subgraph))
    # 7. return all three
    return globals, lep_locals, lep_globals


@profile
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
    lifting_counter += lep_eigs
    # NOTE: set differences are non-trivial with floating point error, especially in the complex
    #   plane. In considering the complexity of the LEParD algorithm, note that this difference 
    #   may have to be performed as a bipartite-matching problem. Solved via max-flow, this can 
    #   theoretically be almost linear with respect to the number of edges in the bipartite graph,
    #   though most algorithms are significantly slower (https://en.wikipedia.org/wiki/Maximum_flow_problem)
    lifting_counter, _ = getSymmetricDifference(lifting_counter, lep_globals)

    return lifting_counter
    

def getSymmetricDifference(list1: List[complex], list2: List[complex]) -> Tuple[List, List]:
    '''
        Gets the symmetric difference of two lists. Returns two lists: list1 - list2,
        and list2 - list1
    '''
    # NOTE: since maximum bipartite matching is generally solved as a max-flow problem, it
    #   may be comparable in speed to this naive method (n^2), and is more robust to edge cases.
    #   (Note, however, that the current implementation in NetworkX is quite slow.)
    #   Consider defaulting to the bipartite method in getSymmetricDifferenceMatching
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
    #   (within some floating-point tolerance)
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

def compareEigenvalues(G: nx.Graph | nx.DiGraph) -> None:
    '''
    Compares the eigenvalues of the divisor matrix of a graph to the eigenvalues
    generated by equitable partition analysis. Prints the results to the console.

    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
    '''
    # STEPS:
    # 1. get eigenvalues of the graph using the complete equitable partitions method
    cep_eigs = getEigenvaluesSparse(nx.adjacency_matrix(G))
    # for checking the networkx implimentation, use
    # cep_eigs = getEigenvaluesNx(G)
    # 2. get eigenvalues of the graph using networkx
    np_eigs = np.linalg.eigvals(nx.adjacency_matrix(G).toarray())

    # use getSymmetricDifferenceMatching for slower, but more robust, performance
    rem_cep, rem_np = getSymmetricDifferenceMatching(cep_eigs, np_eigs)

    if len(rem_cep) != 0 or len(rem_np) != 0:
        print(f"Eigenvalues do not match!\nUnique to CEP: \n{np.asarray(rem_cep)}\nUnique to NP: \n{np.asarray(rem_np)}")
        print(f"Eigenvalues do not match!\nCEP: \n{np.asarray(cep_eigs)}\nNP: \n{np_eigs}")
        return False
    return True


def getEquitablePartitions(G, progress_bars=True, rev=False):
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

    ep = ep_finder.getEquitablePartition(ep_finder.initFromNx(G))

    N_G = lep_finder.initFromNx(G)
    leps = lep_finder.getLocalEquitablePartitions(N_G, ep, progress_bar=progress_bars)
    
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
    ep, N = ep_finder.getEquitablePartition(C, N, progress_bar=progress_bars)
    
    N_G = lep_finder.initFromFile(file_path, num_nodes=num_nodes, delim=delim, comments=comments, directed=directed, rev=rev)
    leps = lep_finder.getLocalEquitablePartitions(N_G, ep, progress_bar=progress_bars)
    if ret_adj_dict:
        return ep, leps, {node: N[node].neighbors for node in N}
    return ep, leps


def plotEquitablePartition(G, pi, pos_dict=None):
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
    
    if pos_dict is None:
        # other layout options include: random, circular, spiral, spring, kamada_kawai, etc
        pos_dict = nx.shell_layout(G)

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

# Joseph Methods

def GetSpectrumFromLEPs(G,partition_data=None,with_grand_divisor=True,progress_bars=False,verbose=False,fake_parallel=False,parallel=False):
    """Gets the spectrum of a graph using the decomposition by leps method
    ARGS:
    G (networkx graph): graph to get the eigenvalues of
    progress_bars (bool): tells whether or not to show the progress of finding the ep data if needed
    verbose (bool): whether or not to annotate the steps being taken during the process
    fake_parallel (bool): whether or not to simulate parallelization by only finding the eigenvalues of part of the lep_list
    parallel (bool): whether or not to use multiprocessing to finding the eigenvalues of the lep list.
    """
    total_spec = Counter()
    div_specs = Counter()
    if partition_data is None:
        ep_dict, lep_list = getEquitablePartitions(G,progress_bars = progress_bars)
    else:
        ep_dict, lep_list = partition_data
    
    # trim list if fake parallelization is wanted.
    if fake_parallel:lep_list = list(lep_list[0])

    for i, lep in enumerate(lep_list): # cycle through each lep                         ## COMPLEXITY: L, times for leps
        if i%1000==0: 
            if verbose: print(f"{i} out of {len(list(lep_list))}")
        lep_vals, div_vals = GetSingleSpectrumFromLEP(G,ep_dict,lep)
        total_spec += lep_vals
        div_specs += div_vals

    # collect everything that could be in the spectrum
    if with_grand_divisor:
        if verbose: print('now getting total divisor spectrum')
        total_spec += Counter(nx.adjacency_spectrum(graphs.genDivGraph(G,ep_dict)))
    # place to store the actual spectum
    if verbose: print('creating counter')
    # account for everything in both including repeats
    if verbose: print('subtracting counter')
    total_spec.subtract(div_specs)
    if verbose: print('returning spectrum')
    return total_spec #list(total_spec.elements())

def GetSingleSpectrumFromLEP(G,ep_dict,lep_set):
    """given an ep dictionary and a single set of leps calculates the spectrum of that lep
    ARGS:
    G (networkx graph): the graph from which we calculated the lep's. Eventually we want to make it
        possible for the code to run even if this is a sparse matrix
    ep_dict (dict): the dictionary mapping the numerical partition element index to the list of nodes
        in that partition element
    lep_set (set): the set containing the numerical partition element indices of a specific LEP
    ------------------
    RETURNS:
    lep_vals (counter object): counter object that contain the eigenvalues from the lep that lift
        into the original graph.
    div_vals (counter object): 
    """
    lep_vals,div_vals = Counter(),Counter()
    node_list = []   # place to get all nodes in lep
    temp_ep_dict = {} # make a place for the original ep partitions
    
    # iterate through each partition element in that lep.
    for partElInd, partEl in enumerate(lep_set):                                ## COMPLEXITY: will sum to k, eventually will hit all partitions elements
        node_list += ep_dict[partEl] # after this loop node_list has all nodes in the lep
        temp_ep_dict[partElInd] = ep_dict[partEl] # make the temporary ep_dict 

    subgraph = nx.subgraph(G,node_list)
    lep_vals += Counter(nx.adjacency_spectrum(subgraph))
    div_vals += Counter(nx.adjacency_spectrum(graphs.genDivGraph(subgraph,temp_ep_dict)))

    return lep_vals, div_vals

def ValidateMethod(G):
    """runs our eigenvalue catching method and then makes sure that it matches the other method"""
    our_spec = np.round(np.array(GetSpectrumFromLEPs(G)),2)
    their_spec = np.round(nx.adjacency_spectrum(G),2)

    return Counter(our_spec) == Counter(their_spec)

if __name__ == "__main__":
    G = nx.erdos_renyi_graph(800, .005, directed=True, seed=0)
    sparse_array = nx.adjacency_matrix(G)
    getEigenvaluesSparse(sparse_array)
    getEigenvaluesNx(G)