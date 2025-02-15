import numpy as np
import networkx as nx
from scipy import sparse
# from alive_progress import alive_bar

from typing import Any, List, Set, Dict, Generator
# imported to support type hint for initFromN
import ep_finder

# TODO: update naming to match paper

# POTENTIAL OPTIMIZATIONS:
#   Using Disjoint Set data structures to store partitions
#   Initialize using csv files

def initFromNx(G: nx.Graph | nx.DiGraph) -> Dict[Any, Set[Any]]:
    """Initializes the inverted neighbor dictionary required to compute leps.
    PAS's Code
    ARGUMENTS:
        G : The graph to analyzed
    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """

    # NOTE: N stores the in-edge neighbors, i.e. N[v] returns all nodes w with an edge w -> v.
    #    Thus, it is different than just calling G.neighbors(v); (hence, we use G.reverse())
    N = [set(G.predecessors(node) if G.is_directed() else G.neighbors(node)) for node in G.nodes()]
    return N

def initFromSparse(mat: sparse.csc_array) -> Dict[Any, Set[Any]]:
    """Initializes the inverted neighbor dictionary required to compute leps.
    
    ARGUMENTS:
        G : The graph to analyzed
    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """

    # ensure that matrix is in csc format (csr format will create an out-edge neighbor mapping)
    # mat = mat.tocsc()
 
    # NOTE: we should revert to using arrays/lists where possible instead of dictionaries to reduce spatial complexity
    N = [set(mat.indices[mat.indptr[i]:mat.indptr[i + 1]]) for i in range(mat.shape[0])]
    
    return N
        

def initFromFile(file_path: str, num_nodes: int=None, delim: str=',', 
                 comments: str='#', directed: bool=False, rev: bool=False) -> Dict[int, Set[int]]:
    """Initializes the inverted neighbor dictionary required to compute leps.
   
    ARGUMENTS:
        file_path : the path to the file storing edge data of the graph to be analyzed
        num_nodes : the total number of nodes; only necessary if the file at file_path
                        does not contain all nodes (i.e., if there are nodes with no edges between them)
        delim :     the delimiter between source and destination nodes for each edge in the
                        file at file_path; uses ',' by default
        comments :  a character used to denote a comment, or line to ignore; uses '#' by default
        directed :  whether the graph is directed; uses False by default

    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """
    N = dict()
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] != comments:
                # NOTE: we assume that the file is formatted as follows:
                #   source_node, destination_node
                if rev:
                    (dest, src) = line.split(delim)
                else:
                    (src, dest) = line.split(delim)
                src = int(src)
                dest = int(dest)
                if dest not in N:
                    N.update({dest: set()})
                N.get(dest).add(src)
                if src not in N:
                    N.update({src: set()})
                if not directed:
                    N.get(src).add(dest)
    # if there are nodes with no edges between them, we need to add them to N
    if num_nodes is not None:
        for i in range(num_nodes):
            if i not in N:
                N.update({i: set()})
    return N

def getLocalEquitablePartitions(N: Dict[Any, Set[Any]], ep: Dict[int, Set[Any]]) -> List[List[int]]:
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        N :     A dictionary containing nodes as keys with their in-edge neighbors as values
        ep :    The equitable partition of the graph, as returned by ep_finder
        progress_bar : whether to show realtime progress bar (disabled by default)
    
    RETURNS:
        A list of sets, with each set containing the indices/keys of partition elements
            that can be grouped together in the same local equitable partition
    """
    
    return __computeLocalEquitablePartitions(N, ep)

def __computeLocalEquitablePartitions(N: List[Set[int]], pi: Dict[int, List[Any]]) \
                                                              -> List[List[int]]:
    """Finds the local equitable partitions of a graph.
   
    ARGUMENTS:
        N :     A dictionary containing nodes as keys with their in-edge neighbors as values
        pi :    The equitable partition of the graph, as returned by ep_finder
    
    RETURNS:
        A list of sets, with each set containing the partition elements that can be
            grouped together in the same local equitable partition
    """

    # dict that maps nodes to their partition element
    partition_dict = np.empty(len(N), dtype=int)
    for i, V in pi.items():
        for node in V:
            partition_dict[node] = i

    # keeps track of which partition elements are stuck together by internal cohesion,
    #   with partition element index as key and internally cohesive elements as values
    lep_network = dict()

    for i, V in pi.items():
        common_neighbors = set(N[V[0]])
        for v in V:
            common_neighbors.intersection_update(N[v])
        for v in V:
            for unique_neighbor in set(N[v]) - common_neighbors:
                __link(i, partition_dict[unique_neighbor], lep_network)

    leps = __extractConnectedComponents(lep_network, len(pi))
    # convert to List of Lists to be consistent with EPFinder
    lep_list = [list(lep) for lep in leps]
    return lep_list

def __link(i: int, j: int, edge_dict: Dict[int, Set[int]]) -> None:
    if i not in edge_dict:
        edge_dict.update({i: set()})
    edge_dict.get(i).add(j)

    if j not in edge_dict:
        edge_dict.update({j: set()})
    edge_dict.get(j).add(i)

def __extractConnectedComponents(edge_dict: Dict[int, Set[int]], num_nodes: int) -> List[Set[int]]:
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