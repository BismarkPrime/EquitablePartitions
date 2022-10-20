import numpy as np
from alive_progress import alive_bar

# TODO: update naming to match paper

# POTENTIAL OPTIMIZATIONS:
#   Using Disjoint Set data structures to store partitions
#   Initialize using csv files

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

def initFromFile(file_path, num_nodes=None, delim=',', comments='#', directed=False):
    """Initializes the inverted neighbor dictionary required to compute leps.
   
    ARGUMENTS:
        file_path : the path to the file storing edge data of the graph to be analyzed
        num_nodes : the total number of nodes; only necessary if the file at file_path
            does not contain all nodes (i.e., if there are nodes with no edges between them)
        delim : the delimiter between source and destination nodes for each edge in the
            file at file_path; uses ',' by default
        comments : a character used to denote a comment, or line to ignore; uses '#' by default
        directed : whether the graph is directed; uses False by default

    
    RETURNS:
        A dictionary with nodes as keys and a set of their in-edge neighbors as values.
    """
    N = dict()
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] != comments:
                # NOTE: we assume that the file is formatted as follows:
                #   source_node, destination_node
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
    partition_dict = dict() #np.empty(len(N), int)
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

    leps = __extractConnectedComponents(lep_network, len(pi))
    yield leps

def __link(i, j, edge_dict):
    if i not in edge_dict:
        edge_dict.update({i: set()})
    edge_dict.get(i).add(j)

    if j not in edge_dict:
        edge_dict.update({j: set()})
    edge_dict.get(j).add(i)

def __extractConnectedComponents(edge_dict, num_nodes):
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