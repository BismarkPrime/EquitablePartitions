# imports to make code sensible
import graphs
import networkx as nx

# EP_UTILS:
from ep_utils import *

def _getEigenvaluesSparseParallel(csc: sparse.csc_array, csr: sparse.csr_array, pi: Dict[int, List[int]], leps: List[List[int]]) -> Tuple[List[float | complex], List[float | complex]]:
    divisor_matrix = getDivisorMatrixSparse(csc, pi)
    
    pool = ThreadPool(2)
    
    # locals_lists = pool.starmap_async(getLocals, [(csr, divisor_matrix, pi, lep) for lep in leps])
    globals = pool.apply_async(getGlobals, (divisor_matrix,))
    locals = pool.apply_async(getLocals, (csr, divisor_matrix, pi, leps))
    # locals = list(itertools.chain.from_iterable(locals_lists.get()))

    return globals.get(), locals.get()

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

def getTransceivingEP2(G: nx.Graph | nx.DiGraph,sparse_alg=False) -> dict[int, set[int]]:
    """
    Finds the transceiving equitable partition of a graph.
   
    ARGUMENTS:
        G : NetworkX Graph
            The graph to analyze
        sparse_alg (bool): whether or not the graph being passed in is a sparse matrix
    
    RETURNS:
        The transceiving equitable partition (dict; int -> set)
    """

    while True:
        # 1. get transmitting equitable partition
        N = ep_finder2.initFromNx(G,sparse_alg = sparse_alg)
        ep = ep_finder2.equitablePartition(N, progress_bar=False)
        # 2. get receiving equitable partition
        # if graph is undirected, the transceiving equitable partition is the same as the transmitting
        return ep, N

"""Code from mpi4py implementation that might be neede later but probably won't be, this is me emulating Peter because he is a bruv"""
"""
    ################################### THIS SECTION IS FOR PARALLELIZATION ONLY ##############################
    if parallel:
        # initialize MPI
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        total_threads =  comm.Get_size()
        current_thread = comm.Get_rank()
        if verbose: print("\n\nINITIALIZING PARALLELIZED LEParD ALGORITHM CALCULATION.\n\n")

        # scatter the lep_list (the pieces of the lep_list on each individual node are now called lep_visitor)
        lep_visitor = [comm.scatter(lep_list+[{'div'}],root=0)] # needs to be in list for syntax reasons
        # run the algorithm like normal to get the pieces on this thread
        for i, lep in enumerate(lep_list):
            if i%5 == 0:
                if verbose: print(f"\n\nanalyzing lep: {lep} on thread: {current_thread}.\n\n")
            node_list = []
            temp_ep_dict = {}

            for partElInd, partEl in enumerate(lep):
                if partEl == 'div':
                    total_spec += list(nx.adjacency_spectrum(graphs.genDivGraph(G,ep_dict)))
                else:
                    node_list =+ ep_dict[partEl]
                    temp_ep_dict[partElInd] = ep_dict[partEl]

            subgraph = nx.subgraph(G,node_list)
            total_spec += list(nx.adjacency_spectrum(subgraph))
            div_specs += list(nx.adjacency_spectrum(graphs.genDivGraph(subgraph,temp_ep_dict)))

            #TODO: find a way to scatter the job of getting the divisor matrix eigenvalues as well.
            # Figure out how to gather in the way that I want.


    ##############################################################################################################"""

if __name__ == '__main__':
    bertha = graphs.genBerthaSparse(1000)
    berthaNx = nx.from_numpy_array(bertha.todense())
    eigs_sparse = getEigenvaluesSparse(bertha)
    eigs_nx = getEigenvaluesNx(berthaNx)

# EP_FINDER:
def initFromFile(file_path: str, num_nodes: int=None, delim: str=',', 
                 comments: str='#', directed: bool=False, rev: bool=False) -> Dict[Any, Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    file_path   : the path to the file storing edge data of the graph to be 
                    analyzed
    num_nodes   : the total number of nodes; only necessary if file_path does 
                    not contain all nodes (i.e., if there are nodes with no 
                    edges between them); if num_nodes is provided, it is assumed 
                    that nodes are labeled with integers (zero-indexed)
    delim       : the delimiter between source and destination nodes for each 
                    edge in the file at file_path; uses ',' by default
    comments    : a character used to denote a comment, or line to ignore; uses 
                    '#' by default
    directed    : a boolean indicating whether the graph is directed or not; 
                    assumes undirected by default

    Returns
    -------
    N   : a list of Node objects representing the nodes of the graph described 
            in file_path
    
    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    # TODO: update this to create a list (not dict) 
    #                   -or-
    # remove it entirely and only use sparse initialization
    # KEEP IN MIND THAT ORIGINALLY THIS WAS A LIST, BUT IT CAUSED ISSUES BECAUSE NOT 
    # ALL VERTICES WERE CONSECUTIVE INTEGERS. IF WE CONTINUE TO INIT FROM FILE, WE MUST 
    # VERIFY THAT THESE CONDITIONS ARE MET. PROBABLY EASIER JUST TO MAKE SPARSE AND USE
    # INIT FROM SPARSE

    N = dict()
    
    with open(file_path, 'r') as f:
        for line in f:
            if not comments.isspace():
                line = line.strip()
            if line[0] != comments:
                line = line.split(delim)
                src = int(line[0])
                dest = int(line[1])
                if rev:
                    src, dest = dest, src
                if src not in N:
                    N[src] = Node(src, 0, [])
                N[src].successors.append(dest)
                if dest not in N:
                    N[dest] = Node(dest, 0, [])
                # in the undirected case, add out edge in the other direction as well
                if not directed:
                    N[dest].successors.append(src)
    
    if num_nodes is not None:
        for node in range(num_nodes):
            if node not in N:
                N[node] = Node(node, 0, [])

    return N


# LEP_FINDER:

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