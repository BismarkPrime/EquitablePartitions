# imports to make code sensible
import graphs
import networkx as nx

# EP_UTILS:
from ep_utils import *

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
    bertha = graphs.GenBerthaSparse(1000)
    berthaNx = nx.from_numpy_array(bertha.todense())
    eigs_sparse = getEigenvaluesSparse(bertha)
    eigs_nx = getEigenvaluesNx(berthaNx)