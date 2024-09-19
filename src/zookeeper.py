import sys,os
import pandas as pd
import numpy as np
from typing import NamedTuple, Dict, List, Any
from scipy import sparse
from time import time
import networkx as nx
from scipy import stats
import pickle
import statistics

import ep_utils
import graphs as g
# import helper as h
import ep_finder
import lep_finder

SUPPORTED_TYPES = ['csv','txt','graphml','gexf','json','edgelist']
UNSUPPORTED_TYPES = ['edges']

class MetaMetrics():
    def __init__(self,m_source_file: str = None, m_ep_file: str = None, m_lep_file: str = None, 
                 m_ep_time: float = None, m_lep_time: float = None, m_eig_time: float = None,
                 m_total_time: float = None):

        self.m_source_file = m_source_file
        self.m_ep_file = m_ep_file 
        self.m_lep_file = m_lep_file 
        self.m_ep_time = m_ep_time 
        self.m_lep_time = m_lep_time 
        self.m_eig_time = m_eig_time 
        self.m_total_time = m_total_time 


class GraphMetrics(NamedTuple):
    g_avg_node_degree: float
    g_diameter: int
    g_order: int  # num nodes
    g_size: int  # num edges
    g_directed: bool
    g_radius: int  # min eccentricity
    g_average_path_length: float
    g_edge_connectivity: int
    g_vertex_connectivity: int
    g_density: float
    g_connected_components: int
    g_assortativity: float
    g_clustering_coefficient: float
    g_transitivity: float

class EPMetrics(NamedTuple):
    ep_percent_nt_vertices: int  # percent of vertices in non-trivial equitable partition elements
    ep_percent_nt_elements: int  # percent of equitable partition elements that are non-trivial (size > 1)
    ep_num_elements: int
    ep_size_max: int  # size of the largest partition element
    ep_size_min: int  # size of the smallest partition element
    ep_size_avg: float
    ep_size_variance: float
    ep_size_std_dev: float
    ep_size_median: int
    ep_size_mode: int
    ep_size_range: int
    ep_size_iqr: int
    ep_size_skewness: float
    ep_size_kurtosis: float
    
class LEPMetrics(NamedTuple):
    lep_percent_vnt_leps: int  # percent of LEPs that are vertex-non-trivial (LEPs with more than one vertex)
    # NOTE: since vertices are in vertex-non-trivial LEPs iff they are in vertex-non-trivial EP elements,
    #       percent_vnt_vertices is the same as percent_nt_vertices in EPMetrics, so we don't need to repeat it here
    lep_percent_ent_leps: int  # percent of LEPs that are element-non-trivial (LEPs with more than one element)
    lep_percent_ent_vertices: int  # percent of vertices in element-non-trivial LEPs
    lep_num_leps: int
    lep_size_max: int  # size of the largest LEP (in terms of elements)
    lep_size_min: int  # size of the smallest LEP (in terms of elements)
    lep_size_avg: float
    lep_size_variance: float
    lep_size_std_dev: float
    lep_size_median: int
    lep_size_mode: int
    lep_size_range: int
    lep_size_iqr: int
    lep_size_skewness: float
    lep_size_kurtosis: float

def main(file_path: str):
    meta_metrics = MetaMetrics()
    meta_metrics.m_source_file = file_path
    # 1a Get the graph as a sparse graph
    tag = file_path.split('.')[-1]
    # type is supported
    if tag in SUPPORTED_TYPES: 
        #TODO: make this an argparser
        if 'visualize' in sys.argv: visualize = True
        else: visualize = False
        G = g.oneGraphToRuleThemAll(file_path,visualize=visualize)
    else:    # type is not
        if tag in UNSUPPORTED_TYPES: print("This type is not yet supported. Maybe you could do it...")
        else: print("We haven't heard of that graph type. Or at least haven't thought about it... Sorry.")
        sys.exit(1)
    
    # 2. Compute desired graph metrics
    graph_metrics = getGraphMetrics(G)
    
    # 3. Compute coarsest EP, save and time to file
    # (remember to track computation time for dataframe!)
    csr = G.tocsr()
    csc = G.tocsc()
    start_time = time()
    pi = ep_finder.getEquitablePartition(ep_finder.initFromSparse(csr))
    ep_time = time() - start_time
    meta_metrics.m_ep_time = ep_time

    ep_filepath = '../Results/'
    
    # 4. Compute EP metrics
    ep_metrics = getEPMetrics(pi)
    
    # 5. Compute Monad Set of LEPs, save to file
    # (remember to track computation time for dataframe!)
    start_time = time()
    leps = lep_finder.getLocalEquitablePartitions(lep_finder.initFromSparse(csc), pi)
    lep_time = time() - start_time
    meta_metrics.m_lep_time = lep_time
    
    # 6. Compute LEP metrics
    lep_metrics = getLEPMetrics(leps, pi)
    
    # 7. Compute eigenvalues
    start_time = time()
    eigenvalues = ep_utils.getEigenvaluesSparse(csr, pi, leps)
    eig_time = time() - start_time
    meta_metrics.m_eig_time = eig_time
    meta_metrics.m_total_time = ep_time + lep_time + eig_time
    
    # 7b. Verify that the eigenvalues are correct
    np_eigenvalues = np.linalg.eigvals(csr.toarray())
    our_unique_eigs, their_unique_eigs = ep_utils.getSetDifference(eigenvalues, np_eigenvalues)
    if len(our_unique_eigs) > 0:
        print(f"Error: Some eigenvalues are unique to the LEPARD eigenvalues")
        prompt = "Would you like to compare LEParD eigenvalues to numpy eigenvalues? (Y/n) > "
        view_eigs = input(prompt)[0].lower() != 'n'
        if view_eigs:
            print(f"LEParD eigenvalues: {our_unique_eigs}")
            print(f"Numpy eigenvalues: {their_unique_eigs}")
    
    # 8. Store metrics in dataframe
    df = pd.read_csv("MetricWarden.csv", index_col='Name')

    file_name = file_path.split('/')[-1].split('.')[0]
    network_path = '../Results/' + file_name

    # make a directory for the network in results
    os.mkdir(network_path)

    with open(network_path + '/ep_data.pkl','wb') as f:
        pickle.dump(pi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(network_path + '/lep_data.pkl','wb') as f:
        pickle.dump(leps, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_metrics.m_ep_file = 'ep_data.pkl'
    meta_metrics.m_lep_file = 'lep_data.pkl'

    new_info = list(meta_metrics) + list(graph_metrics) + list(ep_metrics) + list(lep_metrics)
    new_info = ','.join(map(str, new_info))

    with open('MetricWarden.csv','a') as file:
        file.write(new_info)

def getGraphMetrics(sparseMatrix: sparse.sparray) -> GraphMetrics:
    # Convert sparse matrix to NetworkX graph
    G = nx.from_scipy_sparse_array(sparseMatrix)

    # Compute graph metrics
    avg_node_degree = nx.average_degree_connectivity(G)
    diameter = nx.diameter(G)
    order = G.order()
    size = G.size()
    directed = nx.is_directed(G)
    radius = nx.radius(G)
    average_path_length = nx.average_shortest_path_length(G)
    edge_connectivity = nx.edge_connectivity(G)
    vertex_connectivity = nx.node_connectivity(G)
    density = nx.density(G)
    connected_components = nx.number_connected_components(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    clustering_coefficient = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    metrics = GraphMetrics(avg_node_degree, diameter, order, size, directed, radius, average_path_length,
                           edge_connectivity, vertex_connectivity, density, connected_components, assortativity,
                           clustering_coefficient, transitivity)

    return metrics

def getEPMetrics(pi: Dict[int, List[Any]]) -> EPMetrics:
    # Compute EP metrics
    num_elements = len(pi)
    sizes = np.array([len(pi[i]) for i in pi])
    size_max = sizes.max()
    size_min = sizes.min()
    size_avg = sizes.mean()
    size_variance = sizes.var()
    size_std_dev = sizes.std()
    size_median = statistics.median(sizes)
    size_mode = stats.mode(sizes).mode
    size_range = size_max - size_min
    q75, q25 = np.percentile(sizes, [75 ,25])
    size_iqr = q75 - q25
    size_skewness = stats.skew(sizes)
    size_kurtosis = stats.kurtosis(sizes)

    # Compute percent_nt_vertices
    nt_vertices = 0
    nt_elements = 0
    for i in pi:
        if len(pi[i]) > 1:
            nt_elements += 1
            nt_vertices += len(pi[i])
            
    percent_nt_vertices = nt_vertices / sizes.sum()
    percent_nt_elements = nt_elements / num_elements

    metrics = EPMetrics(percent_nt_vertices, percent_nt_elements, num_elements, size_max, size_min, size_avg,
                        size_variance, size_std_dev, size_median, size_mode, size_range, size_iqr, size_skewness,
                        size_kurtosis)

    return metrics

def getLEPMetrics(leps: List[List[int]], pi: Dict[int, List[Any]]) -> LEPMetrics:
    # Compute LEP metrics
    num_leps = len(leps)
    sizes = np.array([len(lep) for lep in leps])
    size_max = sizes.max()
    size_min = sizes.min()
    size_avg = sizes.mean()
    size_variance = sizes.var()
    size_std_dev = sizes.std()
    size_median = np.median(sizes)
    size_mode = stats.mode(sizes).mode
    size_range = size_max - size_min
    q75, q25 = np.percentile(sizes, [75 ,25])
    size_iqr = q75 - q25
    size_skewness = stats.skew(sizes)
    size_kurtosis = stats.kurtosis(sizes)

    vnt_leps = 0
    ent_leps = 0
    ent_vertices = 0
    for lep in leps:
        if len(lep) > 1:
            ent_leps += 1
            ent_vertices += sum(len(pi[i]) for i in lep)
        if len(pi[lep[0]]) > 1:
            vnt_leps += 1
    percent_vnt_leps = vnt_leps / num_leps
    percent_ent_leps = ent_leps / num_leps
    percent_ent_vertices = ent_vertices / sum(len(pi[i]) for i in pi)

    metrics = LEPMetrics(percent_vnt_leps, percent_ent_leps, percent_ent_vertices, num_leps, size_max, size_min,
                            size_avg, size_variance, size_std_dev, size_median, size_mode, size_range, size_iqr,
                            size_skewness, size_kurtosis)

    return metrics

if __name__=="__main__":
    file_path = sys.argv[1]
    main(file_path)
