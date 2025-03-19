import sys, os
from time import time
import pickle
import statistics
import shutil
import argparse
from tkinter import Tk
from tkinter import filedialog

from typing import NamedTuple, Dict, List, Any, Callable
from collections import Counter

from scipy import sparse, stats
import numpy as np
import networkx as nx
from line_profiler import profile

import ep_utils
import graphs
import ep_finder
import lep_finder

# TODO: consider tracking computation time for each metric

def profile2(fnc):
    """
    Profiles any function in following class just by adding @profile above function
    """
    import cProfile, pstats, io
    def inner (*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc (*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'   #Ordered
        ps = pstats.Stats(pr,stream=s).strip_dirs().sort_stats(sortby)
        n=10                    #reduced the list to be monitored
        ps.print_stats(n)
        #ps.dump_stats("profile.prof")
        print(s.getvalue())
        return retval
    return inner 

SUPPORTED_TYPES = ['csv','txt','graphml','gexf','json','edgelist','edges']
UNSUPPORTED_TYPES = []

class MetaMetrics(NamedTuple):
    m_source_file: str
    m_ep_file: str
    m_lep_file: str
    m_ep_time: float
    m_lep_time: float
    m_eig_time: float
    m_par_eig_time: float
    m_total_time: float
    m_np_eig_time: float

class GraphMetrics(NamedTuple):
    g_avg_node_degree: float
    g_order: int  # num nodes
    g_size: int  # num edges
    g_directed: bool
    g_density: float
    g_connected_components: int
    # NOTE: the following metrics were removed for being too computationally expensive
    # g_diameter: int
    # g_radius: int  # min eccentricity
    # g_average_path_length: float
    # g_edge_connectivity: int
    # g_vertex_connectivity: int
    # g_assortativity: float
    # g_clustering_coefficient: float
    # g_transitivity: float

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

def main(file_path: str, directed: bool, verify_eigenvalues: bool=True):
    print(f"Processing {file_path}...")

    m_source_file = file_path
    # 1a Get the graph as a sparse graph
    start_time = time()
    mat = getGraph(file_path, directed)
    print(f"Graph loaded in {time() - start_time} seconds")
    
    # 2. Compute desired graph metrics
    start_time = time()
    graph_metrics = getGraphMetrics(mat, directed)
    print(f"Graph metrics computed in {time() - start_time} seconds")
    start_time = time()
    
    # 3. Compute coarsest EP, save to file
    csr = mat.tocsr()
    csc = mat.tocsc()
    start_time = time()
    pi = ep_finder.getEquitablePartition(ep_finder.initFromSparse(csr))
    m_ep_time = time() - start_time

    print(f"Coarsest EP computed in {m_ep_time} seconds")
    
    # 4. Compute EP metrics
    ep_metrics = getEPMetrics(pi)
    
    # 5. Compute Monad Set of LEPs, save to file
    start_time = time()
    leps = lep_finder.getLocalEquitablePartitions(lep_finder.initFromSparse(csc), pi)
    m_lep_time = time() - start_time

    print(f"Monad set of LEPs computed in {m_lep_time} seconds")
    
    # 6. Compute LEP metrics
    lep_metrics = getLEPMetrics(leps, pi)
    
    # 7. Compute eigenvalues
    start_time = time()
    global_eigs, local_eigs = ep_utils._getEigenvaluesSparse(csc, csr, pi, leps)
    m_eig_time = time() - start_time
    m_total_time = m_ep_time + m_lep_time + m_eig_time

    print(f"Eigenvalues computed in {m_eig_time} seconds")

    start_time = time()
    global_eigs, local_eigs = ep_utils._getEigenvaluesSparseParallel(csc, csr, pi, leps)
    m_par_eig_time = time() - start_time

    print(f"Eigenvalues parallel computed in {m_par_eig_time} seconds")

    # 8. Store metrics in dataframe

    file_name = file_path[:file_path.rfind('.')].split('/')[-1]
    results_dir = f'Results/{file_name}'

    # remove and replace directory for results (consider warning the user also)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)

    m_ep_file = 'ep_data.pkl'
    m_lep_file = 'lep_data.pkl'

    with open(f"{results_dir}/{m_ep_file}", 'wb') as f:
        pickle.dump(pi, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{results_dir}/{m_lep_file}", 'wb') as f:
        pickle.dump(leps, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 9. Compute eigenvalues using numpy for speed comparison
    start_time = time()
    np_eigenvalues = np.linalg.eigvals(csr.toarray())
    m_np_eig_time = time() - start_time
    print(f"Numpy eigenvalues computed in {m_np_eig_time} seconds")

    # REMEMBER that par time is not currently included in total time
    meta_metrics = MetaMetrics(m_source_file, m_ep_file, m_lep_file, m_ep_time, m_lep_time, m_eig_time, m_par_eig_time, m_total_time, m_np_eig_time)

    metrics = [file_name] + list(meta_metrics) + list(graph_metrics) + list(ep_metrics) + list(lep_metrics)

    with open('MetricWarden.csv','a') as file:
        file.write(f'{','.join(map(str, metrics))}\n')

    # 10. Store metrics for DT
    computeMetricsForDT(pi, leps, global_eigs, local_eigs, file_name)

    # 11. Verify that the eigenvalues are correct
    if verify_eigenvalues:
        eigenvalues = global_eigs + local_eigs
        start_time = time()
        verifyEigenvalues(np_eigenvalues, eigenvalues)
        print(f"Eigenvalues verified in {time() - start_time} seconds")

def getGraph(file_path: str, directed: bool) -> sparse.sparray:
    file_extension = file_path.split('.')[-1]
    if file_extension in SUPPORTED_TYPES: 
        #TODO: make this an argparser
        visualize = 'visualize' in sys.argv
        return graphs.oneGraphToRuleThemAll(file_path, visualize=visualize, directed=directed)
    else:
        if file_extension in UNSUPPORTED_TYPES: print("This type is not yet supported. Maybe you could do it...")
        else: print("We haven't heard of that graph type. Or at least haven't thought about it... Sorry.")
        sys.exit(1)

def computeMetricsForDT(pi: Dict[int, List[Any]], leps: List[List[int]], globals: List[float | complex], locals: List[float | complex], name: str) -> None:
    category = input("Enter a category for this network (e.g., social, biological, etc.) > ")
    url = input("Enter a URL for this network (if available) > ")
    description = input("Enter a description for this network (~10 words) > ")

    stats_folder = f"DT_Stats/{name}_stats"
    os.makedirs(stats_folder, exist_ok=True)

    def writeCountsToFile(counts: Counter, filename: str) -> None:
        with open(f"{stats_folder}/{filename}", 'w') as f:
            for size, count in sorted(counts.items()):
                f.write(f"{size}: {count}\n")

    ep_sizes = Counter([len(part) for part in pi.values()])
    writeCountsToFile(ep_sizes, "ep_sizes.txt")
    lep_ep_sizes = Counter([len(lep) for lep in leps])
    writeCountsToFile(lep_ep_sizes, "lep_sizes_ep_elements.txt")
    lep_vertex_sizes = Counter([sum(len(pi[i]) for i in lep) for lep in leps])
    writeCountsToFile(lep_vertex_sizes, "lep_sizes_vertices.txt")

    with open(f"{stats_folder}/globals.txt", 'w') as f:
        for global_val in globals:
            f.write(f"{global_val}\n")
    with open(f"{stats_folder}/locals.txt", 'w') as f:
        for local_val in locals:
            f.write(f"{local_val}\n")
    with open(f"{stats_folder}/info.txt", 'w') as f:
        f.write(f"Category: {category}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Description: {description}\n")

def verifyEigenvalues(np_eigenvalues: List[float | complex], lepard_eigenvalues: List[float | complex]) -> bool:
    our_unique_eigs, their_unique_eigs = ep_utils.getSymmetricDifference(lepard_eigenvalues, np_eigenvalues)
    if len(our_unique_eigs) > 0:
        print(f"Error: Some eigenvalues are unique to the LEPARD eigenvalues")
        prompt = "Would you like to compare LEParD eigenvalues to numpy eigenvalues? (Y/n) > "
        view_eigs = input(prompt)[0].lower() != 'n'
        if view_eigs:
            print(f"LEParD eigenvalues: {our_unique_eigs}")
            print(f"Numpy eigenvalues: {their_unique_eigs}")

def try_or[T](func: Callable[[], T], default: T, expected_exc: Exception=Exception) -> T:
    try:
        return func()
    except expected_exc:
        return default

@profile
def getGraphMetrics(sparseMatrix: sparse.sparray, directed: bool) -> GraphMetrics:
    G = nx.from_scipy_sparse_array(sparseMatrix, create_using=nx.DiGraph if directed else nx.Graph)
    size = G.size()

    # Compute graph metrics
    avg_node_degree = G.number_of_edges() / G.number_of_nodes() * (1 if directed else 2)
    order = G.order()

    density = nx.density(G)
    connected_components = nx.number_connected_components(G.to_undirected(as_view=True))

    metrics = GraphMetrics(avg_node_degree, order, size, directed, density, connected_components)

    return metrics

def getEPMetrics(pi: Dict[int, List[Any]]) -> EPMetrics:
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
    parser = argparse.ArgumentParser(description="runs a graph through all outr metrics and test and stores"
                                     "The attributes in a .csv file called MatricWarden.csv")
    
    parser.add_argument("--directed","-d", action='store_true', help="Necessary if graph is directed.")
    parser.add_argument("--file", type=str, help="Path to the graph")
    args = parser.parse_args()

    if args.file is None:
        Tk().withdraw()
        args.file = filedialog.askopenfilename()

    main(args.file, args.directed)
