import sys,os
import pandas as pd
import numpy as np
from typing import NamedTuple, Dict, List, Any
from scipy import sparse
from time import time

import graphs as g
import helper as h
import ep_finder
import lep_finder

SUPPORTED_TYPES = ['csv','txt','graphml','gexf']
UNSUPPORTED_TYPES = ['json','edges']

class GraphMetrics(NamedTuple):
    avg_node_degree: int
    diameter: int
    order: int # num nodes
    size: int # num edges
    directed: bool
    # add other graph metrics here

class EPMetrics(NamedTuple):
    percent_nt_vertices: int # percent non-trivial vertices, (nt vertices / total vertices)
    percent_nt_elements: int # percent non-trivial elements, (nt elements / total elements)
    # add other EP metrics here
    
class LEPMetrics(NamedTuple):
    percent_vnt_elements: int # percent vertex-non-trivial LEPs, (LEPs with one vertex / total LEPs)
    percent_ent_elements: int # percent element-non-trivial LEPs, (LEPs with one element / total LEPs)
    # add other LEP metrics here

def main(file_path: str):
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
    
    # 4. Compute EP metrics
    ep_metrics = getEPMetrics(pi)
    
    # 5. Compute Monad Set of LEPs, save to file
    # (remember to track computation time for dataframe!)
    start_time = time()
    leps = lep_finder.getLocalEquitablePartitions(lep_finder.initFromSparse(csc), pi)
    lep_time = time() - start_time
    
    # 6. Compute LEP metrics
    lep_metrics = getLEPMetrics(leps)
    
    # 7. Compute eigenvalues
    
    # 8. Store metrics in dataframe

def getGraphMetrics(G: sparse.sparray) -> GraphMetrics:
    pass

def getEPMetrics(pi: Dict[int, List[Any]]) -> EPMetrics:
    pass

def getLEPMetrics(leps: List[List[int]]) -> LEPMetrics:
    pass

if __name__=="__main__":
    file_path = sys.argv[1]
    main(file_path)
