import sys,os
import pandas as pd
import numpy as np
import graphs as g
import helper as h

SUPPORTED_TYPES = ['csv']
UNSUPPORTED_TYPES = ['txt','graphml','json','gexf','edges']



if __name__=="__main__":
    # get the graph as a sparse graph
    file_path = sys.argv[1]
    tag = file_path.split('.')[-1]
    # type is supported
    if tag in SUPPORTED_TYPES: G = g.oneGraphToRuleThemAll(file_path)
    else:    # type is not
        if tag in UNSUPPORTED_TYPES: print("This type is not yet supported. Maybe you could do it...")
        else: print("We haven't heard of that graph type. Or at least haven't thought about it... Sorry.")
        sys.exit(1)
    
    # Get relevant graph information
    # WRITE THIS
    
    # Store relevant graph information