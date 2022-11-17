
from functools import reduce
import os
from ep_utils import printWithLabel
import ep_utils
from sys import maxsize as MAX_INT
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import signal
import pickle
from scipy.stats import iqr
import numpy as np
from typing import Iterable, List, Tuple
import sys
import networkx as nx

class EPData:

    def __reset(self) -> None:
        self.ep = None
        self.leps = None
        self.G = None
        self.directed = None
        self.num_nodes = None
        self.num_edges = None
        self.plt_num = None
        
    # file_path should only be none if we are loading from a file
    def __init__(self, file_path: str=None, num_nodes: int=None, delim: str=',', comments: str='#', directed: bool=False, progress_bars: bool=True, rev=False) -> None:
        if file_path is None:
            self.__reset()
        else:
            args = (file_path, num_nodes, delim, comments, directed, progress_bars, True, rev)
            self.ep, self.leps, N = ep_utils.getEquitablePartitionsFromFile(*args)
            self.G = {node: N[node].neighbors for node in N}
            self.directed = directed
            self.num_nodes = len(self.G)
            self.num_edges = sum([len(i) for i in self.G.values()])
            if not self.directed:
                self.num_edges //= 2

            self.plt_num = 0

    def saveToFile(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    def loadFromFile(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)
        self.plt_num = 0

    def plotHistogram(self, weighted: bool=False, xscale: str='linear', yscale: str='linear', ax: Axes=None, show: bool=True) -> Tuple[Figure, Axes, List]:
        
        plt.style.use("fivethirtyeight")
        ep_sizes = [len(i) for i in self.ep.values()]
        lep_sizes = [sum([len(self.ep[ep_index]) for ep_index in lep]) for lep in self.leps]
        nt_ep_sizes = [i for i in ep_sizes if i != 1]
        nt_lep_sizes = [i for i in lep_sizes if i != 1]
        # nt_ep_iqr = iqr(nt_ep_sizes)
        # nt_lep_iqr = iqr(nt_lep_sizes)
        # ep_size_set = set(ep_sizes)
        # lep_size_set = set(lep_sizes)
        
        if ax is None:
            f, ax = plt.subplots(layout='constrained')
        else:
            f = ax.figure(layout='constrained')
        
        max_nt_ep_size = max(nt_ep_sizes)
        min_nt_ep_size = min(nt_ep_sizes)
        w = (max_nt_ep_size - min_nt_ep_size) / np.power(len(nt_ep_sizes), 1/2)
        # w = 2 * nt_ep_iqr / np.power(len(nt_ep_sizes), 1/3)
        bin_vals, _, _ = plt.hist(
            [nt_ep_sizes, nt_lep_sizes],
            bins=np.arange(min_nt_ep_size, max_nt_ep_size + w, w), #max(max(ep_size_set), max(lep_size_set))
            label=["EP", "LEP"],
            edgecolor="black",
            weights=[nt_ep_sizes, nt_lep_sizes] if weighted else None)

        plt.legend()
        plt.title("{}Histogram of EP and LEP Sizes" \
                .format("Weighted " if weighted else ""))
        plt.xlabel("Element Size")
        plt.ylabel("Number of {}".format("Nodes" if weighted else "Elements"))
        plt.xscale(xscale)
        plt.yscale(yscale)
        # plt.ion()

        if show:
            f.show()
        return f, ax, bin_vals

    def getPercentNonTrivial(self) -> float:
        nt_nodes = reduce(
            lambda part_sum, curr: part_sum if len(curr) == 1 else part_sum + len(curr), self.ep.values(), 0)
        return nt_nodes * 100 / self.num_nodes
    
    def printStats(self, file=sys.stdout) -> None:
        # keep non-trivial parts of EP and LEPs
        f_ep = list(filter(lambda i: len(i) != 1, self.ep.values()))
        # here, non-trivial just means that there are multiple nodes in the LEP
        f_leps = list(filter(lambda i: len(i) != 1 or len(self.ep[list(i)[0]]) != 1, self.leps))
        # calculate how much is non-trivial
        partitionSize = lambda part_el: len(self.ep[part_el])
        # calculate number of non-trivial nodes
        nt_nodes = reduce(
            lambda part_sum, curr: part_sum + sum([partitionSize(i) for i in curr]),
            f_leps, 0)
        # calculate number of non-trivial nodes from EP (should be same as nt_nodes)
        nt_nodes2 = reduce(
            lambda part_sum, curr: part_sum + len(curr), f_ep, 0)
        # percentage of nodes that are non-trivial
        nt_percent = nt_nodes * 100 / self.num_nodes


        general = "Nodes: {}, Edges: {}, Edges/Node: {}".format(
            self.num_nodes, self.num_edges, self.num_edges / self.num_nodes)
        computational = "EPs: {}, LEPs: {}".format(
            len(self.ep), len(self.leps))
        dist_template = "{} - ({}, {}): {}"
        distribution = dist_template.format("DATA", "MIN", "MAX", "AVG")
        # calculate some basic stats about non-trivial parts
        ep_distribution = dist_template.format("\nEP", *self.__getEPStats(f_ep))
        lep_distribution = dist_template.format("\nLEP", *self.__getEPStats(f_leps))
        printWithLabel("GENERAL COMPUTATION", '=', general + '\n' + computational, file=file)
        printWithLabel("DISTRIBUTIONS", '*', distribution + ep_distribution + lep_distribution, file=file)
        printWithLabel("PERCENT NON-TRIVIAL", '#', "{} %".format(nt_percent), file=file)
    
    def __getEPStats(self, set_list: List[Iterable]) -> Tuple[int, int, float]:
        minSize = lambda min, curr: min if min < len(curr) else len(curr)
        maxSize = lambda max, curr: max if max > len(curr) else len(curr)
        sumSize = lambda part_sum, curr: part_sum + len(curr)
        min_len = reduce(minSize, set_list, MAX_INT)
        max_len = reduce(maxSize, set_list, 0)
        avg_len = reduce(sumSize, set_list, 0) / max(len(list(set_list)), 1)
        return min_len, max_len, avg_len

    # methods to interact with graph data

    def getEquitablePartition(self):
        return self.ep
    
    def getLocalEquitablePartitions(self):
        return self.leps
    
    def getGraphDict(self):
        return self.G

    def getNetworkX(self):
        return nx.DiGraph(self.G) if self.directed else nx.Graph(self.G)
    
def processData(file_path: str, data_name: str, num_nodes: int=None, delim: str=',', \
        comments: str='#', directed: bool=False, progress_bars: bool=True, rev: bool=False) -> None:
    data = EPData(file_path=file_path, num_nodes=num_nodes, delim=delim, \
        comments=comments, directed=directed, progress_bars=progress_bars, rev=rev)
    percent_nt = data.getPercentNonTrivial()
    file_name = file_path.split("/")[-1].split(".")[0]
    file_name = file_name + ", reversed" if rev else file_name
    dir_name = "[{:05.2f}%] {} ({})".format(percent_nt, data_name, file_name)
    os.chdir('results')
    os.mkdir(dir_name)
    os.chdir(dir_name)
    data.saveToFile("data.bin")
    data.printStats(file=open("stats.txt", "w"))

    # TODO: print general stats to a CSV file along with percent nt for correlation
    
    print("Plotting Histograms...", end=' ')
    figure, _, _ = data.plotHistogram(show=False)
    log_figure, _, _ = data.plotHistogram(show=False, xscale="log", yscale="log")
    weighted_figure, _, _ = data.plotHistogram(show=False, weighted=True)
    weighted_log_figure, _, _ = data.plotHistogram(show=False, weighted=True, xscale="log", yscale="log")
    figure.savefig("histogram.png")
    log_figure.savefig("log_histogram.png")
    weighted_figure.savefig("weighted_histogram.png")
    weighted_log_figure.savefig("weighted_log_histogram.png")

    plt.close(figure)
    plt.close(log_figure)
    plt.close(weighted_figure)
    plt.close(weighted_log_figure)

    os.chdir("../..")
    print("Done")

