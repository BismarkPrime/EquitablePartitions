from pprint import pprint
from timeit import Timer
from typing import Iterable
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import timeit
from functools import partial

import ep_finder
import graphs
import ep_utils
from graphs import GetLocalSpec

# TODO: update naming to match paper
# TODO: add LEP verification tests

import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def testSlots():
    ns = list(range(250, 2750, 250))
    print("Generating Graphs...", end="\r")
    Gs = [graphs.genBertha(n) for n in ns]
    print("Computing Coarsest EP w/Dict...", end="\r")
    dict_times = [
        min(timeit.Timer(partial(runEP, G)).repeat(repeat=10, number=20)) / 10
        for G in Gs
    ]
    print("Computing Coarsest EP w/Slots...", end="\r")
    slot_times = [
        min(timeit.Timer(partial(runEPSlots, G)).repeat(repeat=10, number=20)) / 10
        for G in Gs
    ]
    print("Plotting Results...", end="\r")
    plt.plot(ns, list(zip(dict_times, slot_times)))
    plt.xlabel("Size of Bertha")
    plt.ylabel("Time to Compute Coarsest EP")
    plt.title("Slots vs Dicts Runtime in EP Finder")
    plt.legend(("Dictionary", "Slots"))
    plt.show()


def runEP(G):
    ep_finder.equitablePartition(*ep_finder.initialize(G), False)


def runEPSlots(G):
    # for testing, ep_finder1 had slots and ep_finder didn't. Since slots are faster, they have now been added to ep_finder
    # ep_finder1.equitablePartition(*ep_finder1.initialize(G), False)
    pass


# test code
def test(p=0.04, iters=500, nodes=40):
    for nodes in range(20, 160, 20):
        print(f"\nChecking graphs with {nodes} nodes")
        for i in range(iters):
            print(f"\r{i}", end="")
            G = nx.random_internet_as_graph(nodes, seed=i)
            if not ep_utils.compareEigenvalues(G):
                print("ERROR")
                if input() == "v":
                    print(nx.adjacency_matrix(G, dtype=int).todense())
                    ep_dict = ep_utils.getTransceivingEP(G)
                    ep_utils.plotEquitablePartition(G, ep_dict)
                    input()

    # G = nx.random_geometric_graph(40, .15)
    # pi, leps = lep_finder.getEquitablePartitions(G, False, False)
    # lep_finder.plotEquitablePartition(G, pi, nx.get_node_attributes(G, "pos"))


def testCorrectness(p=0.04, iters=3000, nodes=40):
    for nodes in range(20, 80, 20):
        print(f"\nChecking graphs with {nodes} nodes")
        for i in range(iters):
            print(f"\r{i}", end="")
            G = nx.erdos_renyi_graph(nodes, 3.2 / nodes, directed=True, seed=i)
            pi1 = list(ep_utils.getTransceivingEP(G)[0].values())
            pi2 = list(ep_utils.getTransceivingEP2(G)[0].values())
            for l in pi1:
                l.sort()
            for l in pi2:
                l.sort()
            pi1.sort()
            pi3 = sorted(pi2)
            if pi1 != pi3:
                print("ERROR")
                if input() == "v":
                    print(nx.adjacency_matrix(G, dtype=int).todense())
                    pprint(f"EP from ep_finder:\n{pi1}")
                    pprint(f"EP from ep_finder2:\n{pi3}")
                    pprint(f"EP2 orig: {pi2}")
                    input()


# function to test runtime complexity
def complexityTest():
    num_nodes = list()
    ep_comp_time = list()
    ep2_comp_time = list()
    for nodes in range(200, 4000, 200):
        print(f"\rComputing iteration w/ {nodes} nodes.", end="")
        num_nodes.append(nodes)
        # G = nx.erdos_renyi_graph(nodes, 2.4 / nodes, directed=True)
        G = nx.random_internet_as_graph(nodes)
        func = lambda: ep_utils.getTransceivingEP(G)
        func2 = lambda: ep_utils.getTransceivingEP2(G)
        t = Timer(func)
        t2 = Timer(func2)
        ep_comp_time.append(t.timeit(15))
        ep2_comp_time.append(t2.timeit(15))

    plt.scatter(num_nodes, ep_comp_time, color="b", label="ep_finder")
    plt.scatter(num_nodes, ep2_comp_time, color="r", label="ep_finder2")
    plt.title("EP vs EP2 Computation Time")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Computation Time")
    plt.legend(loc="upper left")
    plt.show()


def compareEPEigenvalues(G: nx.Graph | nx.DiGraph, pi, leps) -> None:
    spec_dict, div_spec, orig_spec = GetLocalSpec(G, pi, leps)
    print("EP Eigenvalues:")
    pprint(orig_spec)

    print("LEP Eigenvalues:")
    lep_eigs = []
    for key, value in list(spec_dict.items()):
        if "Divisor" not in key:
            lep_eigs.append(value)
    pprint(lep_eigs)
    print("Divisor Eigenvalues:")
    pprint(div_spec)


def getEPEigenvalues(G: nx.Graph | nx.DiGraph) -> Iterable[numpy.ndarray]:
    ep, leps = ep_utils.getEquitablePartitions(G, progress_bars=False)
    for lep in leps:
        if len(lep) > 1:
            lep = list(lep)
            for i in lep[1:]:
                ep[lep[0]] += ep[i]
                del ep[i]
    eigen_dict = {}
    for index, partition_element in ep.items():
        eigen_dict[index] = getEigenvalues(G.subgraph(partition_element))
    return eigen_dict


def getEigenvalues(G: nx.Graph | nx.DiGraph) -> Iterable[numpy.ndarray]:
    L = nx.to_numpy_array(G)
    return numpy.linalg.eigvals(L)


def findInterestingGraphs():
    while True:
        G = nx.gnp_random_graph(20, 0.1, directed=True)
        vals = test.getEPEigenvalues(G).values()
        for l in vals:
            if np.any(l):
                pprint(vals)
                input()
                break


# generates random geometric graphs until it finds ep_finder problems
def findBadEPs():
    i = 0
    while True:
        G = nx.gnp_random_graph(20, 0.14, seed=i, directed=False)
        G = graphs.randomRelabel(G)
        if not validEpResults(G):
            print("Current seed: {}".format(i))
            print("Press <Enter> to continue...")
            input()
        i += 1
        print("\r{}".format(i), end="")


def validEpResults(G):
    pi, leps = ep_utils.getEquitablePartitions(G, False, False)
    if not isPartition(pi, G):
        printWithLabel("PI IS NOT A PARTITION!!!", "=", pi)
        return False
    if not isEquitable(pi, G):
        print("PI IS NOT EQUITABLE!!!")
        print(pi)
        ep_utils.plotEquitablePartition(G, pi)
        return False
    return True


def isPartition(pi, G):
    # vertex_count = np.ones(G.number_of_nodes())
    vertices = set()
    # verify that each vertex shows up exactly once in pi
    for V_i in pi.values():
        for vertex in V_i:
            if vertex in vertices:
                return False
            vertices.add(vertex)

    return len(vertices) == G.number_of_nodes()


def isEquitable(pi, G):
    # create table for fast node-to-partition lookups
    partition_dict = dict()  # np.empty(G.number_of_nodes(), int)
    for element, nodes in pi.items():
        for node in nodes:
            partition_dict[node] = element

    g_rev = G.reverse() if G.is_directed() else G

    for V_i in pi.values():
        if len(V_i) > 1:
            # dict of partition_element -> number of connections
            rule = {}
            for i, vertex in enumerate(V_i):
                # construct rule
                if i == 0:
                    rule = getPartitionNeighbors(vertex, G, partition_dict, g_rev)
                # test other vertices against the rule
                else:
                    conns = getPartitionNeighbors(vertex, G, partition_dict, g_rev)
                    if conns != rule:
                        print(V_i)
                        print(
                            "last call{}getPartitionNeighbors({}, {}, part)".format(
                                "-" * 30, vertex, G
                            )
                        )
                        print("RULE: {}\nCONNS: {}".format(rule, conns))
                        return False
    return True


# specifically, we are getting the in-edge neighbors
def getPartitionNeighbors(vertex, G, partition_dict, g_rev):
    conns = {}
    for neighbor in g_rev.neighbors(vertex):
        part_el = partition_dict[neighbor]
        if part_el not in conns:
            conns.update({part_el: 0})
        conns.update({part_el: conns[part_el] + 1})
    return conns


# OLD CODE:


# duplicated here for usability (also found in lep_finder)
def printWithLabel(label, delim, item):
    print("{}\n{}\n{}\n".format(label, delim * len(label), item))


def PartitionAristotle(part_dict):
    """Aristotle was the great organizer, and fittingly, so is this function.
    It creates the permutation matrices necessary to group the partitions of the
    coarsest equitable partition in the adjacency matrix

    ARGUMENTS:
        part_dict (dict): contains lists of the vertices belonging to each
        partition element

    RETURNS:
        P (array): the right bound permutation matrix
        Pinv (array): the left bound permutation matrix
    """
    # get the size of the permutation matrices by
    # checking how many nodes are in the graph
    Psize = max([max(partElement) for partElement in part_dict.values()]) + 1

    P = np.zeros((Psize, Psize))  # create the zero matrix to fill
    I = np.eye(Psize, Psize)  # create identity as your pantry to pull from
    # to fill the columns of the zero matrix

    col = 0
    for part_element in part_dict.values():  # itertate through partition elements
        for vertex in part_element:  # iterate through vertices in partition elements
            P[:, col] = I[
                :, vertex
            ].copy()  # replace next column with appropriate part of I
            col += 1  # make sure you're getting the next column in the next iteration

    Pinv = np.linalg.inv(P)  # create P inverse
    return Pinv, P


REV_LABEL_ATTR = "relabel_reverse_mapping"


def reverseRelabel(G, ep_dict=None, lep_list=None):
    if hasattr(G, REV_LABEL_ATTR):
        unmapping = getattr(G, REV_LABEL_ATTR)
        nx.relabel_nodes(G, unmapping, copy=False)
        if ep_dict is not None:
            for i in ep_dict.keys():
                ep_dict[i] = {unmapping[node] for node in ep_dict[i]}
        if lep_list is not None:
            for i in range(len(lep_list)):
                lep_list[i] = {unmapping[node] for node in lep_list[i]}

    else:
        raise Exception("Graph must be relabeled before being reverseRelabeled")


def relabel(G):
    # unmapping = {index: label for index, label in enumerate(G.nodes())}
    # setattr(G, "relabel_reverse_mapping", unmapping)
    # if mapping is None:
    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    nx.relabel_nodes(G, mapping, copy=False)
