import unittest
import random
import networkx as nx
import numpy as np
from multiprocessing import Pool as ThreadPool

from unittest import TestCase
from scipy import sparse
from typing import Callable

import tqdm

import sys
import os

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ep_finder
import lep_finder
from ep_utils import getEigenvaluesSparse, plotEquitablePartition
from utils import getSymmetricDifference, getSymmetricDifferenceMatching
from graphs import genBerthaSparse

def testSpecificCase():
    # seeds that failed (Erdos-Renyi Directed): 
    #   size 250; p=.004; seeds: 5117, 204, 6217
    #   size 80; p=.025; seeds: 6514, 5383
    #   size 40; p=0.0375; seeds: 74887
    G = nx.erdos_renyi_graph(40, .0375, directed=True, seed=74887)
    pi = ep_finder.getEquitablePartition(ep_finder.initFromNx(G))
    leps = lep_finder.getLocalEquitablePartitions(lep_finder.initFromNx(G), pi)
    #write the dense array to a file
    with open('mathematica_graph.txt', 'w') as f:
        for i in range(40):
            f.write('{')
            for j in range(40):
                f.write(str(nx.to_scipy_sparse_array(G)[i, j]) + ', ')
            f.write('},\n')
        f.write(str(nx.to_scipy_sparse_array(G).toarray()))
    noop = 1
    plotEquitablePartition(G, pi)
    sp_array = nx.to_scipy_sparse_array(G)
    lepard_eigenvalues = getEigenvaluesSparse(sp_array)
    np_eigenvalues = np.linalg.eigvals(sp_array.toarray())
    getSymmetricDifference(np_eigenvalues, lepard_eigenvalues)
    l1, l2 = getSymmetricDifferenceMatching(np_eigenvalues, lepard_eigenvalues)
    noop = 1

def getNonTrivialGraph(getGraph: Callable[[int], sparse.sparray]) -> sparse.sparray:
    def isNonTrivial(sp_array: sparse.sparray) -> bool:
        pi = ep_finder.getEquitablePartition(ep_finder.initFromSparse(sp_array))
        return len(pi) < sp_array.shape[0]
    
    while True:
        seed = random.randint(0, 10000)
        sp_array = getGraph(seed)
        if isNonTrivial(sp_array):
            return sp_array, seed

def assertSameEigenvalues(self: TestCase, sp_array: sparse.sparray, config: str):
    np_eigenvalues = np.linalg.eigvals(sp_array.toarray())
    lepard_eigenvalues = getEigenvaluesSparse(sp_array)

    unique_nx_vals, unique_lepard_vals = getSymmetricDifference(np_eigenvalues, lepard_eigenvalues)
    self.assertEqual(len(unique_nx_vals), 0, f"Failed for {config}")
    self.assertEqual(len(unique_lepard_vals), 0, f"Failed for {config}")

def assertSameEigenvaluesNonTest(sp_array: sparse.sparray, config: str):
    np_eigenvalues = np.linalg.eigvals(sp_array.toarray())
    lepard_eigenvalues = getEigenvaluesSparse(sp_array)

    unique_nx_vals, unique_lepard_vals = getSymmetricDifference(np_eigenvalues, lepard_eigenvalues)
    assert len(unique_nx_vals) == 0, f"Failed for {config}"
    assert len(unique_lepard_vals) == 0, f"Failed for {config}"

# Tests the eigenvalues returned by the LEParD algorithm for correctness
class TestEigvalsUndirected(TestCase):

    graph_sizes = [10, 100, 250]

    def testErdosRenyi(self):
        def getErdosRenyi(n: int, p: float, i: int) -> sparse.sparray:
            return nx.to_scipy_sparse_array(nx.erdos_renyi_graph(n, p, seed=i, directed=False))
        
        for node_degree in [1, 2, 4]:
            for num_nodes in self.graph_sizes:
                for _ in range(50):
                    sp_array, seed = getNonTrivialGraph(lambda i: getErdosRenyi(num_nodes, node_degree / num_nodes, i))
                    assertSameEigenvalues(self, sp_array, f"Erdos-Renyi({num_nodes}, {node_degree / num_nodes}, {seed})")

    def testRandomGeometric(self):
        def getRandomGeometric(n: int, r: float, i: int) -> sparse.sparray:
            return nx.to_scipy_sparse_array(nx.random_geometric_graph(n, r, seed=i))

        for radius in [0.01, 0.05, 0.1, 0.25]:
            for num_nodes in self.graph_sizes:
                for _ in range(50):
                    sp_array, seed = getNonTrivialGraph(lambda i: getRandomGeometric(num_nodes, radius, i))
                    assertSameEigenvalues(self, sp_array, f"RandGeo({num_nodes}, {radius}, {seed}")
    
    def testBertha(self):
        bertha_sizes = [10, 100, 250, 1000]
        for num_nodes in bertha_sizes:
            sp_array = genBerthaSparse(num_nodes)
            assertSameEigenvalues(self, sp_array, f"Bertha({num_nodes})")

class TestEigvalsDirected:

    graph_sizes = [10, 100, 250]
    test_sizes = [15, 20, 25, 30, 35, 40]

    def _testErdosRenyiDirected(self, node_degree: int, num_nodes: int):
        def getErdosRenyiDirected(n: int, p: float, i: int) -> sparse.sparray:
            return nx.to_scipy_sparse_array(nx.erdos_renyi_graph(n, p, seed=i, directed=True))
        
        iter = ''
        try:
            for seed in tqdm.tqdm(range(100000)):
                sp_array = getErdosRenyiDirected(num_nodes, node_degree / num_nodes, seed)
                iter = f"Erdos-Renyi[Dir]({num_nodes}, {node_degree / num_nodes}, {seed})"
                assertSameEigenvaluesNonTest(sp_array, iter)
        except Exception as e:
            print('=====================ERROR=====================')
            print(iter)
            print()
            print()

    def testErdosRenyiDirected(self):
        pool = ThreadPool(15)
        # get all combinations of node degree and test size
        combinations = [(node_degree, num_nodes) for node_degree in [1.5, 2.5, 3.5] for num_nodes in self.test_sizes]
        pool.starmap(self._testErdosRenyiDirected, combinations)
        pool.close()
        pool.join()

        # iter = ''
        # try:
            # for node_degree in [1, 2, 4]:
            #     for num_nodes in self.test_sizes:
            #         for seed in tqdm.tqdm(range(100000)):
            #             sp_array = getErdosRenyiDirected(num_nodes, node_degree / num_nodes, seed)
            #             # sp_array, seed = getNonTrivialGraph(lambda i : getErdosRenyiDirected(num_nodes, node_degree / num_nodes, i))
            #             iter = f"Erdos-Renyi[Dir]({num_nodes}, {node_degree / num_nodes}, {seed})"
            #             assertSameEigenvalues(self, sp_array, f"Erdos-Renyi[Dir]({num_nodes}, {node_degree / num_nodes}, {seed})")
        # except Exception as e:
        #     print(iter)


if __name__ == '__main__':
    testSpecificCase()
    # TestEigvalsDirected().testErdosRenyiDirected()