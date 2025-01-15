import unittest
import random
import networkx as nx
import numpy as np

from unittest import TestCase
from scipy import sparse
from typing import Callable

import ep_finder
from ep_utils import getEigenvaluesSparse
from utils import getSymmetricDifference
from graphs import genBerthaSparse

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

class TestEigvalsDirected(TestCase):

    graph_sizes = [10, 100, 250]

    def testErdosRenyiDirected(self):
        def getErdosRenyiDirected(n: int, p: float, i: int) -> sparse.sparray:
            return nx.to_scipy_sparse_array(nx.erdos_renyi_graph(n, p, seed=i, directed=True))
        for node_degree in [1, 2, 4]:
            for num_nodes in self.graph_sizes:
                for _ in range(50):
                    sp_array, seed = getNonTrivialGraph(lambda i : getErdosRenyiDirected(num_nodes, node_degree / num_nodes, i))
                    assertSameEigenvalues(self, sp_array, f"Erdos-Renyi[Dir]({num_nodes}, {node_degree / num_nodes}, {seed})")