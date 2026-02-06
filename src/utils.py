import cmath
import math
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from typing import List, Tuple, Sequence

# for debugging purposes
from matplotlib import pyplot as plt

def getSymmetricDifference(list1: Sequence[complex], list2: Sequence[complex], epsilon_start=1e-4, epsilon_max=1e-1) -> Tuple[List[complex], List[complex]]:
    '''
        Gets the symmetric difference of two lists. (Returns list1 - list2, list2 - list1)
        Assumes that two elements are equal if they are within epsilon of one another.
        Increases the value of epsilon by an order of magnitude, up to epsilon_max, until all elements of list2 can be removed from list1.
        If elements remain in list2 after reaching epsilon_max, throws an error.
    '''
    # NOTE: since maximum bipartite matching is generally solved as a max-flow problem, it
    #   may be comparable in speed to this naive method (n^2), and is more robust to edge cases.
    #   (Note, however, that the current implementation of max flow in NetworkX is quite slow.)
    #   Consider defaulting to the bipartite method in getSymmetricDifferenceMatching
    
    res1 = []
    res2 = []

    skip_indices1 = set()
    skip_indices2 = set()
    epsilon = epsilon_start

    while len(skip_indices2) < len(list2):
        for j, cnum2 in enumerate(list2):
            if j in skip_indices2:
                continue
            for i, cnum1 in enumerate(list1):
                if i in skip_indices1:
                    continue
                if cmath.isclose(cnum2, cnum1, abs_tol=epsilon):
                    skip_indices1.add(i)
                    skip_indices2.add(j)
                    break
        #TODO:nomerge should raise exception if trying to remove globals from total spectrum, but should
        # not raise exception if doing general copmparison (e.g., for testing). Consider splitting this function
        
        # if, with epsilon = epsilon_max, we still can't perform the operation, raise an error
        if epsilon >= epsilon_max and len(skip_indices2) < len(list2):
            unique_to_list1 = [cnum for i, cnum in enumerate(list1) if i not in skip_indices1]
            unique_to_list2 = [cnum for i, cnum in enumerate(list2) if i not in skip_indices2]
            print("Elements of list2 not present in list1:\n" +
                            f"{unique_to_list2}\n" +
                            "Elements of list1 not present in list2:\n" +
                            f"{unique_to_list1}" +
                            "Consider increasing epsilon_max.")
            
            # for debugging purposes
            # plot all points in list1 and list2
            plt.plot([cnum.real for cnum in list1], [cnum.imag for cnum in list1], 'ro')
            # don't fill in these points so we can see the overlap
            plt.plot([cnum.real for cnum in list2], [cnum.imag for cnum in list2], 'bo', fillstyle='none')
            plt.show()

            noop = 1 # breakpoint here
            # plot points in list1 that are not in list2 and vice versa
            plt.plot(*zip(*[(cnum.real, cnum.imag) for cnum in unique_to_list1]), 'ro')
            plt.plot(*zip(*[(cnum.real, cnum.imag) for cnum in unique_to_list2]), 'bo', fillstyle='none')
            plt.show()
            
            # raise Exception("set diff exception")
            break
            # raise Exception("Elements of list2 not present in list1:\n" +
            #                 f"{[cnum for i, cnum in enumerate(list2) if i not in skip_indices2]}\n" +
            #                 "Elements of list1 not present in list2:\n" +
            #                 f"{[cnum for i, cnum in enumerate(list1) if i not in skip_indices1]}" +
            #                 "Consider increasing epsilon_max.")
        # double epsilon until we reach epsilon_max
        epsilon = min(epsilon * 2, epsilon_max)
        

    for i, cnum in enumerate(list1):
        if i not in skip_indices1:
            res1.append(cnum)
    
    for j, cnum in enumerate(list2):
        if j not in skip_indices2:
            res2.append(cnum)

    return res1, res2

def _try_full_matching(
    list1: List[complex], list2: List[complex], threshold: float
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Try to find a min-cost bipartite matching where ALL elements of list2
    are matched to elements in list1 at the given threshold.

    Returns:
        success: True if all elements of list2 were matched
        matches: List of (idx1, idx2) pairs that were matched
    """
    n1, n2 = len(list1), len(list2)

    if n2 == 0:
        return True, []
    if n1 == 0:
        return False, []

    # Build cost matrix (n1 x n2)
    cost_matrix = np.full((n1, n2), math.inf)

    for i in range(n1):
        for j in range(n2):
            dist = abs(list1[i] - list2[j])
            if dist <= threshold:
                cost_matrix[i, j] = dist

    # Check if any column is all inf (element of list2 can't match anything)
    for j in range(n2):
        if np.all(np.isinf(cost_matrix[:, j])):
            return False, []

    # Solve assignment problem
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ValueError:
        return False, []

    # Extract matches - verify all have finite cost
    matches = []
    for i, j in zip(row_ind, col_ind):
        if math.isinf(cost_matrix[i, j]):
            # An element of list2 wasn't matched - failure
            return False, []
        matches.append((i, j))

    # Success: all n2 elements of list2 were matched
    return True, matches


def getSymmetricDifferenceMatching(
    list1: List[complex],
    list2: List[complex],
    min_threshold: float = 1e-8,
    max_threshold: float = 1e-2
) -> Tuple[List[complex], List[complex]]:
    """
    Gets the symmetric difference of two lists of complex numbers using min-cost
    bipartite matching (Hungarian algorithm) to find optimal pairings.

    Uses iterative threshold approach: starts at min_threshold and increases by
    10x each iteration until a complete matching is found or max_threshold is reached.
    Each iteration tries to match ALL elements fresh (not incrementally).

    Returns two lists: list1 - list2, and list2 - list1
    """
    if len(list1) == 0:
        return [], list(list2)
    if len(list2) == 0:
        return list(list1), []

    threshold = min_threshold
    matches = []

    # Try matching at increasing thresholds until success or max reached
    while threshold <= max_threshold:
        success, matches = _try_full_matching(list1, list2, threshold)
        if success:
            break
        threshold *= 10

    # Build sets of matched indices
    matched_idx1 = {i for i, j in matches}
    matched_idx2 = {j for i, j in matches}

    # Collect unmatched elements
    res1 = [list1[i] for i in range(len(list1)) if i not in matched_idx1]
    res2 = [list2[j] for j in range(len(list2)) if j not in matched_idx2]

    return res1, res2