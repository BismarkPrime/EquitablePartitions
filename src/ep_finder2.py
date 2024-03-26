# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)

To improve readability, we have changed some of the variable names used in Bastert's
paper. To translate between our naming and his, we provide the following table:

NAME IN PAPER       || NAME IN CODE
====================||====================
f (color)           || old_color
f bar (pseudo color)|| new_color
hit                 || curr_color_in_edge_neighbors (or curr_color_out_edge_neighbors)
p (structure_value) || in_edge_count (or out_edge_count)
current_p           || curr_conns
current_color       || split_color

TODO:
    Improve variable naming for code readability
    Parallelization
    Use some sort of bucket sort in splitcolor to achieve lower complexity (is paper wrong on this point?)
"""

from typing import Any, List, Set, Dict
import scipy.sparse as sp
import networkx as nx

import numpy as np

class LinkedListNode:
    """Base class for doubly-linked list nodes"""
    __slots__ = 'next', 'prev'

    def __init__(self):
        self.next = None
        self.prev = None


class Node(LinkedListNode):
    """
    Base class for network nodes. Inherits from LinkedListNode

    Attributes
    ----------
    label : int
        integer label of the vertex
    old_color : int
        integer representing the node's current color class
    new_color : ColorClass object
        integer representing the node's new color class
    predecessors : list(int)
        list of integers corresponding to the node's in-edge neighbors
    successors : list(int)
        list of integers corresponding to the node's out-edge neighbors
    """
    __slots__ = 'label', 'old_color', 'new_color', 'predecessors', 'successors', \
        'in_edge_count', 'out_edge_count', 'structure_value'

    def __init__(self, label: Any, color_class_ind: int, predecessors: List[int]=None, successors: List[int]=None):
        """
        Initialize the node with it's label and initial color.

        Parameters
        ----------
        label : int
            Integer label of the vertex
        color_class : ColorClass object
            ColorClass object representing the nodes current color class
        neighbors : list(int)
            List of integers corresponding to the node's neighbors, defined by out-edges
        structure_value : int
            Current structure value. Used in ColorClass().ComputeStructureSet().
            Needs to be set to zero at the begining of each call to ComputeStructureSet.
        """
        super().__init__()
        self.label = label

        self.old_color = color_class_ind
        self.new_color = color_class_ind

        if predecessors is None:
            predecessors = []
        if successors is None:
            successors = []
        #NOTE: this is temporarily set to an empty list to work with out-edges only (not transceiving)
        self.predecessors = [] # predecessors # list of node indices of nodes with in-edges to self (in-edge neighbors)
        self.successors = successors # list of the indices of the neighboring nodes (out-edge neighbors)
        self.in_edge_count = 0 # value used to count connections to a given color class
        self.out_edge_count = 0 # value used to count connections to a given color class

    # magic methods
    def __hash__(self):
        if type(self.label) == int:
            return self.label
        return self.label.__hash__()

    def __str__(self):
        return str(self.label)

    
class LinkedList:
    """Base doubly-linked list class"""

    def __init__(self, data: List[Node]=None):
        """
        Initialize doubly-linked list.

        Parameters
        ----------
        data    : a list of nodes from which to create the doubly-linked list. If 
                    data is not given, initializes an empty linked list.
        """

        if data is not None:
            self.head = data[0]
            self.tail = data[-1]
            
            if len(data) > 1:
                self.head.next = data[1]
                self.tail.prev = data[-2]

            self.size = len(data)

            for i in range(1, self.size - 1):
                data[i].prev = data[i - 1]
                data[i].next = data[i + 1]
        else:
            self.head = None
            self.tail = None
            self.size = 0


    def append(self, node):
        """Appends `node` to the list"""

        if self.head is None:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node # set the current tail's next to `node`
            node.prev = self.tail # set `node`'s previous to the current tail
            self.tail = node      # set current tail to node

    def remove(self, node):
        """
        Removes `node` from list. Assumes `node` is an element of the list,
        however, an error will *not* be raised if it is not.
        """

        if self.head is None:
            raise ValueError("List is empty.")

        if node is self.head and node is self.tail: # if `node` is the only member of the list
            node.next, node.prev = None, None
            self.head, self.tail = None, None

        elif node is self.head:
            # reset head and set prev to None
            self.head = node.next
            self.head.prev = None
            # remove node.next reference
            node.next = None
        
        elif node is self.tail:
            # reset tail and set next to None
            self.tail = node.prev
            self.tail.next = None
            # remove node.prev reference
            node.prev = None
        else:
            # link node.prev to node.next
            node.prev.next, node.next.prev = node.next, node.prev
            # drop next and prev references from node
            node.prev, node.next = None, None

    # magic methods
    def __list__(self):
        result = list()
        n = self.head

        while n is not None:
            result.append(n)
            n = n.next
        return result
    
    
class ColorClass(LinkedList):
    """
    Base class representing a color class.
    Attributes
    ----------


    """
    __slots__ = 'in_edge_neighbors', 'out_edge_neighbors', 'curr_color_in_edge_neighbors', \
        'curr_color_out_edge_neighbors', 'split_color', 'curr_conns'

    def __init__(self) -> None:
        """
        Initializes a ColorClass object.
        """
        super().__init__()
        self.in_edge_neighbors = list()
        self.out_edge_neighbors = list()
        self.curr_color_in_edge_neighbors = 0
        self.curr_color_out_edge_neighbors = 0

    def relabel(self, c):
        """Relabels the v.new_color/v.old_color values of every node v in the linked list"""
        v = self.head
        if v is None:
            raise ValueError("Color Class has no Nodes and therefore cannot be relabeled.")

        while True:
            v.new_color = c
            v.old_color = c
            if v.next is None:
                break
            v = v.next

    def nodes(self):
        """Returns a list of node labels for nodes in this ColorClass"""
        labels = list()

        node = self.head

        while node is not None:
            labels.append(node.label)
            node = node.next
        return labels

    def _nodeObjects(self):
        """Returns a list of Node objects for nodes in this ColorClass"""
        nodes = list()
        node = self.head

        while node is not None:
            nodes.append(node)
            node = node.next
        return nodes

    def computeColorStructure(self, C: List['ColorClass'], N: Dict[Any, Node]) -> None:
        """
        Computes the number of edges and color of neighbors of each node in this 
            color class. These metrics are used in splitColor to determine which 
            vertices should be separated into their own color class.

        Parameters
        ----------
        C   : the list of ColorClasses
        N   : the node dictionary

        Complexity
        ----------
        Time: Linear with number of edges in this color class
        Space: Linear with number of neighboring nodes to this class; 
            potentially up to number of edges in this color class, but never 
            worse than all nodes in the graph

        """

        # reset neighbor lists
        self.out_edge_neighbors = list()
        self.in_edge_neighbors = list()

        self.max_out_edge_count = 0
        self.max_in_edge_count = 0

        # All the nodes in a color class are stored as a linked list, so 
        #   self.head returns the first node in the color class
        w = self.head
        while w is not None:
            # loop over each neighbor v of w 
            #   (i.e., there exists an edge v <-- w)
            for v_ind in w.successors:
                v = N[v_ind]                                            # get node object
                if v.out_edge_count == 0:                               # check if node v has already been seen
                    C[v.new_color].curr_color_out_edge_neighbors += 1   # if not, increment out-edge neighbor count for its color
                    self.out_edge_neighbors.append(v)                   # and add v to this color class's out-edge neighbors
                v.out_edge_count += 1                                   # increment count of out-edges from this color class to v
                # track largest number of outgoing edges to a single node
                if v.out_edge_count > self.max_out_edge_count:
                    self.max_out_edge_count = v.out_edge_count
            
            # do the same for each in-edge neighbor v of w 
            #   (i.e., there exists an edge v --> w)
            for v_ind in w.predecessors:
                v = N[v_ind]
                if v.in_edge_count == 0:
                    C[v.new_color].curr_color_in_edge_neighbors += 1
                    self.in_edge_neighbors.append(v)
                v.in_edge_count += 1
                if v.in_edge_count > self.max_in_edge_count:
                    self.max_in_edge_count = v.in_edge_count

            w = w.next # move to next node in the color class

    def splitColor(self, C: List['ColorClass'], L: Set[Node]) -> None:
        """
        Uses metrics collected in computeColorStructure to determine which nodes 
            must be moved to a new color class; new color classes are assigend 
            for such nodes (i.e., the new_color attribute is set), but the 
            ColorClass list is not yet changed.

        Parameters
        ----------
        N   : the node dictionary
        C   : the list of ColorClasses
        L   : the set of nodes that will get new colors

        Complexity
        ----------
        Time: Log-linear (n log(n)) with the number of neighboring nodes
        Space: Linear with number of neighboring nodes

        """
        
        

        # sort neighbors by number of edges from connecting them to this color class, ascending
        # a bucket sort of these vertices will be bounded by the number of edges between a vertex
        #   and this color class, which will not be more than the size of the color class. To 
        #   improve performance on large color classes, we bound it by the actual max number of 
        #   edges.

        # NOTE: this may actually be somewhat slower in practice for many graphs, but is necessary 
        #   guarantee linear sorting complexity and m log(n) complexity overall. The alternative:
        # self.in_edge_neighbors.sort(key=operator.attrgetter('in_edge_count'))
        # self.out_edge_neighbors.sort(key=operator.attrgetter('out_edge_count'))
        bucketSort(self.in_edge_neighbors, 'in_edge_count', 1, self.max_in_edge_count)
        bucketSort(self.out_edge_neighbors, 'out_edge_count', 1, self.max_out_edge_count)

        visited = set() # tracking which ColorClasses have been visited
        for v in self.in_edge_neighbors:
            # new_color may have been changed in previous iterations, so we may 
            #   not use old_color here
            if v.new_color not in visited:
                visited.add(v.new_color)
                b = v.new_color
                # set curr_conns to the smallest number of connections that a 
                #   node in C[b] has with this color class
                if C[b].curr_color_in_edge_neighbors < C[b].size:
                    # if not all nodes in C[b] neighbor a node in this color class, 
                    #   then the min number of connections to this color class is zero
                    C[b].curr_conns = 0
                else:
                    # otherwise, v.in_edge_count is the minimum number of connections 
                    #   (since in_edge_neighbors was sorted by in_edge_count)
                    C[b].curr_conns = v.in_edge_count

                C[b].split_color = b # initializing split_color for use in next loop
                C[b].curr_color_in_edge_neighbors = 0 # resetting count for the next iteration

        for v in self.in_edge_neighbors:
            b = v.new_color
            # curr_conns is the min number of connections in C[b] to the current color class. Nodes 
            #   with more than this number of connections get moved into a different color class.
            # Note on the logic here: this `if` is entered every time that a node from C[b] has more 
            #   connections to this color class than did previous nodes. Nodes in in_edge_neighbors 
            #   are sorted by connections to this color class, so iterating over them yields 
            #   in_edge_counts that are strictly increasing. When the node v has more connections to 
            #   the current color class than did its predecessors from C[b], we change the 
            #   curr_conns to match the in_edge_count, so this will not run again until we see 
            #   another v in C[b] with a larger in_edge_count.
            if C[b].curr_conns != v.in_edge_count:
                C[b].curr_conns = v.in_edge_count   # update curr_conns with the new in_edge_count
                C.append(ColorClass())              # add new color
                C[b].split_color = len(C) - 1       # update split to apply to subsequent nodes

            # As soon as we have gotten past all nodes v from C[b] with minimum in_edge_count, the 
            #   split_color of C[b] will change (in the above if statement). All subsequent nodes 
            #   from C[b] will go into this if statement and will recieve new_color values according 
            #   to their in_edge_count (thus, all v in C[b] with equal in_edge_count will be given 
            #   the same new_color value). The only nodes that will retain their original color 
            #   value will be the nodes from each C[b] with the same minimum in_edge_count
            if v.new_color != C[b].split_color:   # if split_color of C[b] changed
                L.add(v)
                
                # NOTE: it may seem more intuitive to update the ColorClass sizes when nodes are 
                #   added or removed (in recolor); HOWEVER, we use the updated sizes before that 
                #   point (e.g., finding largest new colorclass before relabeling in recolor)
                C[v.new_color].size -= 1
                if v.out_edge_count != 0:                               # if v also has out edges to the colorclass
                    C[v.new_color].curr_color_out_edge_neighbors -= 1   # decrement the out-edge neighbor count associated with the old colorclass
                v.new_color = C[b].split_color
                C[v.new_color].size += 1
                if v.out_edge_count != 0:                               # if v also has out edges to the colorclass
                    C[v.new_color].curr_color_out_edge_neighbors += 1   # and increment the out-edge neighbor count for the new colorclass

        # same logic as above, but for out-edge neighbors
        # TODO: if possible, abstract out common logic to reduce code duplication with the above
        visited = set()
        for v in self.out_edge_neighbors:
            if v.new_color not in visited:
                visited.add(v.new_color)
                b = v.new_color
                if C[b].curr_color_out_edge_neighbors < C[b].size:
                    C[b].curr_conns = 0
                else:
                    C[b].curr_conns = v.out_edge_count
                
                C[b].split_color = b
                C[b].curr_color_out_edge_neighbors = 0

        for v in self.out_edge_neighbors:
            b = v.new_color
            if C[b].curr_conns != v.out_edge_count:
                C[b].curr_conns = v.out_edge_count
                C.append(ColorClass())
                C[b].split_color = len(C) - 1

            if v.new_color != C[b].split_color:
                L.add(v)
                
                C[v.new_color].size -= 1
                v.new_color = C[b].split_color
                C[v.new_color].size += 1

    # magic methods
    def __str__(self):
        v = self.head
        if v is None:
            return 'None'

        data = f'{v}'
        while True:
            if v.next is None:
                return data

            v = v.next
            data += f', {v}'

def bucketSort(objs: List[Any], attribute: str, attr_min_val: int, attr_max_val: int) -> None:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    objs        : the list to be sorted
    attribute   : the attribute with which to perform a bucket sort
    attr_min_val: the smallest possible value of `attribute`
    attr_max_val: the largest possible value of `attribute`

    Complexity
    ----------
    Time: Linear with length of objs and (attr_max_val - attr_min_val)
    Space: Linear with length of objs and (attr_max_val - attr_min_val)

    """
    buckets = [None for _ in range(attr_min_val, attr_max_val + 1)]
    for obj in objs:
        index = getattr(obj, attribute) - attr_min_val
        if buckets[index] is None:
            buckets[index] = set()
        buckets[index].add(obj)
    i = 0
    for bucket in buckets:
        if bucket is not None:
            for obj in bucket:
                objs[i] = obj
                i += 1
   
def initFromNx(G: nx.Graph | nx.DiGraph | sp.coo_matrix, sparse_alg=False) -> Dict[Any, Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    G   : the graph to be analyzed

    Returns
    -------
    N   : a list of Node objects representing the nodes of G

    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    # initialize Node list -- all start with ColorClass index of 0
    N = dict()
    if sparse_alg:
        #CGPT warning, if errors, check this.
        if G.format == 'csr':
            for node in range(G.shape[0]):
                predecessors = G.indices[G.indptr[node]:G.indptr[node + 1]].astype(str)
                successors = G.indices[G.indptr[node]:G.indptr[node + 1]].astype(str)
                N[str(node)] = Node(str(node), 0, predecessors, successors)
        elif G.format == 'coo':
            for node in [n for n in range(G.shape[0])]:
                predecessors = G.row[G.col == node].astype(str) # incoming edges
                successors = G.col[G.row == node].astype(str)   # outgoing edges
                N[str(node)] = Node(str(node),0,predecessors, successors)
    else:
        for node in G.nodes():
            predecessors = list(G.predecessors(node) if nx.is_directed(G) else [])
            # in DiGraphs, neighbors() is the same as successors()
            successors = list(G.neighbors(node))
            N[node] = Node(node, 0, predecessors, successors)

    return N

def initFromSparse(mat: sp.sparray) -> Dict[Any, Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    G   : the graph to be analyzed

    Returns
    -------
    N   : a list of Node objects representing the nodes of G

    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    # initialize Node list -- all start with ColorClass index of 0
    # TODO: this is very similar to the initialization in lep_finder.initFromSparse
    #   we should probably reuse shared logic instead of duplicating it here
    rows, cols = mat.nonzero()
    start = 0
    N = {i: Node(i, 0) for i in range(mat.shape[0])}
    while start < len(rows):
        curr_row = rows[start]
        end = start + 1
        while end < len(rows) and rows[end] == curr_row:
            end += 1
        N[curr_row].successors = cols[start:end]
        start = end
    
    matT = mat.transpose()
    rowsT, colsT = matT.nonzero()
    # if mat is symmetric, calculating predecessors is redundant
    # NOTE: since we already have rows and cols, this check for matrix symmetry 
    #   is empirically faster than the more intuitive (mat != mat.transpose()).nnz == 0
    if not all((np.array_equal(rows, rowsT), 
                np.array_equal(cols, colsT),
                np.array_equal(mat.data, matT.data))):
        # mat is not symmetric, populate predecessors
        start = 0
        while start < len(rowsT):
            curr_row = rowsT[start]
            end = start + 1
            while end < len(rowsT) and rowsT[end] == curr_row:
                end += 1
            N[curr_row].predecessors = set(colsT[start:end])
            start = end

    return N

def initFromFile(file_path: str, num_nodes: int=None, delim: str=',', 
                 comments: str='#', directed: bool=False, rev: bool=False) -> Dict[Any, Node]:
    """
    Initializes the Node list necessary for equitablePartition.

    Parameters
    ----------
    file_path   : the path to the file storing edge data of the graph to be 
                    analyzed
    num_nodes   : the total number of nodes; only necessary if file_path does 
                    not contain all nodes (i.e., if there are nodes with no 
                    edges between them); if num_nodes is provided, it is assumed 
                    that nodes are labeled with integers (zero-indexed)
    delim       : the delimiter between source and destination nodes for each 
                    edge in the file at file_path; uses ',' by default
    comments    : a character used to denote a comment, or line to ignore; uses 
                    '#' by default
    directed    : a boolean indicating whether the graph is directed or not; 
                    assumes undirected by default

    Returns
    -------
    N   : a list of Node objects representing the nodes of the graph described 
            in file_path
    
    Complexity
    ----------
    Time: Linear with number of nodes and with number of edges
    Space: Linear with number of nodes and with number of edges

    """

    N = dict()
    
    with open(file_path, 'r') as f:
        for line in f:
            if not comments.isspace():
                line = line.strip()
            if line[0] != comments:
                line = line.split(delim)
                src = int(line[0])
                dest = int(line[1])
                if rev:
                    src, dest = dest, src
                if src not in N:
                    N[src] = Node(src, 0, [], [])
                N[src].successors.append(dest)
                if dest not in N:
                    N[dest] = Node(dest, 0, [], [])
                # only populate predecessors list in the directed case
                if directed:
                    N[dest].predecessors.append(src)
                else:
                    N[dest].successors.append(src)
    
    if num_nodes is not None:
        for node in range(num_nodes):
            if node not in N:
                N[node] = Node(node, 0, [])

    return N


def recolor(C: List[ColorClass], L: Set[Node]) -> None:
    """
    Updates color classes to reflect the coloring stored in each node's 
        new_color attribute. When a color class splits, the largest derived 
        color class keeps the original color.
    
    Parameters
    ----------
    C   : the list of ColorClasses
    L   : the set of nodes that will get new colors

    Complexity
    ----------
    Time: Linear with len(L)
    Space: Linear with len(L)

    """

    for v in L:
        C[v.old_color].remove(v)
        C[v.new_color].append(v)

    # make sure largest new color retains old color label (for a more efficient next iteration)
    for c in {v.old_color for v in L}:
        # get index of largest new colorclass from same previous colorclass
        d = max({(C[c].size, c) for v in L if v.old_color == c})[1]
        # if color d has more nodes than the original, switch their coloring
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    for v in L:
        v.old_color = v.new_color


def equitablePartition(N: Dict[Any, Node], progress_bar: bool=False) -> Dict[int, List[Any]]:
    """
    Finds the coarsest equitable partition of a network.
    
    Parameters
    ----------
    N   : a list of Node objects representing the nodes of the graph to be 
            analyzed
    
    Returns
    -------
    ep  : a dictionary of (int, list) where each list represents nodes in the 
            same partition element

    Complexity
    ----------
    Time: 
    Space: 
    
    """

    # initialize ColorClass list
    # NOTE: it might be slightly faster just to initialize color class list to 
    #   have size len(N) (rather than adding to it until it whenever we add a 
    #   color class), but we may actually care about that wasted memory when 
    #   working with very large graphs
    C = [ColorClass()]
    
    # add all nodes to their correspnding ColorClass
    for n in N.values():
        C[0].append(n)

    C[0].size = len(N)

    progress = 0
    if progress_bar:
        print("Finding Coarsest EP...")
    
    prev_color_count = 0 # number of colors in the previous iteration

    # NOTE: the complexity of each iteration is proportional to 
    #   SUM from i=0 -> len(L) of degree(L[i]).
    #   Hence, if every node is recolored once, the complexity is bounded by the 
    #   total number of edges, m. Each node may be recolored at most log(n) 
    #   times, where n is the number of nodes. Hence, we have an overall 
    #   complexity of O(m log(n)).

    while len(C) > prev_color_count:
        L = set() # nodes with new colors

        color_count = len(C)

        # iterate over newly created colors from previous iteration
        for c in range(prev_color_count, len(C)):
            C[c].computeColorStructure(C, N)

            C[c].splitColor(C, L)

            for v in C[c].in_edge_neighbors:
                v.in_edge_count = 0
            for v in C[c].out_edge_neighbors:
                v.out_edge_count = 0

        recolor(C, L)

        prev_color_count = color_count

        progress += 1
        if progress_bar:
            updateProgress(progress)
        
    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C))}

    if progress_bar:
        updateProgress(progress, finished=True)

    return ep

def updateProgress(iterations, finished=False):
    print(f"\r{iterations} iterations completed.", end='')
    if finished:
        print(" EP algorithm complete!")