# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)
"""

import operator
from typing import Any, List, Set, Dict, Tuple

import networkx as nx

class LinkedListNode:
    """Base class for doubly-linked list nodes"""

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
    color_class : ColorClass object
        ColorClass object representing the node's current color class
    pseudo_class : ColorClass object
        ColorClass object representing the node's pseudo color class
    neighbors : list(int)
        list of integers corresponding to the node's neighbors, defined by in-edges
    """

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

        self.predecessors = predecessors # list of node indices of nodes with in-edges to self (in-edge neighbors)
        self.successors = successors # list of the indices of the neighboring nodes (out-edge neighbors)
        self.in_edge_count = 0
        self.out_edge_count = 0

    # magic methods
    def __hash__(self):
        if type(self.label) == int:
            return self.label
        return self.label.__hash__()

    def __str__(self):
        return str(self.label)

    
class LinkedList:
    """Base doubly-linked list class"""

    def __init__(self, data=None):
        """
        Initialize doubly-linked list.

        Parameters
        ----------
        data : list(Node)
            Creates a doubly-linked list from the elements of data. If data is
            not given, initializes an empty linked list.
        """

        self.head = None
        self.tail = None
        self.size = 0

        if data is not None:
            self.head = data[0]

            if len(data) == 1:
                self.tail = data[0]
            else:
                self.head.next = data[1]

                self.tail = data[-1]
                self.tail.prev = data[-2]

                self.size = len(data)

            node = self.head.next
            for i in range(1, self.size-1):
                data[i].prev = data[i-1]
                data[i].next = data[i+1]

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
        if self.head is None:
            raise ValueError("Cannot create list from empty LinkedList object.")

        temp = list()
        n = self.head

        while True:
            temp.append(n)

            if n.next is None:
                return temp
            n = n.next
    
    
class ColorClass(LinkedList):
    """
    Base class representing a color class.
    Attributes
    ----------

    Methods
    -------
    append(node)
        appends `node` to the ColorClass (linked list)
    remove(node)
        removes `node` from the ColorClass (linked list)


    computeStructureSet

    splitColor

    """

    def __init__(self, label=None):
        """
        TODO: add documentation here :)

        Parameters
        ----------
        label : int
            Integer value uniquely identifying this color class.
        """
        super().__init__()
        self.in_edge_neighbors = list()
        self.out_edge_neighbors = list()
        self.curr_color_in_edge_neighbors = 0
        self.curr_color_out_edge_neighbors = 0

        # color class label
        self.current_color = label

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
        data = list()
        if self.head is None:
            return data
        else:
            v = self.head

        while True:
            data.append(v.label)
            if v.next is None:
                return data

            v = v.next

    def _nodeObjects(self):
        """Returns a list of Node objects for nodes in this ColorClass"""
        data = list()
        if self.head is None:
            return data
        else:
            v = self.head

        while True:
            data.append(v)
            if v.next is None:
                return data

            v = v.next

    def computeColorStructure(self, C: List['ColorClass'], N: Dict[Any, Node]) -> None:
        """
        Computes the number of edges and color of neighbors of each node in this color class. These
        metrics are used in splitColor to determine which vertices should be separated into their
        own color class.

        Notes:
            We do not need to track which color classes contain in-edge or out-edge neighbors of 
            nodes in our color class to reset the value to zero; this is done already automatically
            in splitColor.
            
            This function counts the edges for each of the nodes and who they are connected to.

        Complexity
        ----------
        Time: Linear with number of edges in this color class (all edges on first pass)
        Space: Potentially up to number of edges, but never worse than all nodes in the graph

        """

        # reset neighbor lists
        self.out_edge_neighbors = list()
        self.in_edge_neighbors = list()
        # All the nodes in a color class are stored as a linked list
        # thus self.head returns the first node in the color class
        w = self.head
        while w is not None:
            # loop over each neighbor `v` of `w` (i.e., there exists an edge v <-- w)
            for v_ind in w.successors:
                v = N[v_ind]                                            # get node object
                if v.out_edge_count == 0:                               # checks if node v has already been seen
                    C[v.new_color].curr_color_out_edge_neighbors += 1   # if not, increments the out-edge neighbor count for its color
                    self.out_edge_neighbors.append(v)                   # and adds v to this color class's out-edge neighbors
                v.out_edge_count += 1                                   # increment count of out-edges from this color class to v
            
            # do the same for each in-edge neighbor `v` of `w` (i.e., there exists an edge v --> w)
            for v_ind in w.predecessors:
                v = N[v_ind]
                if v.in_edge_count == 0:
                    C[v.new_color].curr_color_in_edge_neighbors += 1
                    self.in_edge_neighbors.append(v)
                v.in_edge_count += 1

            w = w.next # move to next node in the color class

    def splitColor(self, C: List['ColorClass'], updated_nodes: Set[Node], n_colors: int, new_colors: Set[int]) -> int:
        """
        TODO: add documentation...

        C - color classes array
        L - set of nodes that got new pseudo colors
        n_colors - current number of colors
        new_colors - set of indices for new colors

        Complexity
        ----------
        Time: n log(n) where n is the number of neighboring nodes
        Space: linear with number of neighboring colorclasses, max neighboring nodes

        """
        
        # sort neighbors by number of edges from connecting them to this color class, ascending
        self.in_edge_neighbors.sort(key=operator.attrgetter('in_edge_count'))
        self.out_edge_neighbors.sort(key=operator.attrgetter('out_edge_count'))

        visited = set() # tracking which ColorClasses have been visited
        for v in self.in_edge_neighbors:
            # new_color has not been changed yet, so old_color is the same as new_color
            if v.old_color in visited:
                continue
            else:
                visited.add(v.old_color)
                b = v.old_color
                # set curr_conns to the smallest number of connections that a node in C[b] has with this color class
                if C[b].curr_color_in_edge_neighbors < C[b].size:
                    # if not all nodes in C[b] neighbor a node in this color class, then the min number of connections to this color class is zero
                    C[b].curr_conns = 0
                else:
                    # otherwise, v.in_edge_count is the minimum number of connections (since in_edge_neighbors was sorted by in_edge_count)
                    C[b].curr_conns = v.in_edge_count

                C[b].current_color = b # initializing current_color for use in next loop
                C[b].curr_color_in_edge_neighbors = 0 # resetting the number of in-edge neighbors for the next iteration

        for v in self.in_edge_neighbors: # iterate through all in-edge neighbor nodes of this color class
            b = v.old_color
            # curr_conns is the min number of connections in C[b] to the current color class. 
            #   nodes with more than this number of connections get moved into a different color class
            # Note on the logic here: this `if` is entered every time that a node from C[b] has more connections to this color class than did previous nodes. 
            #   Nodes in in_edge_neighbors are sorted by connections to this color class, so iterating over them yields in_edge_counts that
            #   are strictly increasing. When the node v has more connections to the current color class than did its predecessors from C[b],
            #   we change the curr_conns to match the in_edge_count, so this will not run again until we see another v in C[b] with a larger in_edge_count.
            if C[b].curr_conns != v.in_edge_count:
                C[b].curr_conns = v.in_edge_count   # update curr_conns with the new in_edge_count
                n_colors += 1                       # add new color
                C[b].current_color = n_colors       # update current_color to apply to subsequent nodes
                new_colors.add(n_colors)            # track new colors

            # As soon as we have gotten past all nodes v from C[b] with minimum in_edge_count, the current_color of C[b] will change (in the above if statement).
            #   All subsequent nodes from C[b] will go into this if statement and will recieve new_color values according to their in_edge_count (thus, all v in C[b]
            #   with equal in_edge_count will be given the same new_color value). The only nodes that will retain their original color value will be the nodes from 
            #   each C[b] with the same minimum in_edge_count
            if v.old_color != C[b].current_color:   # if current_color of C[b] changed
                updated_nodes.add(v)
                
                # NOTE: it may seem more intuitive to update the ColorClass sizes when node v is removed from one class
                #   added to the other (see recolor method); HOWEVER, we use the updated sizes before that point (e.g., 
                #   finding largest new colorclass before relabeling in recolor)
                C[v.new_color].size -= 1
                v.new_color = C[b].current_color
                C[v.new_color].size += 1
                if v.out_edge_count != 0:                               # if v also has out edges to the colorclass
                    C[v.old_color].curr_color_out_edge_neighbors -= 1   # decrement the out-edge neighbor count associated with the old colorclass
                    C[v.new_color].curr_color_out_edge_neighbors += 1   # and increment the out-edge neighbor count for the new colorclass

        # same logic as above, but for out-edge neighbors
        visited = set()
        for v in self.out_edge_neighbors:
            if v.new_color in visited:
                continue
            else:
                visited.add(v.new_color)
                b = v.new_color
                if C[b].curr_color_out_edge_neighbors < C[b].size:
                    C[b].curr_conns = 0
                else:
                    C[b].curr_conns = v.out_edge_count
                
                C[b].current_color = b
                C[b].curr_color_out_edge_neighbors = 0

        for v in self.out_edge_neighbors:
            b = v.new_color
            if C[b].curr_conns != v.out_edge_count:
                C[b].curr_conns = v.out_edge_count
                n_colors += 1
                C[b].current_color = n_colors
                new_colors.add(n_colors)

            if v.new_color != C[b].current_color:
                updated_nodes.add(v)
                
                C[v.new_color].size -= 1
                v.new_color = C[b].current_color
                C[v.new_color].size += 1

        return n_colors
    
    
    # magic methods
    def __hash__(self):
        return self.label

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
            
def initialize(G: nx.Graph | nx.DiGraph) -> Tuple[List[ColorClass], Dict[Any, Node]]:
    """
    Initializes the Node and ColorClass objects necessary for equitablePartition.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph

    Returns
    -------
    C : list(ColorClass)
        List containing ColorClass objects for finding the coarsest equitable
        partition of `G`.

        NOTE: Current implementation creates one ColorClass object per node in
        `G`; all but the first ColorClass objects are placeholders for *possible*
        future color classes. In future, the algorithm should be reworked to add
        ColorClass objects to the list as needed to reduce spatial complexity.

    N : list(Node)
        List of Node objects representing the nodes of `G`.

    Complexity
    ----------
    Time: Linear with number of nodes
    Space: Linear with number of nodes

    """
    num_nodes = G.number_of_nodes()
    
    # TODO: perhaps later optimize to not repeat calculations for undirected graphs

    # initialize Node list -- all start with ColorClass index of 0
    N = dict()
    for node in G.nodes():
        predecessors = list(G.predecessors(node) if nx.is_directed(G) else G.neighbors())
        # in DiGraphs, neighbors() is the same as successors()
        successors = list(G.neighbors(node))
        N[node] = Node(node, 0, predecessors, successors)
    
                
    # initialize ColorClass list
        # this creates n ColorClass objects for each of the n nodes. 
        # It's not the most efficient since the coarsest ep will generally not be trivial
    C = [ColorClass() for _ in range(num_nodes)]
    
    # add all nodes to their correspnding ColorClass
    for n in N.values():
        C[0].append(n)

    C[0].size = num_nodes # set ColorClass 0 size attribute
    return C, N


def initFromFile(file_path, num_nodes=None, delim=',', comments='#', directed=False, rev=False):
    """
    Initializes the Node and ColorClass objects necessary for equitablePartition.

    Parameters
    ----------
    file_path : the path to the file storing edge data of the graph to be analyzed
    num_nodes : the total number of nodes; only necessary if the file at file_path
        does not contain all nodes (i.e., if there are nodes with no edges between them)
    delim : the delimiter between source and destination nodes for each edge in the
        file at file_path; uses ',' by default
    comments : a character used to denote a comment, or line to ignore; uses '#' by default
    directed : a boolean indicating whether the graph is directed or not; uses False by default

    Returns
    -------
    C : list(ColorClass)
        List containing ColorClass objects for finding the coarsest equitable
        partition of `G`.

        NOTE: Current implementation creates one ColorClass object per node in
        `G`; all but the first ColorClass objects are placeholders for *possible*
        future color classes. In future, the algorithm should be reworked to add
        ColorClass objects to the list as needed to reduce spatial complexity.

    N : list(Node)
        List of Node objects representing the nodes of `G`.
    """

    # OUTDATED

    N = dict()
    
    with open(file_path, 'r') as f:
        for line in f:
            if line[0] != comments:
                line = line.strip().split(delim)
                if rev:
                    src = int(line[1])
                    dest = int(line[0])
                else:
                    src = int(line[0])
                    dest = int(line[1])
                if src not in N:
                    N[src] = Node(src, 0, [])
                N[src].neighbors.append(dest)
                if dest not in N:
                    N[dest] = Node(dest, 0, [])
                if not directed:
                    N[dest].neighbors.append(src)
    
    if num_nodes is not None:
        for node in range(num_nodes):
            if node not in N:
                N[node] = Node(node, 0, [])
    num_nodes = len(N)

    C = [ColorClass() for c in range(num_nodes)]

    # add all nodes to ColorClass 0
    for n in N.values():
        C[0].append(n)
    C[0].size = num_nodes

    return C, N


def recolor(C: List[ColorClass], updated_nodes: Set[Node]) -> None:
    """
    TODO: add documentation.

    Officially recolor nodes with the largest color classes keeping their old colors.

    Complexity
    ----------
    Time: Number of new colors * len(L) ???
    Space: Constant

    """

    for v in updated_nodes:
        C[v.old_color].remove(v)
        C[v.new_color].append(v)

    # make sure largest new color retains old color label (for a more efficient next iteration)
    for c in {v.old_color for v in updated_nodes}:
        d = max([(C[v.new_color].size, v.new_color) for v in updated_nodes if v.old_color == c])[1] # index of largest new colorclasses from same previous colorclass
        # if color d has more nodes than the original, switch their coloring
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    for v in updated_nodes:
        v.old_color = v.new_color


def equitablePartition(C: List[ColorClass], N: Dict[Any, Node], progress_bar: bool= True) -> Tuple[Dict[int, List[Any]], Dict[Any, Node]]:
    """
    Finds the coarsest equitable partition of a network
    
    Parameters
    ----------
    C : list(ColorClass)
        List containing ColorClass objects for finding the coarsest equitable
        partition of `G`
    
    N : list(Node)
        List of Node objects representing the nodes of `G`.
    
    Returns
    -------
    ep : dict(ColorClass.nodes())
         Dictionary of lists where each list represents nodes in the same partition 
    
    """

    progress = 0
    if progress_bar:
        print("Finding Coarsest EP...")
    
    new_colors = {0} # all nodes are in first color class to begin with
    n_colors = 0 # maximum color index; like a zero-indexed color count

    while len(new_colors) != 0:
        updated_nodes = set() # nodes with new colors
        temp_new_colors = set() # indices of new colors in C

        for c in new_colors:
            C[c].computeColorStructure(C, N)

            n_colors = C[c].splitColor(C, updated_nodes, n_colors, temp_new_colors)

            for v in C[c].in_edge_neighbors:
                v.in_edge_count = 0
            for v in C[c].out_edge_neighbors:
                v.out_edge_count = 0

        recolor(C, updated_nodes)
        new_colors = temp_new_colors

        progress += 1
        if progress_bar:
            updateProgress(progress)
        
    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C)) if C[color].size > 0}

    if progress_bar:
        updateProgress(progress, finished=True)

    return ep, N

def updateProgress(iterations, finished=False):
    print("\r{} iterations completed.".format(iterations), end='')
    if finished:
        print(" EP algorithm complete!")