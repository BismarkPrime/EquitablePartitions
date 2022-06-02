# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)
"""

from itertools import chain, combinations, groupby
import operator

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

    def __init__(self, label, color_class_ind, neighbors=None):
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

        self.f = color_class_ind
        self.temp_f = color_class_ind

        self.neighbors = neighbors # this is a list of the indices of the neighboring nodes
        self.structure_value = 0

    # magic methods
    def __hash__(self):
        return self.label

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
        self.structure_set = list()
        self.hit = 0

        # color class label
        self.current_color = label

    def relabel(self, c):
        """Relabels the v.temp_f/v.f values of every node v in the linked list"""
        v = self.head
        if v is None:
            raise ValueError("Color Class has no Nodes and therefore cannot be relabeled.")

        while True:
            v.temp_f = c
            v.f = c
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

    def computeStructureSet(self, C, N):
        """
        Computes the (color) structure set L(c)

        Notes:
            We do not need to track which color classes have been hit as to reset
            the .hit value to zero; this is done already automatically in splitColor.
            
            This function counts the edges for each of the nodes and who they are connected to.
        """
        # reset structure_set
        self.structure_set = list()

        w = self.head
        while True:
            # loop over each neighboring node `v` of `w` (i.e., there exists an edge v <-- w)
            for v_ind in w.neighbors:
                v = N[v_ind] # get node object
                if v.structure_value == 0:       # checks if node v has been seen
                    C[v.temp_f].hit += 1         # records the total number of nodes the ColorClass sees
                    self.structure_set.append(v) # records that the ColorClass sees node v 
                v.structure_value += 1           # increment structure value of v -- records that node v has been seen

            w = w.next # move to next node in the color class
            # break condition
            if w is None:
                return C, N

    def splitColor(self, C, L, n_colors, new_colors):
        """
        TODO: add documentation...

        C - color classes array
        L - set of nodes that got new pseudo colors
        n_colors - current number of colors
        new_colors - set of indices for new colors
        """
        # sort structure set by structure values, ascending
        self.structure_set.sort(key=operator.attrgetter('structure_value')) # sort neighbors by structure_value

        visited = set() # tracking which ColorClasses have been visited
        for v in self.structure_set:
            if v.temp_f in visited:
                continue
            else:
                visited.add(v.temp_f)
                b = v.temp_f
                if C[b].hit < C[b].size:
                    C[b].current_p = 0
                else:
                    C[b].current_p = v.structure_value

                C[b].current_color = b # current color is no longer none or previous value
                C[b].hit = 0

        for v in self.structure_set:
            b = v.temp_f
            if C[b].current_p != v.structure_value:
                C[b].current_p = v.structure_value
                n_colors += 1
                C[b].current_color = n_colors
                new_colors.add(n_colors)    # track new colors

            if v.temp_f != C[b].current_color:
                L.add(v)
                # change temp_f (pseudo color) of v
                C[v.temp_f].size -= 1
                v.temp_f = C[b].current_color
                C[v.temp_f].size += 1

        return C, L, n_colors, new_colors
    
    
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
            
def initialize(G):
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
    """
    num_nodes = G.number_of_nodes()
    
    # initialize Node list -- all start with ColorClass index of 0
    N = [Node(node, 0, list(G.neighbors(node))) for node in G.nodes()]
   
    # initialize ColorClass list
        # this creates n ColorClass objects for each of the n nodes. 
        # Its not the most efficient since the coarsest ep will not be trivial
    C = [ColorClass() for c in range(num_nodes)]
    
    # add all nodes to ColorClass 0
    for n in N:
        C[0].append(n)

    C[0].size = num_nodes # set ColorClass 0 size attribute

    return C, N


def recolor(C, L):
    """
    TODO: add documentation.

    Officially recolor nodes with the largest color classes keeping their old colors.
    """

    for v in L: # L is a list of vertices that got new pseudo colors
        # remove v from old color class
        C[v.f].remove(v)
        # append v to new color class
        C[v.temp_f].append(v)

    # make sure largest new color retains old color label
    for c in {v.f for v in L}:
        d = max([(C[v.temp_f].size, v.temp_f) for v in L if v.f == c])[1] # max index of new colorclasses from same previous colorclass
        # if color d has more nodes than the original, switch their coloring
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    # set f = temp_f
    for v in L:
        v.f = v.temp_f

    return C


def equitablePartition(C, N):
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

    new_colors = {0} # notice all nodes are in the same color class
    n_colors = 0

    while True:
        L = set()
        temp_new_colors = set()

        for c in new_colors:
            C, N = C[c].computeStructureSet(C, N)

            args = (C, L, n_colors, temp_new_colors)
            args = C[c].splitColor(*args)
            C, L, n_colors, temp_new_colors = args

            for v in C[c].structure_set:
                v.structure_value = 0

        C = recolor(C, args[1])
        new_colors = temp_new_colors

        # break condition
        if new_colors == set():
            break

    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C)) if C[color].size > 0}
    
    return ep, N

