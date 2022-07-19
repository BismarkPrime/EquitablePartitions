# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)
"""

from itertools import chain, combinations, groupby
import math
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

        self.f = color_class_ind        # color?
        self.temp_f = color_class_ind   # pseudo color?

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

        R

        Notes:
            We do not need to track which color classes have been hit as to reset
            the .hit value to zero; this is done already automatically in splitColor.
            
            This function counts the edges for each of the nodes and who they are connected to.

        Complexity
        ----------
        Time: Linear with number of edges in this color class (all edges on first pass)
        Space: Potentially up to number of edges, but never worse than all nodes in the graph

        """
        # reset structure_set
        self.structure_set = list()

        w = self.head
        while True:
            # loop over each neighboring node `v` of `w` (i.e., there exists an edge v <-- w)
            for v_ind in w.neighbors:
                v = N[v_ind] # get node object
                if v.structure_value == 0:       # checks if node v has been seen
                    C[v.temp_f].hit += 1         # records the total number of nodes the ColorClass sees                |   number of distinct vertices in temp_f color that neighbor a vertex in this color
                    self.structure_set.append(v) # records that the ColorClass sees node v                              |   the set of vertices adjacent to this color
                v.structure_value += 1           # increment structure value of v -- records that node v has been seen  |   the number of edges connecting v to this color

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

        Complexity
        ----------
        Time: n log(n) where n is the number of neighboring nodes
        Space: linear with number of neighboring colorclasses, max neighboring nodes

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
                # set current_p to the smallest number of connections that a node in C[b] has with this color class (in preparation for next loop, below)
                if C[b].hit < C[b].size: # if not all nodes in C[b] neighbor a node in this color class, then the min number of connections to this color class is zero
                    C[b].current_p = 0
                else:
                    C[b].current_p = v.structure_value # otherwise, v.structure_value is the minimum number of connections (since structure_set was sorted by structure_value)

                # current_color gets set to the temp_f value of the node in C[b] with the smallest number of connections to this color class
                C[b].current_color = b # current color is no longer none or previous value
                C[b].hit = 0

        for v in self.structure_set: # iterate through all vertices that neighbor nodes in this color class
            b = v.temp_f
            # current_p is the min number of connections in C[b] to the current color class. 
            #   nodes with more than this number of connections get moved into a different color class
            # this if is entered every time that a node from C[b] has more connections to this color class than did previous nodes. 
            #   Nodes in structure set are sorted by connections to this color class, so iterating over them yields structure_values that
            #   are strictly increasing. When the node v has more connections to the current color class than did its predecessors from C[b],
            #   we change the current_p to match the structure_value, so this will not run again until we see another v in C[b] with a larger structure_value.
            if C[b].current_p != v.structure_value: # if not all nodes in C[b] neighbor a node in this color class
                C[b].current_p = v.structure_value  # set current_p to the smallest number of connections that a node in C[b] has with this color class (gonna happen here or in previous loop)
                n_colors += 1                       # add new color
                C[b].current_color = n_colors
                new_colors.add(n_colors)            # track new colors

            # As soon as we have gotten past all nodes v from C[b] with minimum structure_value, the current_color of C[b] will change (in the above if statement).
            #   All subsequent nodes from C[b] will go into this if statement and will recieve new temp_f values according to their structure_value (thus, all v in C[b]
            #   with equal structure_value will be given the same temp_f value). The only nodes that will retain their original temp_f value will be the nodes from 
            #   each C[b] with the same minimum structure_value
            if v.temp_f != C[b].current_color:          # if color number of C[b] changed,
                                                        #   track which nodes were in C[b] and and move them to the new color class
                L.add(v)                                # add it to the set of nodes with new (pseudo?) colors
                # change temp_f (pseudo color) of v
                C[v.temp_f].size -= 1                   # decrement the size of the color class that v used to be in
                v.temp_f = C[b].current_color
                C[v.temp_f].size += 1                   # increment the size of the color class that v is in now

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

    Complexity
    ----------
    Time: Linear with number of nodes
    Space: Linear with number of nodes

    """
    num_nodes = G.number_of_nodes()
    
    # initialize Node list -- all start with ColorClass index of 0
    N = [Node(node, 0, list(G.neighbors(node))) for node in G.nodes()]
   
    # initialize ColorClass list
        # this creates n ColorClass objects for each of the n nodes. 
        # It's not the most efficient since the coarsest ep will generally not be trivial
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

    Complexity
    ----------
    Time: Number of new colors * len(L) ???
    Space: Constant

    """

    for v in L: # L is a list of vertices that got new pseudo colors
        # remove v from old color class
        C[v.f].remove(v)
        # append v to new color class
        C[v.temp_f].append(v)

    # make sure largest new color retains old color label
        # i think this is just for reducing the complexity (because computeStructureSet will iterate through new colors; more efficient to iterate over fewer vertices)
    for c in {v.f for v in L}:
        d = max([(C[v.temp_f].size, v.temp_f) for v in L if v.f == c])[1] # index of largest new colorclasses from same previous colorclass
        # if color d has more nodes than the original, switch their coloring
            # WRONG: equally (i think), if c != d (since d has max size, then if c != d then C[c].size < C[d].size)
            #   ...because if temp_f is different than f, then v.temp_f might not include c, so d not guaranteed to be max until compared with c
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    # set f = temp_f
    for v in L:
        v.f = v.temp_f

    return C


def equitablePartition(C, N, progress_bar = True):
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

    # import pdb; pdb.set_trace()

    progress = 0
    if progress_bar:
        print("Finding Coarsest EP...")

    new_colors = {0} # notice all nodes are in the same color class
    n_colors = 0
    iters = 0

    while True:
        iters += 1
        L = set() # nodes with new colors
        temp_new_colors = set() # indices of nodes with new colors

        iters_per_percent = len(new_colors) / 25

        for i, c in enumerate(new_colors):
            C, N = C[c].computeStructureSet(C, N) # has to do with counting the number of neighbors of each vertex and their respective colors

            args = (C, L, n_colors, temp_new_colors)
            args = C[c].splitColor(*args)
            C, L, n_colors, temp_new_colors = args

            for v in C[c].structure_set:
                v.structure_value = 0
            
            if progress_bar and iters_per_percent != 0 \
                    and i % math.ceil(iters_per_percent) == 0:
                updateLoadingBar(progress + i / iters_per_percent)
        
        progress += 25

        C = recolor(C, L)
        new_colors = temp_new_colors

        # break condition
        if new_colors == set():
            break

    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C)) if C[color].size > 0}

    progress = 100
    if progress_bar:
        updateLoadingBar(progress)
        print()

    return ep, N

def getIters(C, N, progress_bar = True):
    progress = 0
    if progress_bar:
        print("Finding Coarsest EP...")


    new_colors = {0} # notice all nodes are in the same color class
    n_colors = 0
    iters = 0

    while True:
        iters += 1
        L = set() # nodes with new colors
        temp_new_colors = set() # indices of nodes with new colors

        iters_per_percent = len(new_colors) / 25

        for i, c in enumerate(new_colors):
            C, N = C[c].computeStructureSet(C, N) # has to do with counting the number of neighbors of each vertex and their respective colors

            args = (C, L, n_colors, temp_new_colors)
            args = C[c].splitColor(*args)
            C, L, n_colors, temp_new_colors = args

            for v in C[c].structure_set:
                v.structure_value = 0
            
            if progress_bar and iters_per_percent != 0 \
                    and i % math.ceil(iters_per_percent) == 0:
                updateLoadingBar(progress + i / iters_per_percent)
        
        progress += 25

        C = recolor(C, args[1])
        new_colors = temp_new_colors

        # break condition
        if new_colors == set():
            break

    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C)) if C[color].size > 0}

    progress = 100
    updateLoadingBar(progress)

    return iters

def updateLoadingBar(percent):
    percent = min(100, int(percent))
    print("\r [{0}] {1}%".format('#' * percent + ' ' * (100 - percent), percent), end='')