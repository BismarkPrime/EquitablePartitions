# ep_finder.py
"""
Provides the data structures and algorithm necessary to calculate the
coarsest equitable partition of a graph consisting of n vertices and m edges
in O(n log(m)) time.

Implementation based on the 1999 paper "Computing equitable partitions of graphs",
by Bastert (http://match.pmf.kg.ac.rs/electronic_versions/Match40/match40_265-272.pdf)
"""

import operator

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
    color_class : ColorClass object
        ColorClass object representing the node's current color class
    pseudo_class : ColorClass object
        ColorClass object representing the node's pseudo color class
    neighbors : list(int)
        list of integers corresponding to the node's neighbors, defined by in-edges
    """
    slots = 'label', 'f', 'temp_f', 'neighbors', 'structure_value'

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
        if type(self.label) == int:
            return self.label
        return self.label.__hash__()

    def __str__(self):
        return str(self.label)

    
class LinkedList:
    """Base doubly-linked list class"""
    __slots__ = 'head', 'tail', 'size'

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
    __slots__ = 'structure_set', 'hit', 'current_color', 'current_p'

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

    #@profile
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
        # JOSEPH NOTES:
            # .hit is how many nodes see the current color class. period
            # structure_value - how many connections the current node has 
            # to the current color class.
            # structure_set - set of nodes outside our partition element that connect
            # to our color class.
        # reset structure_set
        self.structure_set = list()
        # All the nodes in a color class are stored as a linked list
        # thus self.head returns the first node in the color class
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

    #@profile
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
        #JOSEPH NOTES:
        #   current_p is the minimum number of connections to the color class that we're considering
        #   this could be 0 or not...
        
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
                C[b].hit = 0 # resetting the hit value for the next iteration

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
                # current_color better named (split_off_off, this is what they will be assigned when split)
                # C.append(ColorClass(n_colors))
                C[b].current_color = n_colors
                new_colors.add(n_colors)            # track new colors
                # if len(C) <= b:
                #     print("len(C) = {}, b = {}!!!!!!!!!!!!!!!".format(len(C), b))

            # As soon as we have gotten past all nodes v from C[b] with minimum structure_value, the current_color of C[b] will change (in the above if statement).
            #   All subsequent nodes from C[b] will go into this if statement and will recieve new temp_f values according to their structure_value (thus, all v in C[b]
            #   with equal structure_value will be given the same temp_f value). The only nodes that will retain their original temp_f value will be the nodes from 
            #   each C[b] with the same minimum structure_value
            if v.temp_f != C[b].current_color:          # if color number of C[b] changed,
                # can call L nodes_to_be_updated        #   track which nodes were in C[b] and and move them to the new color class
                try:
                    L.add(v)                                # add it to the set of nodes with new (pseudo?) colors
                except:
                    print("v = {} with type {}".format(v, type(v)))
                    L.add(v)
                # change temp_f (pseudo color) of v
                # TODO: perhaps better to update the sizes when node v is removed from one class and appended to the other? (see recolor)
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
            
#@profile
def initialize(G, init_C=None):
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
    
    # initialize Node list -- all start with ColorClass index of 0, unless C was passed in
    if init_C is None:
        N = {node: Node(node, 0, list(G.neighbors(node))) for node in G.nodes()}
    else:
        # if init_C was passed in, initialize each node with the color class it had in init_C
        N = dict()
        for color in range(len(init_C)):
            curr_node = init_C[color].head
            while curr_node is not None:
                N[curr_node.label] = Node(curr_node.label, color, list(G.neighbors(curr_node.label)))
                curr_node = curr_node.next
                
    # initialize ColorClass list
        # this creates n ColorClass objects for each of the n nodes. 
        # It's not the most efficient since the coarsest ep will generally not be trivial
    C = [ColorClass() for _ in range(num_nodes)]
    
    # add all nodes to their correspnding ColorClass (usually 0, unless C was passed in)
    for n in N.values():
        C[n.f].append(n)

    if init_C is None:
        C[0].size = num_nodes # set ColorClass 0 size attribute
    else:
        # set size attribute for each non-empty color class in C
        # note: this assumes that init_C has at least as many elements as the number of nodes in the
        #    graph, since the size of C is len(G)
        for color in range(len(C)):
            if C[color].head is not None:
                C[color].size = init_C[color].size

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

    C[0].size = num_nodes # set ColorClass 0 size attribute

    return C, N


#@profile
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
    for c in {v.f for v in L}: # this for loop is so we don't repeat
        d = max([(C[v.temp_f].size, v.temp_f) for v in L if v.f == c])[1] # index of largest new colorclasses from same previous colorclass
        # if color d has more nodes than the original, switch their coloring
            # WRONG: equally (i think), if c != d (since d has max size, then if c != d then C[c].size < C[d].size)
            #   ...because if temp_f is different than f, then v.temp_f might not include c, so d not guaranteed to be max until compared with c
            #       In other words, because L is the set of new colors, and d = v.temp_f for some v \in L, d is one of the new colors, and c is the old color.
            #           Thus, d != c and they must be compared
        if C[c].size < C[d].size:
            C[c].relabel(d)
            C[d].relabel(c)
            C[c], C[d] = C[d], C[c]

    # set f = temp_f, this is a reset for the next iteration
    for v in L:
        v.f = v.temp_f

    return C

#@profile
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


    progress = 0
    if progress_bar:
        print("Finding Coarsest EP...")
    
    # Note: unless C was custom initialized, the first color class will be the only one with nodes
    new_colors = {color for color in range(len(C)) if C[color].size > 0} # generally equivalent to new_colors = {0}
    # all colors treated as new colors in first iteration, ergo new_colors contains all colors
    # n_colors is the maximum color index
    n_colors = len(new_colors) - 1 # generally equivalent to n_colors = 1
    iters = 0

    while True:
        iters += 1
        L = set() # nodes with new colors (possible new name: nodes_updated)
        temp_new_colors = set() # indices of nodes with new colors

        for c in new_colors:
            # assignment likely unnecessary here as well
            C, N = C[c].computeStructureSet(C, N) # has to do with counting the number of neighbors of each vertex and their respective colors

            args = (C, L, n_colors, temp_new_colors)
            # n_color is (perhaps) the only argument that needs reassignment
            args = C[c].splitColor(*args)
            C, L, n_colors, temp_new_colors = args

            for v in C[c].structure_set:
                v.structure_value = 0

        # I don't think reassignment is necessary
        # C = recolor(C, L)
        recolor(C, L)
        new_colors = temp_new_colors

        progress += 1
        if progress_bar:
            updateProgress(progress)

        # break condition
        if new_colors == set():
            break
        

    # put equitable partition into dictionary form {color: nodes}
    ep = {color: C[color].nodes() for color in range(len(C)) if C[color].size > 0}

    if progress_bar:
        updateProgress(progress, finished=True)

    return ep, N

def updateProgress(iterations, finished=False):
    print("\r{} iterations completed.".format(iterations), end='')
    if finished:
        print(" EP algorithm complete!")
    # percent = min(100, int(percent))
    # print("\r [{0}] {1}%".format('#' * percent + ' ' * (100 - percent), percent), end='')