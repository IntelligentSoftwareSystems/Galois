#!/usr/bin/env python

"""cholesky - Cholesky factorization as a graph algorithm."""

import networkx as nx           # Graph library
import matplotlib.pyplot as plt # For plotting
import numpy as np              # Does matrices and stuff
import scipy.io                 # For reading MatrixMarket files
from math import sqrt
import sys, time
import struct                   # For writing .gr files

def load_graph(filename):
    """Load a graph from named file, return the new (undirected) graph"""
    graphfh = open(filename, 'r')
    edges = []
    maxnode = 0
    for line in graphfh:
        line = line.strip()
        if not line:
            break
        row = [int(x) for x in line.split()]
        assert(len(row) == 2)
        maxnode = max(row+[maxnode])
        edges.append(row)
    graphfh.close()
    graph = nx.Graph()
    graph.add_nodes_from(range(maxnode+1))
    graph.add_edges_from(edges)
    return graph

def load_matrix(filename):
    """Load a (symmetric) matrix as an (undirected) graph.
    Supports whitespace separated table or MatrixMarket."""
    sparse = False
    try:
        data = np.loadtxt(filename, ndmin=2)
    except ValueError, err:
        # Maybe it's a MatrixMarket file
        if '%%MatrixMarket' in str(err):
            data = scipy.io.mmread(filename).tolil()
            sparse = True
            # Use numpy array because scipy matrices don't support indexing?
        else:
            raise
    print 'Loaded matrix.'
    if sparse:
        # This is much faster than doing it by hand
        return nx.from_scipy_sparse_matrix(data)
    else:
        return nx.from_numpy_matrix(data)

def write_total_ordering(graph, filename=None):
    """Convert a tree (directed graph) to a total ordering and write
    it to the named file (or stdout)."""
    my_str = ' '.join(str(i) for i in follow_tree(graph))
    if filename:
        outfh = open(filename, 'w')
        outfh.write(my_str)
        outfh.write("\n")
        outfh.close()
    else:
        print my_str

# Write a matrix to a file
def write_matrix(matrix, filename=None):
    """Write a matrix to named file (or stdout), formatted as whitespace
    separated values."""
    if filename:
        outfh = open(filename, 'w')
    else:
        outfh = sys.stdout
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if j != 0:
                outfh.write("\t")
            outfh.write(str(matrix[i,j]))
        outfh.write("\n")
    if filename:
        outfh.close()

def draw(graph, show=False, color='r', labels='weight', title=None):
    """Draw a graph on the screen using NetworkX + matplotlib"""
    layout = nx.shell_layout(graph)
    if labels:
        labeldict = dict((x[0:2], '%.2f' % x[2][labels])
                         for x in graph.edges_iter(data=True))
        nodelabels = dict((x, "\n\n%s\n%s" % (x, labeldict[(x, x)]))
                          for x in graph.nodes_iter())
        for key in labeldict.keys():
            if key[0] == key[1]:
                labeldict[key] = ''
        nx.draw(graph, pos=layout, node_color=color, labels=nodelabels)
        nx.draw_networkx_edge_labels(graph, pos=layout, edge_labels=labeldict)
    else:
        nx.draw(graph, pos=layout, node_color=color)
    if title:
        plt.title(title)
        print title
        nx.write_edgelist(graph, sys.stdout)
    if show:
        plt.show()

def ordering_sequential(graph):
    """Return the nodes of the graph in sequential order"""
    for i in range(graph.number_of_nodes()):
        yield i

def ordering_pointless(graph):
    """Return the nodes of the graph in order [1, 6, 4, 5, 0, 3, 7, 2]
    (extended to the number of nodes in the graph)"""
    order = [1, 6, 4, 5, 0, 3, 7, 2]
    count = graph.number_of_nodes()
    for i in range(0, count, len(order)):
        for j in order:
            if i+j < count:
                yield i+j

def ordering_least_degree(graph, tree=False):
    """Return nodes of the graph in order of least degree (not
    counting previously seen nodes). Also used for traversal of the
    elimination graph."""
    count = graph.number_of_nodes()
    done = set()
    for _ in range(count):
        # Find node of least degree (in-degree for trees)
        best = (-1, count+1)
        for node in graph.nodes_iter():
            if node in done:
                continue
            degree = 0
            if tree:
                neighbors = graph.predecessors_iter(node)
            else:
                neighbors = graph.neighbors_iter(node)
            # Only count unseen neighbors in the degree
            for j in neighbors:
                if j not in done:
                    degree += 1
                if degree >= best[1]:
                    # Never going to work
                    break
            else:
                assert(degree <= best[1])
                best = (node, degree)
            # If we've reached degree 0, quit now because we can't do better.
            if degree == 0:
                break
        assert(best[0] > -1)
        if tree:
            assert(best[1] == 1) # Self-edge only
        #else:
        #    print best[0], best[1]
        done.add(best[0])
        yield best[0]

def follow_tree(graph):
    """Generate an ordering from an elimination tree (directed graph)"""
    return ordering_least_degree(graph, tree=True)

def create_matrix(graph):
    """Convert a graph to a matrix"""
    count = graph.number_of_nodes()
    matrix = np.zeros((count, count)) # FIXME: Sparse
    for i in range(graph.number_of_nodes()):
        for j in range(count):
            if j in graph[i]:
                matrix[i][j] = graph[i][j]['weight']
    return matrix

def write_edgelist(graph, filename=None):
    """Write a graph to a file (or stdout) in edgelist format."""
    if filename:
        outfh = open(filename, 'w')
    else:
        outfh = sys.stdout
    for src, dest, data in graph.edges_iter(data=True):
        line = "%d %d" % (src, dest)
        if 'weight' in data:
            # http://stackoverflow.com/questions/3481289/
            line += " " + repr(data['weight'])
        outfh.write(line)
        outfh.write("\n")
    if filename:
        outfh.close()
        write_gr(graph, filename + ".gr")

def write_gr(graph, filename, fmt='d'):
    """Write a graph to a file in Galois .gr format."""
    # File format V1:
    # version (1) {uint64_t LE}
    # EdgeType size {uint64_t LE}
    # numNodes {uint64_t LE}
    # numEdges {uint64_t LE}
    # outindexs[numNodes] {uint64_t LE} (outindex[nodeid] is index of
    #     first edge for nodeid + 1 (end interator.  node 0 has an implicit
    #     start iterator of 0.
    # outedges[numEdges] {uint32_t LE}
    # potential padding (32bit max) to Re-Align to 64bits
    # EdgeType[numEdges] {EdgeType size}
    outfh = open(filename, 'wb')
    # Write header
    gr_version = 1
    edgetype_size = struct.calcsize(fmt)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    outfh.write(struct.pack("<4Q", gr_version, edgetype_size,
                            num_nodes, num_edges))
    # Assemble edge information
    outindexs = []
    outedges = []
    edgedata = []
    for node in graph.nodes_iter():
        for node_, dest, data in graph.edges_iter((node,), data=True):
            assert(node_ == node)
            outedges.append(dest)
            edgedata.append(float(data['weight']))
        outindexs.append(len(outedges))
    # Write outindexs, outedges. I'm using * notation because I have
    # a list of values I want to pack, but the function takes each
    # value as a separate argument (like printf). See section 4.7.4 of
    # http://docs.python.org/2.7/tutorial/controlflow.html
    outfh.write(struct.pack("<%dQ" % len(outindexs), *outindexs))
    outfh.write(struct.pack("<%dL" % len(outedges), *outedges))
    # Compute and write padding before edgedata
    pos = (4+len(outindexs))*2+len(outedges)
    print pos
    if pos % 2 != 0:
        outfh.write(struct.pack("4x"))
    # Write edge data
    outfh.write(struct.pack("<%d%s" % (len(edgedata), fmt), *edgedata))
    outfh.close()

def symbolic_factorization(graph, ordering, verbose=False):
    """Perform symbolic factorization on the graph, following given
    ordering."""
    done = set()                # Eliminated nodes
    tree = nx.DiGraph()         # Elimination tree
    for node in ordering(graph):
        done.add(node)          # Eliminate node
        neighbors = set(graph.neighbors(node))
        deps = neighbors & done # Previously eliminated neighbors (Dependencies)
        neighbors -= done       # Uneliminated Neighbors (Form a clique)
        if verbose and verbose > 1:
            print node, 'has uneliminated neighbors', neighbors

        # Make sure remaining neighbors form a clique:
        for src in neighbors:
            for dest in neighbors:
                if dest not in graph[src]:
                    graph.add_edge(src, dest, weight=0)

        # Determine dependencies
        tree.add_node(node)
        tree.add_edge(node, node, weight=graph[node][node]['weight'])
        for dep in deps:
            if node != dep:
                tree.add_edge(dep, node, weight=graph[dep][node]['weight'])
    return tree

def numeric_factorization(graph, verbose=False):
    """Perform numeric factorization on the directed graph"""
    done = set()
    for node in follow_tree(graph):
        try:
            factor = graph[node][node]['weight']
            assert(factor > 0)
            graph[node][node]['weight'] = factor = sqrt(factor)
            if verbose:
                print "STARTING %4d %10.5f" % (node, factor)
        except:
            print "ERROR %d %f" % (node, graph[node][node]['weight'])
            raise
        done.add(node)
        # Update self and outgoing edges
        for neighbor in graph.neighbors_iter(node):
            if neighbor not in done:
                graph[node][neighbor]['weight'] /= factor
                if verbose:
                    print "N-EDGE %4d %4d %10.5f" % \
                        (node, neighbor, graph[node][neighbor]['weight'])
        # Update edges between neighbors
        for src in graph.neighbors_iter(node):
            if src in done:
                continue
            doneself = False
            for dest in graph.neighbors_iter(node):
                if dest in graph[src] and dest not in done:
                    if src == dest:
                        doneself = True
                    graph[src][dest]['weight'] -= graph[node][src]['weight'] * \
                        graph[node][dest]['weight']
                    if verbose:
                        print "I-EDGE %4d %4d %10.5f" % \
                            (src, dest, graph[src][dest]['weight'])
            # Update self edge for this neighbor
            assert(doneself)

def do_cholesky(matrixfn, ordering=ordering_least_degree,
                display=False, verbose=False):
    """Perform a Cholesky factorization"""

    edgesfn = matrixfn + EDGES_SUFFIX
    filledfn = matrixfn + FILLED_SUFFIX
    depfn = matrixfn + DEP_SUFFIX
    choleskyfn = matrixfn + CHOLESKYEDGES_SUFFIX

    times = []
    # Read matrix from file
    graph = load_matrix(matrixfn)
    write_edgelist(nx.DiGraph(graph), filename=edgesfn)

    # Draw graph
    if display:
        plt.subplot(2, 2, 1)
        draw(graph, color=(1.0, 0.6, 0.6), title='Input Graph')

    # Build elimination tree
    print 'Symbolic factorization:'
    start = time.clock()
    tree = symbolic_factorization(graph, ordering, verbose=verbose)
    times.append(time.clock()-start)
    tree_edges = tree.number_of_edges()
    print 'Need to add', tree_edges-graph.number_of_edges(), 'edges.'
    print 'Total', tree_edges, 'edges.'

    # Draw filled and elimination graphs
    if display:
        plt.subplot(2, 2, 2)
        draw(graph, color=(0.8, 1.0, 0.8), title='After Elimination')
        plt.subplot(2, 2, 3)
        draw(tree, labels=False, color=(0.8, 0.8, 1.0),
             title='Elimination Graph')   
    # Write filled matrix, dependency graph
    if filledfn:
        write_edgelist(tree, filename=filledfn)
    if depfn:
        write_total_ordering(tree, filename=depfn)

    # Perform numeric factorization
    print 'Numeric factorization:'
    start = time.clock()
    numeric_factorization(tree, verbose=verbose)
    times.append(time.clock()-start)

    # Draw Cholesky factorization
    if display:
        plt.subplot(2, 2, 4)
        draw(tree, color=(1.0, 1.0, 0.8), title='Cholesky Factorization')
    # Write Cholesky matrix
    if choleskyfn:
        write_edgelist(tree, filename=choleskyfn)
    #if verbose:
    #print create_matrix(tree)

    # Show plots
    if display:
        plt.show()

    print "Took %s seconds" % times

ORDERINGS = {'sequential': ordering_sequential,
             'leastdegree': ordering_least_degree,
             'pointless': ordering_pointless}
EDGES_SUFFIX = '.edges'
FILLED_SUFFIX = '.filled'
DEP_SUFFIX = '.dep'
CHOLESKYEDGES_SUFFIX = '.choleskyedges'

def _main(argv):
    """Main function (entry point for command-line use)."""
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
Do Cholesky factorization.

Will create three output files:
MATRIX.filled        Filled graph as edge list
MATRIX.filled.gr     Filled graph as Galois .gr file with "double" edge data
MATRIX.dep           Dependency list (execution order for numeric part)
MATRIX.choleskyedges Edgelist for L, the Cholesky factor
''')
    parser.add_argument('matrix', metavar='MATRIX',
                        help='Filename of input matrix')
    parser.add_argument('--display', action='count',
                        help='Display the result on the screen')
    parser.add_argument('--verbose', action='count',
                        help='Display verbose output')
    parser.add_argument('--ordering', choices=ORDERINGS.keys(),
                        default='sequential',
                        help='Elimination order for symbolic factorization')
    args = parser.parse_args(argv)
    matrixfn = args.matrix

    do_cholesky(matrixfn, ordering=ORDERINGS[args.ordering],
                display=True if args.display else False,
                verbose=True if args.verbose else False)

if __name__ == '__main__':
    _main(sys.argv[1:])
