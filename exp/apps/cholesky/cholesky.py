#!/usr/bin/env python

# Cholesky factorization as a graph algorithm.

import networkx as nx           # Graph library
import matplotlib.pyplot as plt # For plotting
import numpy as np              # Does matrices and stuff
import scipy.io as sio          # Supports writing Matrix Market files 
import math, sys

# Load a graph from a file, return the new graph
def load_graph(fn):
    f = open(fn, 'r')
    edges = []
    maxnode = 0
    for line in f:
        line = line.strip()
        if not line:
            break
        row = [int(x) for x in line.split()]
        assert(len(row) == 2)
        maxnode = max(row+[maxnode])
        edges.append(row)
    f.close()
    G = nx.Graph()
    G.add_nodes_from(range(maxnode+1))
    G.add_edges_from(edges)
    return G

# Load a (symmetric) matrix as a graph
def load_matrix(fn):
    data = np.loadtxt(fn)
    print 'Loaded matrix:'
    print data
    assert(len(data) > 0)
    assert(len(data) == len(data[0]))
    nodes = range(len(data))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in nodes:     # Row
        for j in nodes: # Column
            if data[i][j] == 0:
                pass
            elif i > j:
                assert(j in G[i])
            else:
                G.add_edge(i, j, weight=data[i][j])
    return G

# Convert a tree to a total ordering
def write_total_ordering(T, f=None):
    my_str = ' '.join(str(i) for i in follow_tree(T))
    if f:
        fh = open(f, 'w')
        fh.write(my_str)
        fh.write("\n")
        fh.close()
    else:
        print my_str

# Write a matrix to a file
def write_matrix(A, fn=None):
    if fn:
        fh = open(fn, 'w')
    else:
        fh = sys.stdout
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            if j != 0:
                fh.write("\t")
            fh.write(str(A[i][j]))
        fh.write("\n")
    if fn:
        fh.close()

# Draw a graph on the screen
def draw(G, show=False, color='r', labels='weight', title=None):
    layout = nx.shell_layout(G)
    if labels:
        labeldict = dict((x[0:2], '%.2f' % x[2][labels]) for x in G.edges(data=True))
        nodelabels = dict((x, "\n\n%s\n%s"%(x,labeldict[(x,x)])) for x in G.nodes())
        for key in labeldict.keys():
            if key[0] == key[1]:
                labeldict[key] = ''
        nx.draw(G, pos=layout, node_color=color, labels=nodelabels)
        nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labeldict)
    else:
        nx.draw(G, pos=layout, node_color=color)
    if title:
        plt.title(title)
        print title
        nx.write_edgelist(G, sys.stdout)
    if show:
        plt.show()

# Return nodes of the graph in sequential order
def ordering_sequential(G):
    for i in range(G.number_of_nodes()):
        yield i

# Return nodes of the graph in order [1, 6, 4, 5, 0, 3, 7, 2]
def ordering_given(G):
    order = [1, 6, 4, 5, 0, 3, 7, 2]
    assert(G.number_of_nodes() == len(order))
    for i in order:
        yield i

# Return nodes of the graph in order of least degree (not counting previously
# seen nodes). Also used for traversal of the elimination tree.
def ordering_least_degree(G, tree=False):
    n = G.number_of_nodes()
    done = set()
    for i in range(n):
        # Find node of least degree (in-degree for trees)
        best = (-1, n+1)
        for node in G.nodes_iter():
            if node in done:
                continue
            degree = 0
            if tree:
                neighbors = G.predecessors_iter(node)
            else:
                neighbors = G.neighbors_iter(node)
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
            assert(best[1] == 0)
        done.add(best[0])
        yield best[0]

# Generate an ordering from an elimination tree
def follow_tree(T):
    return ordering_least_degree(T, tree=True)

# Convert a graph to a matrix
def matrix(G):
    L = np.zeros((G.number_of_nodes(),G.number_of_nodes())) # FIXME: Sparse
    for i in range(G.number_of_nodes()):
        for j in range(i+1):
            if j in G[i]:
                L[i][j] = G[i][j]['weight']
    return L

# Convert a graph to an edgelist
def write_edgelist(G, f=None):
    if f:
        fh = open(f, 'w')
    else:
        fh = sys.stdout
    for i in G.edges_iter(data=True):
        line = "%d %d" % (i[0], i[1])
        if 'weight' in i[2]:
            line += " %f" % i[2]['weight']
        fh.write(line)
        fh.write("\n")
    if f:
        fh.close()

# Perform a Cholesky factorization
def main(matrixfn, filledfn, depfn, choleskyfn,
         ordering=ordering_least_degree):
    # Read matrix from file
    G = load_matrix(matrixfn)
    
    # Draw graph
    plt.subplot(2,2,1)
    draw(G, color=(1,.6,.6), title='Input Graph')

    # Build elimination tree
    done = set()                # Eliminated nodes
    T = nx.DiGraph()            # Elimination tree
    added_edges = 0
    total_edges = 0
    print 'Symbolic factorization:'
    for node in ordering(G):
        done.add(node)          # Eliminate node
        neighbors = set(G.neighbors(node))
        deps = neighbors & done # Previously eliminated neighbors (Dependencies)
        neighbors -= done       # Uneliminated Neighbors (Form a clique)
        print node, 'has uneliminated neighbors', neighbors

        # Make sure remaining neighbors form a clique:
        for node1 in neighbors:
            for node2 in neighbors:
                if node2 not in G[node1]:
                    G.add_edge(node1, node2, weight=0)
                    added_edges += 1
                total_edges += 1

        # Determine dependencies
        T.add_node(node)
        for dep in deps:
            if node != dep:
                # FIXME: Transitive reduction
                T.add_edge(dep, node)

    print 'Need to add', added_edges, 'edges.'
    print 'Total', total_edges, 'edges.'

    # Draw filled graph
    plt.subplot(2,2,2)
    draw(G, color=(.8,1,.8), title='After Elimination')
    
    # Write filled matrix, dependency graph
    if filledfn:
        write_edgelist(G, f=filledfn)
    if depfn:
        write_total_ordering(T, f=depfn)

    # Draw elimination graph
    plt.subplot(2,2,3)
    draw(T, labels=False, color=(.8,.8,1),
         title='Elimination Tree (TODO: transitivity)')

    # Perform numeric factorization
    done = set()
    print 'Numeric factorization:'
    for node in follow_tree(T):
        factor = math.sqrt(G[node][node]['weight'])
        # Update self and outgoing edges
        for neighbor in G.neighbors_iter(node):
            if neighbor not in done:
                G[node][neighbor]['weight'] /= factor
                print node, neighbor, G[node][neighbor]['weight']
        done.add(node)
        # Update edges between neighbors
        for n1 in G.neighbors_iter(node):
            if n1 in done:
                continue
            for n2 in G.neighbors_iter(node):
                if n1 < n2 and n2 not in done:
                    G[n1][n2]['weight'] -= G[node][n1]['weight']*G[node][n2]['weight']
                    print n1, n2, G[n1][n2]['weight']
            # Update self edge for this neighbor
            G[n1][n1]['weight'] -= G[node][n1]['weight']*G[node][n1]['weight']
            print n1, n1, G[n1][n1]['weight']

    # Draw Cholesky factorization
    plt.subplot(2,2,4)
    draw(G, color=(1,1,.8), title='Cholesky Factorization')

    # Write Cholesky matrix
    print matrix(G)
    if choleskyfn:
        write_edgelist(G, f=choleskyfn)    

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 5:
        print "Usage: %s <matrix_in> [<fillededges_out> <depgraph_out> <choleskyedges_out>]" % \
            (sys.argv[0],)
        print "Example: %s matrix1.txt edgelist.txt deplist.txt" % \
            (sys.argv[0],)
        print "If unspecified, output to <inputfile>.filled, <inputfile>.dep, and"
        print "<inputfile>.choleskyedges"
        print
        print "filled is the filled graph (after symbolic factorization)"
        print "dep is the dependency graph (for running numeric factorization)"
        print "choleskyedges is the graph of the the answer (Cholesky factor L)"
        sys.exit(1)
    matrixfn = sys.argv[1]
    try:
        filledfn = sys.argv[2]
        depfn = sys.argv[3]
        choleskyfn = sys.argv[4]
    except:
        filledfn = matrixfn + ".filled"
        depfn = matrixfn + ".dep"
        choleskyfn = matrixfn + ".choleskyedges"
    main(matrixfn, filledfn, depfn, choleskyfn,
         ordering=ordering_sequential)#least_degree)
    # Show plots
    plt.show()

