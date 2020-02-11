DESCRIPTION 
===========

This application does vertex classification in an undirected graph.
It uses graph neural network (GNN) to train the vertex features 
which are then used to classify vertices into different classes.

INPUT
===========

The input dataset contains three parts:
1. the input graph file: edgelist format of a |V| x |V| sparse matrix.
2. the vertex label file: |V| lines with each line a integer.
3. the input feature file: edgelist format of |V| x |D| sparse matrix.

Vertex ids are expected to be sequential integers between 0 and |V|-1.
|V| is the number of vertices. |D| is the dimension of input feature vectors.

BUILD
===========

1. Run cmake at BUILD directory `cd build; cmake -DUSE_EXP=1 ../`

2. Run `cd <BUILD>/lonestar/experimental/gnn; make -j`

RUN
===========

The following are a few example command lines.

$ ./gnn citeseer -t=56 -k=3
$ ./gnn cora -t=14 -k=3

PERFORMANCE
===========
- I
- I
- I

REFERENCES
===========
The GCN model:
Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)  
http://arxiv.org/abs/1609.02907 
https://github.com/tkipf/gcn

DGL:
Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs
https://arxiv.org/abs/1909.01315
https://github.com/dmlc/dgl

GraphSAGE: 
Inductive Representation Learning on Large Graphs
http://snap.stanford.edu/graphsage/

NeuGraph: Parallel Deep Neural Network Computation on Large Graphs
https://www.usenix.org/conference/atc19/presentation/ma

