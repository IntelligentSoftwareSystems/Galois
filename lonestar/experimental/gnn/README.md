DESCRIPTION 
===========

This application does vertex classification in an undirected graph 

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

PERFORMANCE
===========
- I
- I
- I
