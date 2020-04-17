#pragma once
#include "deepgalois/gtypes.h"

void subgraph_sampler(Graph &g, Graph &sg);
galois::runtime::iterable<galois::NoDerefIterator<Graph::edge_iterator> > neighbor_sampler(Graph &g, GNode v);
Graph::edge_iterator sampled_edge_begin(Graph &g, GNode v) { return g.edge_begin(v); }
Graph::edge_iterator sampled_edge_end(Graph &g, GNode v) { return g.edge_end(v); }
