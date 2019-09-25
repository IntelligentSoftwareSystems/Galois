/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#define USE_DAG
#define USE_DFS
#define ALGO_EDGE
//#define USE_EGONET
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

// This is a implementation of the WWW'18 paper:
// Danisch et al., Listing k-cliques in Sparse Real-World Graphs, WWW 2018
const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in a graph using DFS traversal (inputs do NOT need to be symmetrized)";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np, bool use_dag, unsigned c) : VertexMiner(g, size, np, use_dag, c) {}
	~AppMiner() {}
	// toExtend (only extend the last vertex in the embedding: fast)
	bool toExtend(unsigned n, const BaseEmbedding &emb, VertexId src, unsigned pos) {
		return pos == n-1;
	}
	// toAdd (only add vertex that is connected to all the vertices in the embedding)
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		#ifdef USE_DAG
		return is_all_connected_dag(dst, emb, n-1);
		#else
		VertexId src = emb.get_vertex(pos);
		return (src < dst) && is_all_connected(dst, emb, n-1);
		#endif
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "DfsMining/engine.h"
