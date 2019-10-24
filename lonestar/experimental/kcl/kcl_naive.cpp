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
#define USE_SIMPLE
#define USE_BASE_TYPES
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Kcl";
const char* desc = "Counts the K-Cliques in an undirected graph using BFS extension";
const char* url  = 0;

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g) : VertexMiner(g) {}
	~AppMiner() {}
	void init() {
		assert(k > 2);
		set_max_size(k);
		set_num_patterns(1);
	}
	// toExtend (extend every vertex in the embedding: slow)
	bool toExtend(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		return true;
	}
	bool toAdd(unsigned n, const BaseEmbedding &emb, VertexId dst, unsigned pos) {
		VertexId src = emb.get_vertex(pos);
		if (dst <= src) return false;
		if (is_vertexInduced_automorphism<BaseEmbedding>(n, emb, pos, src, dst)) return false;
		return is_all_connected_except(dst, pos, emb);
	}
	void print_output() {
		std::cout << "\n\ttotal_num_cliques = " << get_total_count() << "\n";
	}
};

#include "BfsMining/engine.h"

