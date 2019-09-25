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

#define USE_PID
#define USE_MAP
#define USE_SIMPLE
#define USE_CUSTOM
#define VERTEX_INDUCED
#define CHUNK_SIZE 256
#include "pangolin.h"

const char* name = "Motif Counting";
const char* desc = "Counts the vertex-induced motifs in a graph using BFS extension";
const char* url  = 0;
int num_patterns[3] = {2, 6, 21};

class AppMiner : public VertexMiner {
public:
	AppMiner(Graph *g, unsigned size, int np) : VertexMiner(g, size, np) {}
	~AppMiner() {}
	#ifdef USE_CUSTOM
	// customized pattern classification method
	unsigned getPattern(unsigned n, unsigned i, VertexId dst, const VertexEmbedding &emb, unsigned pos) { 
		if (n < 4) return find_motif_pattern_id(n, i, dst, emb, pos);
		return 0;
	}
	#endif
	void print_output() { printout_motifs(); }
};

#include "BfsMining/engine.h"

