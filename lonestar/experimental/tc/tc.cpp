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

#include "galois/Galois.h"
#include "galois/Reduction.h"
#include "galois/Bag.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"
#include "galois/runtime/Profile.h"
#include <boost/iterator/transform_iterator.hpp>
#define CHUNK_SIZE 256

const char* name = "TC";
const char* desc = "Counts the triangles in a graph (only works for undirected neighbor-sorted graphs)";
const char* url  = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Mining/util.h"

void TcSolver(Graph& graph) {
	galois::GAccumulator<unsigned int> total_num;
	total_num.reset();
	galois::for_each(
		galois::iterate(graph.begin(), graph.end()),
		[&](const GNode& src, auto& ctx) {
			for (auto e1 : graph.edges(src)) {
				GNode dst = graph.getEdgeDst(e1);
				if (dst > src) break;
				for (auto e2 : graph.edges(dst)) {
					GNode dst_dst = graph.getEdgeDst(e2);
					if (dst_dst > dst) break;
					for (auto e3 : graph.edges(src)) {
						GNode dst2 = graph.getEdgeDst(e3);
						if (dst_dst == dst2) {
							total_num += 1;
							break;
						}
						if (dst2 > dst_dst) break;
					}
				}
			}
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), 
		galois::loopname("Couting")
	);
	galois::gPrint("\ttotal_num_triangles = ", total_num.reduce(), "\n\n");
}

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	read_graph(graph, filetype, filename);
	Tinitial.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n\n");

	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	TcSolver(graph);
	Tcomp.stop();
	return 0;
}
