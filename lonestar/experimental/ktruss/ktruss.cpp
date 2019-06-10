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
#include "galois/graphs/Graph.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

#define ENABLE_LABEL

const char* name = "Maximal k-trusses";
const char* desc = "Computes the maximal k-trusses for a given undirected graph";
const char* url = "k_truss";

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<filetype: txt,adj,mtx,gr>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<filename: symmetrized graph>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("trussNum of k-truss (default value 3)"), cll::init(3));
static cll::opt<std::string> outName("o", cll::desc("output file for the edgelist of resulting truss"));
//typedef galois::graphs::LC_CSR_Graph<uint32_t, void>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;
typedef std::pair<GNode, GNode> NodePair;
typedef galois::InsertBag<NodePair> EdgeVec;

#define CHUNK_SIZE 256
#include "Mining/util.h"

static const uint32_t valid   = 0x0;
static const uint32_t removed = 0x1;

void initialize(Graph& g) {
	g.sortAllEdgesByDst();
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const GNode& n) {
		for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) g.getEdgeData(e) = valid;
	}, galois::steal());
}

void reportKTruss(Graph& g) {
	if (outName.empty()) return;
	std::ofstream of(outName);
	if (!of.is_open()) {
		std::cerr << "Cannot open " << outName << " for output." << std::endl;
		return;
	}
	for (auto n : g) {
		for (auto e : g.edges(n, galois::MethodFlag::UNPROTECTED)) {
			auto dst = g.getEdgeDst(e);
			if (n < dst && (g.getEdgeData(e) & 0x1) != removed)
				of << n << " " << dst << std::endl;
		}
	}
}

bool isSupportNoLessThanJ(Graph& g, GNode src, GNode dst, unsigned j) {
	size_t numValidEqual = 0;
	auto srcI = g.edge_begin(src, galois::MethodFlag::UNPROTECTED),
		 srcE = g.edge_end(src, galois::MethodFlag::UNPROTECTED),
		 dstI = g.edge_begin(dst, galois::MethodFlag::UNPROTECTED),
		 dstE = g.edge_end(dst, galois::MethodFlag::UNPROTECTED);
	while (true) {
		// find the first valid edge
		while (srcI != srcE && (g.getEdgeData(srcI) & removed)) ++srcI;
		while (dstI != dstE && (g.getEdgeData(dstI) & removed)) ++dstI;
		if (srcI == srcE || dstI == dstE) return numValidEqual >= j;
		// check for intersection
		auto sN = g.getEdgeDst(srcI), dN = g.getEdgeDst(dstI);
		if (sN < dN) ++srcI;
		else if (dN < sN) ++dstI;
		else {
			numValidEqual += 1;
			if (numValidEqual >= j) return true;
			++srcI;
			++dstI;
		}
	}
	return numValidEqual >= j;
}

// BSPTrussJacobiAlgo:
// 1. Scan for unsupported edges.
// 2. If no unsupported edges are found, done.
// 3. Remove unsupported edges in a separated loop.
// 4. Go back to 1.
void TrussJacobi(Graph& g) {
	if (k == 2) return;
	EdgeVec unsupported;
	EdgeVec cur, next;
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const GNode& src) {
		for (auto e : g.edges(src, galois::MethodFlag::UNPROTECTED)) {
			auto dst = g.getEdgeDst(e);
			if (dst > src) cur.push_back(std::make_pair(src, dst));
		}
	}, galois::steal());

	while (true) {
		galois::do_all(galois::iterate(cur), [&](NodePair& e) {
			EdgeVec& w = isSupportNoLessThanJ(g, e.first, e.second, k-2) ? next : unsupported;
			w.push_back(e);
		}, galois::steal());
		if (std::distance(unsupported.begin(), unsupported.end()) == 0) break;
		// mark unsupported edges as removed
		galois::do_all(galois::iterate(unsupported), [&](NodePair e) {
			g.getEdgeData(g.findEdgeSortedByDst(e.first, e.second)) = removed;
			g.getEdgeData(g.findEdgeSortedByDst(e.second, e.first)) = removed;
		}, galois::steal());
		unsupported.clear();
		cur.clear();
		cur.swap(next);
	}
} // end operator()

int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	assert(k >= 2);
	Graph graph;
	galois::StatTimer Tinit("GraphReadingTime");
	Tinit.start();
	read_graph(graph, filetype, filename);
	initialize(graph);
	Tinit.stop();
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	std::cout << "Running k-truss algorithm for maximal " << k << "-truss\n";
	galois::StatTimer Tcomp("ComputeTime");
	Tcomp.start();
	TrussJacobi(graph);
	Tcomp.stop();
	std::cout << "Done\n\n";
	//reportKTruss(graph);
	return 0;
}
