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
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "Lonestar/common_types.h"

const char* name = "FSM";
const char* desc = "Frequent subgraph mining using DFS code";
const char* url  = 0;

enum Algo {
	nodeiterator,
	edgeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<file type>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<file name>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), cll::values(
	clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd),
	cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif (default value 0)"), cll::init(0));
static cll::opt<unsigned> minsup("minsup", cll::desc("minimum suuport (default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<int, int>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Dfscode/miner.h"
#include "Lonestar/mgraph.h"
#include "Mining/util.h"
#define CHUNK_SIZE 4

typedef galois::substrate::PerThreadStorage<LocalStatus> Status;

void init(Graph& graph, Miner& miner, Projected_map3 &pattern_map, std::deque<DFS> &queue) {
	int single_edge_dfscodes = 0;
	printf("\n=============================== Init ===============================\n\n");
	for (auto src : graph) {
		auto& src_label = graph.getData(src);
		Graph::edge_iterator first = graph.edge_begin(src);
		Graph::edge_iterator last = graph.edge_end(src);
		for (auto e = first; e != last; ++ e) {
			GNode dst = graph.getEdgeDst(e);
			auto elabel = graph.getEdgeData(e);
			auto& dst_label = graph.getData(dst);
			if (src_label <= dst_label) {
				if (pattern_map.count(src_label) == 0 || pattern_map[src_label].count(elabel) == 0 || pattern_map[src_label][elabel].count(dst_label) == 0)
					single_edge_dfscodes++;
				LabEdge *eptr = &(miner.edge_list[*e]);
				pattern_map[src_label][elabel][dst_label].push(0, eptr, 0);
			}
		}
	}
	int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / numThreads);
	std::cout << "Single edge DFScodes " << single_edge_dfscodes << ", dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
	for(Projected_iterator3 fromlabel = pattern_map.begin(); fromlabel != pattern_map.end(); ++fromlabel) {
		for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
			for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
				DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
				queue.push_back(dfs);
			} // for tolabel
		} // for elabel
	} // for fromlabel
}

#define DEBUG 0
#define SHOW_OUTPUT 0

void FsmSolver(Graph& graph, Miner& miner) {
	Status status;
	std::cout << "\n=============================== Start ===============================\n";
	Projected_map3 pattern_map;
	std::deque<DFS> task_queue;
	init(graph, miner, pattern_map, task_queue);
	//if(DEBUG) printout_embeddings(miner, queue);
	for(size_t i = 0; i < status.size(); i++)
		status.getLocal(i)->frequent_patterns_count = 0;

	std::cout << "\n=============================== DFS ===============================\n";
	size_t total = 0;
	galois::for_each(
		galois::iterate(task_queue),
		[&](const DFS& dfs, auto& ctx) {
			LocalStatus *ls = status.getLocal();
			ls->current_dfs_level = 0;
			std::deque<DFS> tmp;
			ls->dfs_task_queue.push_back(tmp);
			ls->dfs_task_queue[0].push_back(dfs);
			ls->DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			miner.grow(pattern_map[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, *ls);
			ls->DFS_CODE.pop();
		},
		galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<CHUNK_SIZE>>(),
		galois::loopname("ParallelDFS")
	);
	for(int i = 0; i < numThreads; i++)
		total += status.getLocal(i)->frequent_patterns_count;
	std::cout << "\n\tnum_frequent_patterns (minsup=" << minsup << "): " << total << "\n";
	std::cout << "\n=============================== Done ===============================\n\n";
}
 
int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	MGraph mgraph;
	galois::StatTimer Tinit("GraphReading");
	Tinit.start();
	if (filetype == "txt") {
		printf("Reading .lg file: %s\n", filename.c_str());
		mgraph.read_txt(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "adj") {
		printf("Reading .adj file: %s\n", filename.c_str());
		mgraph.read_adj(filename.c_str());
		genGraph(mgraph, graph);
	} else if (filetype == "gr") {
		printf("Reading .gr file: %s\n", filename.c_str());
		galois::graphs::readGraph(graph, filename);
		for (GNode n : graph) {
			graph.getData(n) = rand() % 10 + 1;
			for (auto e : graph.edges(n)) {
				graph.getEdgeData(e) = 1;
			}
		}
	} else { printf("Unkown file format\n"); exit(1); }
	Miner miner(&graph, k, minsup, numThreads, true);
	Tinit.stop();

	std::cout << "k = " << k << std::endl;
	std::cout << "minsup = " << minsup << std::endl;
	std::cout << "num_threads = " << numThreads << std::endl;
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");
	galois::StatTimer Tcomp("Compute");
	Tcomp.start();
	FsmSolver(graph, miner);
	Tcomp.stop();
	return 0;
}
