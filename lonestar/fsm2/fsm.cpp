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

#define ENABLE_LABEL
#define DEBUG 0

const char* name = "FSM";
const char* desc = "Frequent subgraph mining using DFS code";
const char* url  = 0;
namespace cll = llvm::cl;
static cll::opt<std::string> filetype(cll::Positional, cll::desc("<file type>"), cll::Required);
static cll::opt<std::string> filename(cll::Positional, cll::desc("<file name>"), cll::Required);
static cll::opt<unsigned> k("k", cll::desc("max number of vertices in k-motif (default value 0)"), cll::init(0));
static cll::opt<unsigned> minsup("minsup", cll::desc("minimum suuport (default value 0)"), cll::init(0));
static cll::opt<unsigned> show("s", cll::desc("print out the frequent patterns"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<int, int>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Dfscode/miner.h"
#include "Lonestar/mgraph.h"
#include "Mining/util.h"
#define CHUNK_SIZE 4

typedef galois::substrate::PerThreadStorage<LocalStatus> Status;
typedef galois::InsertBag<DFS> PatternQueue;
typedef std::deque<DFS> DFSQueue;

void init(Graph& graph, Miner& miner, PatternMap3D &pattern_map, PatternQueue &queue) {
	int single_edge_dfscodes = 0;
	int num_embeddings = 0;
	printf("\n=============================== Init ===============================\n\n");
	// classify each edge into its single-edge pattern accodding to its (src_label, edge_label, dst_label)
	for (auto src : graph) {
		auto& src_label = graph.getData(src);
		Graph::edge_iterator first = graph.edge_begin(src);
		Graph::edge_iterator last = graph.edge_end(src);
		for (auto e = first; e != last; ++ e) {
			GNode dst = graph.getEdgeDst(e);
			auto elabel = graph.getEdgeData(e);
			auto& dst_label = graph.getData(dst);
			//if (src_label <= dst_label) { // when src_label == dst_label (the edge will be added twice since the input graph is symmetrized)
			if (src_label < dst_label || (src_label == dst_label && src < dst)) {
				if (pattern_map.count(src_label) == 0 || pattern_map[src_label].count(elabel) == 0 || pattern_map[src_label][elabel].count(dst_label) == 0)
					single_edge_dfscodes++;
				LabEdge *eptr = &(miner.edge_list[*e]);
				pattern_map[src_label][elabel][dst_label].push(2, eptr, 0); // single-edge embedding: (num_vertices, edge, pointer_to_parent_embedding)
				num_embeddings ++;
			}
		}
	}
	int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / numThreads);
	std::cout << "num_single_edge_patterns = " << single_edge_dfscodes << ", dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
	std::cout << "num_embeddings = " << num_embeddings << std::endl; 
	// for each single-edge pattern, generate a DFS code and push it into the task queue
	for(EmbeddingList_iterator3 fromlabel = pattern_map.begin(); fromlabel != pattern_map.end(); ++fromlabel) {
		for(EmbeddingList_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
			for(EmbeddingList_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
				DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
				queue.push_back(dfs);
			} // for tolabel
		} // for elabel
	} // for fromlabel
}

void FsmSolver(Graph& graph, Miner& miner) {
	Status status;
	std::cout << "\n=============================== Start ===============================\n";
	PatternMap3D pattern_map; // mapping patterns to their embedding list
	PatternQueue task_queue; // task queue holding the DFScodes of patterns
	init(graph, miner, pattern_map, task_queue); // insert single-edge patterns into the queue
	//if(DEBUG) printout_embeddings(miner, queue);
	//std::cout << "numThreads = " << numThreads << " status.size() = " << status.size() << "\n";
	for(size_t i = 0; i < status.size(); i++) {
		status.getLocal(i)->frequent_patterns_count = 0;
		status.getLocal(i)->thread_id = i;
#ifdef ENABLE_LB
		status.getLocal(i)->task_split_level = 0;
		status.getLocal(i)->embeddings_regeneration_level = 0;
		status.getLocal(i)->is_running = true;
#endif
	}

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
#ifdef ENABLE_LB
			if(ls->dfs_task_queue[0].size() == 0 && !miner.all_threads_idle()) {
				bool succeed = miner.try_task_stealing(ls);
				if (succeed) {
					ls->embeddings_regeneration_level = 0;
					miner.set_regen_level(ls->thread_id, 0);
				}
			}
#endif
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
	Miner miner(&graph, k, minsup, numThreads, show);
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
