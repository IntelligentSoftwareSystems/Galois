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
#include <utility>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

const char* name = "FSM";
const char* desc = "Frequent subgraph mining";
const char* url  = 0;

enum Algo {
	nodeiterator,
	edgeiterator,
};

namespace cll = llvm::cl;
static cll::opt<std::string>
inputFilename(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<Algo> algo("algo", cll::desc("Choose an algorithm:"), cll::values(
	clEnumValN(Algo::nodeiterator, "nodeiterator", "Node Iterator"),
	clEnumValN(Algo::edgeiterator, "edgeiterator", "Edge Iterator"), clEnumValEnd),
	cll::init(Algo::nodeiterator));
static cll::opt<unsigned> k("k",
	cll::desc("max number of vertices in k-motif (default value 0)"), cll::init(0));
static cll::opt<unsigned> minsup("minsup",
	cll::desc("minimum suuport (default value 0)"), cll::init(0));
typedef galois::graphs::LC_CSR_Graph<uint32_t, uint32_t>::with_numa_alloc<true>::type ::with_no_lockable<true>::type Graph;
typedef Graph::GraphNode GNode;

#include "Dfscode/miner.h"
typedef galois::substrate::PerThreadStorage<LocalStatus> Status;

void init(CGraph& graph, Projected_map3 &root, std::deque<DFS> &queue) {
	EdgeList edges;
	int single_edge_dfscodes = 0;
	printf("\n=============================== Init ===============================\n\n");
	for (unsigned int from = 0; from < graph.size(); ++from) {
		if (get_forward_root(graph, graph[from], edges)) {   // get the edge list of the node g[from] in graph g
			for (EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
				//embeddings with a single edge
				int from_label = graph[from].label;
				if (root.count(from_label) == 0 || root[from_label].count((*it)->elabel) == 0 || root[from_label][(*it)->elabel].count(graph[(*it)->to].label) == 0) {
					single_edge_dfscodes++;
				}
				root[graph[from].label][(*it)->elabel][graph[(*it)->to].label].push(0, *it, 0);
			} //for
		} // if
	} // for from
	std::cout << "Single edge DFScodes " << single_edge_dfscodes << std::endl;
	int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / numThreads);
	std::cout << "dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
	for(Projected_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
		for(Projected_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
			for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
				DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
				queue.push_back(dfs);
			} // for tolabel
		} // for elabel
	} // for fromlabel
}

void init(Graph& graph, Miner& miner, Projected_map3 &root, std::deque<DFS> &queue) {
	int single_edge_dfscodes = 0;
	printf("\n=============================== Init ===============================\n\n");
	/*
	EdgeList edges;
	for (auto from : graph) {
		if (get_forward_root(graph, miner.edge_list, from, edges)) {
			auto& from_label = graph.getData(from);
			for (EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
				if (root.count(from_label) == 0 || root[from_label].count((*it)->elabel) == 0 || root[from_label][(*it)->elabel].count(graph.getData((*it)->to)) == 0) {
					single_edge_dfscodes++;
				}
				root[from_label][(*it)->elabel][graph.getData((*it)->to)].push(0, *it, 0);
			}
		}
	}
	//*/
	///*
	for (auto src : graph) {
		auto& src_label = graph.getData(src);
		Graph::edge_iterator first = graph.edge_begin(src);
		Graph::edge_iterator last = graph.edge_end(src);
		for (auto e = first; e != last; ++ e) {
			GNode dst = graph.getEdgeDst(e);
			auto elabel = graph.getEdgeData(e);
			auto& dst_label = graph.getData(dst);
			if (src_label <= dst_label) {
				if (root.count(src_label) == 0 || root[src_label].count(elabel) == 0 || root[src_label][elabel].count(dst_label) == 0)
					single_edge_dfscodes++;
				LabEdge *eptr = &(miner.edge_list[*e]);
				root[src_label][elabel][dst_label].push(0, eptr, 0);
			}
		}
	}
	//*/
	int dfscodes_per_thread =  (int) ceil((single_edge_dfscodes * 1.0) / numThreads);
	std::cout << "Single edge DFScodes " << single_edge_dfscodes << ", dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
	for(Projected_iterator3 fromlabel = root.begin(); fromlabel != root.end(); ++fromlabel) {
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
	std::cout << "=============================== Start ===============================\n";
	Projected_map3 root;
	std::deque<DFS> task_queue;
	init(graph, miner, root, task_queue);
	//init(cgraph, root, task_queue);
	//if(DEBUG) printout_embeddings(miner, queue);
	for(unsigned i = 0; i < status.size(); i++)
		status.getLocal(i)->frequent_patterns_count = 0;

	std::cout << "=============================== DFS ===============================\n";
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
			miner.project(root[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, *ls);
			ls->DFS_CODE.pop();
		},
		galois::chunk_size<4>(), galois::steal(), galois::no_conflicts(),
		galois::wl<galois::worklists::PerSocketChunkFIFO<128>>(),
		galois::loopname("ParallelDFS")
	);
	for(unsigned i = 0; i < numThreads; i++)
		total += status.getLocal(i)->frequent_patterns_count;
	std::cout << "Number of frequent subgraphs (minsup=" << minsup << "): " << total << "\n";
	std::cout << "=============================== Done ===============================\n";
}
 
int main(int argc, char** argv) {
	galois::SharedMemSys G;
	LonestarStart(argc, argv, name, desc, url);
	Graph graph;
	CGraph cgraph;
	galois::StatTimer Tinitial("GraphReadingTime");
	Tinitial.start();
	galois::graphs::readGraph(graph, inputFilename);
	for (GNode n : graph) {
		graph.getData(n) = rand() % 10 + 1;
		for (auto e : graph.edges(n)) {
			graph.getEdgeData(e) = 1;
		}
	}
	Miner miner(&graph, k, minsup, numThreads, true);
	miner.construct_edgelist();
	miner.construct_cgraph();
	Tinitial.stop();

	std::cout << "k = " << k << std::endl;
	std::cout << "minsup = " << minsup << std::endl;
	std::cout << "num_threads = " << numThreads << std::endl;
	galois::gPrint("num_vertices ", graph.size(), " num_edges ", graph.sizeEdges(), "\n");

	galois::reportPageAlloc("MeminfoPre");
	galois::StatTimer T;
	T.start();
	switch (algo) {
		case nodeiterator:
			FsmSolver(graph, miner);
			break;
		case edgeiterator:
			break;
		default:
			std::cerr << "Unknown algo: " << algo << "\n";
	}
	T.stop();
	galois::reportPageAlloc("MeminfoPost");
	return 0;
}
