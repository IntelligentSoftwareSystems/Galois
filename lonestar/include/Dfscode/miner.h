// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>

#include <map>
#include <set>
#include <omp.h>
#include <deque>
#include <vector>
#include <math.h>
#include <cstdio>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "graph_types.hpp"
//#define USE_CGRAPH
class Miner {
public:
	std::vector<LabEdge> edge_list;
	Miner(Graph *g, unsigned k, unsigned minsup, int num_threads, bool label_f = true, bool enable_threshold = true) {
		this->graph = g;
		minimal_support = minsup;
		max_level = k;
		label_flag = label_f;
		use_threshold = enable_threshold;
		nthreads = num_threads;
		for(int i = 0; i < nthreads; i++) {
			frequent_patterns_count.push_back(0);
			std::vector<std::deque<DFS> > tmp;
			dfs_task_queue.push_back(tmp);
			dfs_task_queue_shared.push_back(tmp);
		}
		construct_edgelist();
	}
	virtual ~Miner() {}
	// returns the total number of frequent patterns
	size_t get_count() {
		size_t total = 0;
		for(int i = 0; i < nthreads; i++)
			total += frequent_patterns_count[i];
		return total;
	}
	void construct_edgelist() {
		unsigned eid = 0;
		for (Graph::iterator it = graph->begin(); it != graph->end(); it ++) {
			GNode src = *it;
			//auto& src_label = graph->getData(src);
			Graph::edge_iterator first = graph->edge_begin(src);
			Graph::edge_iterator last = graph->edge_end(src);
			// foe each edge of this vertex
			for (auto e = first; e != last; ++ e) {
				GNode dst = graph->getEdgeDst(e);
				auto& elabel = graph->getEdgeData(e);
				//auto& dst_label = graph->getData(dst);
				LabEdge edge(src, dst, elabel, eid);
				edge_list.push_back(edge);
				eid ++;
			}
		}
		assert(edge_list.size() == graph->sizeEdges());
	}
/*
	void construct_cgraph() {
		cgraph.resize(graph->size());
		for (Graph::iterator it = graph->begin(); it != graph->end(); it ++) {
			GNode src = *it;
			cgraph[src].global_vid = src;
			auto& src_label = graph->getData(src);
			cgraph[src].label = src_label;
			Graph::edge_iterator first = graph->edge_begin(src);
			Graph::edge_iterator last = graph->edge_end(src);
			// foe each edge of this vertex
			for (auto e = first; e != last; ++ e) {
				GNode dst = graph->getEdgeDst(e);
				auto& elabel = graph->getEdgeData(e);
				cgraph[src].push(src, dst, elabel);
			}
		}
		cgraph.max_local_vid = cgraph.size() - 1;
		cgraph.buildEdge();
	}
//*/
	// edge extension by DFS traversal: recursive call
	void grow(Projected &emb_list, unsigned dfs_level, LocalStatus &gprv) {
		if (dfs_level >= max_level) return; // TODO: check the number of vertices in the embedding
		if (use_threshold) {
			unsigned sup = support(emb_list, gprv);
			if (sup < minimal_support) return;
		}
		if (is_min(gprv) == false) return; // check if this pattern is canonical: minimal DFSCode
		if (use_threshold) {
			gprv.frequent_patterns_count ++;
			// list frequent patterns here!!!
		} else std::cout << "motif_count = " << emb_list.size() << std::endl;
		const RMPath &rmpath = gprv.DFS_CODE.buildRMPath(); // build the right-most path of this pattern
		LabelT minlabel = gprv.DFS_CODE[0].fromlabel; 
		VeridT maxtoc = gprv.DFS_CODE[rmpath[0]].to; // right-most vertex
		Projected_map3 new_fwd_root;
		Projected_map2 new_bck_root;
		EdgeList edges;
		gprv.current_dfs_level = dfs_level;
		for(size_t n = 0; n < emb_list.size(); ++n) {
			unsigned id = emb_list[n].id;
			PDFS *cur = &emb_list[n];
			History history(cur);
			// backward
			for(size_t i = rmpath.size() - 1; i >= 1; --i) {
#ifdef USE_CGRAPH
				LabEdge *e = get_backward(cgraph, history[rmpath[i]], history[rmpath[0]], history);
#else
				LabEdge *e = get_backward(*graph, edge_list, history[rmpath[i]], history[rmpath[0]], history);
#endif
				if(e) new_bck_root[gprv.DFS_CODE[rmpath[i]].from][e->elabel].push(id, e, cur);
			}
			// pure forward
#ifdef USE_CGRAPH
			if(get_forward_pure(cgraph, history[rmpath[0]], minlabel, history, edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					new_fwd_root[maxtoc][(*it)->elabel][cgraph[(*it)->to].label].push(id, *it, cur);
#else
			if(get_forward_pure(*graph, edge_list, history[rmpath[0]], minlabel, history, edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					new_fwd_root[maxtoc][(*it)->elabel][graph->getData((*it)->to)].push(id, *it, cur);
#endif
				}
			}
			// backtracked forward
			for(size_t i = 0; i < rmpath.size(); ++i) {
#ifdef USE_CGRAPH
				if(get_forward_rmpath(cgraph, history[rmpath[i]], minlabel, history, edges)) {
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
						new_fwd_root[gprv.DFS_CODE[rmpath[i]].from][(*it)->elabel][cgraph[(*it)->to].label].push(id, *it, cur);
#else
				if(get_forward_rmpath(*graph, edge_list, history[rmpath[i]], minlabel, history, edges)) {
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
						new_fwd_root[gprv.DFS_CODE[rmpath[i]].from][(*it)->elabel][graph->getData((*it)->to)].push(id, *it, cur);
#endif
					} // for it
				} // if
			} // for i
		} // for n
		std::deque<DFS> tmp;
		if(gprv.dfs_task_queue.size() <= dfs_level) {
			gprv.dfs_task_queue.push_back(tmp);
		}
		// insert all extended subgraphs into the task queue
		// backward
		for(Projected_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
			for(Projected_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
				DFS dfs(maxtoc, to->first, (LabelT)-1, elabel->first, (LabelT)-1);
				gprv.dfs_task_queue[dfs_level].push_back(dfs);
			}
		}
		// forward
		for(Projected_riterator3 from = new_fwd_root.rbegin(); from != new_fwd_root.rend(); ++from) {
			for(Projected_iterator2 elabel = from->second.begin(); elabel != from->second.end(); ++elabel) {
				for(Projected_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					DFS dfs(from->first, maxtoc + 1, (LabelT)-1, elabel->first, tolabel->first);
					gprv.dfs_task_queue[dfs_level].push_back(dfs);
				}
			}
		}
		// grow to the next level
		while(gprv.dfs_task_queue[dfs_level].size() > 0) {
			#ifdef ENABLE_LB
			if(nthreads > 1) threads_load_balance(gprv);
			#endif
			DFS dfs = gprv.dfs_task_queue[dfs_level].front();
			gprv.dfs_task_queue[dfs_level].pop_front();
			gprv.current_dfs_level = dfs_level;
			gprv.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			if(dfs.is_backward())
				grow(new_bck_root[dfs.to][dfs.elabel], dfs_level + 1, gprv);
			else
				grow(new_fwd_root[dfs.from][dfs.elabel][dfs.tolabel], dfs_level + 1, gprv);
			gprv.DFS_CODE.pop();
		}
		return;
	}
	// This is used for load balancing
	void regenerate_embeddings(Projected &projected, unsigned dfs_level, LocalStatus &gprv) {
		for(size_t i = 0; gprv.dfs_task_queue[dfs_level].size() > 0; i++) {
			gprv.current_dfs_level = dfs_level;
			#ifdef ENABLE_LB
			if(num_threads > 1) threads_load_balance(gprv);
			#endif
			DFS dfs = gprv.dfs_task_queue[dfs_level].front();
			gprv.dfs_task_queue[dfs_level].pop_front();
			gprv.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			Projected new_root;
			for(size_t n = 0; n < projected.size(); ++n) {
				unsigned id = projected[n].id;
				PDFS *cur = &projected[n];
				History history(cur);
				if(dfs.is_backward()) {
					LabEdge *e = get_backward(*graph, edge_list, gprv.DFS_CODE, history);
					if(e) new_root.push(id, e, cur);
				} else {
					EdgeList edges;
					if(get_forward(*graph, edge_list, gprv.DFS_CODE, history, edges)) {
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							new_root.push(id, *it, cur);
					}
				}
			}
			if (gprv.embeddings_regeneration_level > dfs_level) {
				regenerate_embeddings(new_root, dfs_level + 1, gprv);
			} else grow(new_root, dfs_level + 1, gprv);
			gprv.DFS_CODE.pop();
		}
	}

protected:
	Graph *graph;
	CGraph cgraph;
	int nthreads;
	unsigned minimal_support;
	unsigned max_level;
	bool use_threshold;
	bool label_flag;
	std::vector<int> frequent_patterns_count;
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue;       //keep the sibling extensions for each level and for each thread
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue_shared;       //keep a separate queue for sharing work

	//support function for a single large graph, computes the minimum count of a node in the embeddings
	virtual unsigned support(Projected &projected, LocalStatus &gprv) {
		Map2D node_id_counts;
		for(Projected::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
			PDFS *em = &(*cur);
			size_t dfsindex = gprv.DFS_CODE.size() - 1;
			while(em != NULL) {
				if(gprv.DFS_CODE[dfsindex].to > gprv.DFS_CODE[dfsindex].from) {    //forward edge
					node_id_counts[gprv.DFS_CODE[dfsindex].to][em->edge->to]++;
				}
				if(!em->prev) {
					node_id_counts[gprv.DFS_CODE[dfsindex].from][em->edge->from]++;
				}
				em = em->prev;
				dfsindex--;
			}
		}
		unsigned min = 0xffffffff;
		for(Map2D::iterator it = node_id_counts.begin(); it != node_id_counts.end(); it++) {
			if((it->second).size() < min)
				min = (it->second).size();
		}
		if(min == 0xffffffff) min = 0;
		return min;
	}
	// check whether a DFSCode is minimal or not, i.e. canonical check (minimal DFSCode is canonical)
	bool is_min(LocalStatus &gprv) {
		if(gprv.DFS_CODE.size() == 1) return true;
		gprv.DFS_CODE.toGraph(gprv.GRAPH_IS_MIN);
		gprv.DFS_CODE_IS_MIN.clear();
		Projected_map3 root;
		EdgeList edges;
		for(size_t from = 0; from < gprv.GRAPH_IS_MIN.size(); ++from) {
			if(get_forward_root(gprv.GRAPH_IS_MIN, gprv.GRAPH_IS_MIN[from], edges)) {
			//if(get_forward_root(gprv.GRAPH_IS_MIN, from, edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[gprv.GRAPH_IS_MIN[from].label][(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0);
					//root[gprv.GRAPH_IS_MIN.getData(from)][(*it)->elabel][gprv.GRAPH_IS_MIN.getData((*it)->to)].push(0, *it, 0);
				} // for it
			} // if get_forward_root
		} // for from
		Projected_iterator3 fromlabel = root.begin();
		Projected_iterator2 elabel = fromlabel->second.begin();
		Projected_iterator1 tolabel = elabel->second.begin();
		gprv.DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
		return (project_is_min(gprv,tolabel->second));
	}

	bool project_is_min(LocalStatus &gprv, Projected &projected) {
		const RMPath& rmpath = gprv.DFS_CODE_IS_MIN.buildRMPath();
		LabelT minlabel         = gprv.DFS_CODE_IS_MIN[0].fromlabel;
		VeridT maxtoc           = gprv.DFS_CODE_IS_MIN[rmpath[0]].to;

		// SUBBLOCK 1
		{
			Projected_map1 root;
			bool flg = false;
			VeridT newto = 0;

			for(size_t i = rmpath.size() - 1; !flg  && i >= 1; --i) {
				for(size_t n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(cur);
					LabEdge *e = get_backward(gprv.GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
					if(e) {
						root[e->elabel].push(0, e, cur);
						newto = gprv.DFS_CODE_IS_MIN[rmpath[i]].from;
						flg = true;
					} // if e
				} // for n
			} // for i

			if(flg) {
				Projected_iterator1 elabel = root.begin();
				gprv.DFS_CODE_IS_MIN.push(maxtoc, newto, (LabelT)-1, elabel->first, (LabelT)-1);
				if(gprv.DFS_CODE[gprv.DFS_CODE_IS_MIN.size() - 1] != gprv.DFS_CODE_IS_MIN[gprv.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(gprv, elabel->second);
			}
		} // SUBBLOCK 1

		// SUBBLOCK 2
		{
			bool flg = false;
			VeridT newfrom = 0;
			Projected_map2 root;
			EdgeList edges;

			for(size_t n = 0; n < projected.size(); ++n) {
				PDFS *cur = &projected[n];
				History history(cur);
				if(get_forward_pure(gprv.GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
					flg = true;
					newfrom = maxtoc;
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
						root[(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
						//root[(*it)->elabel][gprv.GRAPH_IS_MIN.getData((*it)->to)].push(0, *it, cur);
				} // if get_forward_pure
			} // for n
			for(size_t i = 0; !flg && i < rmpath.size(); ++i) {
				for(size_t n = 0; n < projected.size(); ++n) {
					PDFS *cur = &projected[n];
					History history(cur);
					if(get_forward_rmpath(gprv.GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
						flg = true;
						newfrom = gprv.DFS_CODE_IS_MIN[rmpath[i]].from;
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							root[(*it)->elabel][gprv.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
							//root[(*it)->elabel][gprv.GRAPH_IS_MIN.getData((*it)->to)].push(0, *it, cur);
					} // if get_forward_rmpath
				} // for n
			} // for i

			if(flg) {
				Projected_iterator2 elabel  = root.begin();
				Projected_iterator1 tolabel = elabel->second.begin();
				gprv.DFS_CODE_IS_MIN.push(newfrom, maxtoc + 1, (LabelT)-1, elabel->first, tolabel->first);
				if(gprv.DFS_CODE[gprv.DFS_CODE_IS_MIN.size() - 1] != gprv.DFS_CODE_IS_MIN[gprv.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(gprv, tolabel->second);
			} // if(flg)
		} // SUBBLOCK 2
		return true;
	} // end project_is_min
};

