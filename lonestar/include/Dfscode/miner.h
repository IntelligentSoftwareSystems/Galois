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
#include "embedding.h"

class Miner {
public:
	std::vector<LabEdge> edge_list;
	Miner(Graph *g, unsigned k, unsigned minsup, int num_threads, bool label_f = true, bool enable_threshold = true) {
		this->graph = g;
		minimal_support = minsup;
		max_size = k;
		label_flag = label_f;
		nthreads = num_threads;
		for(int i = 0; i < nthreads; i++) {
			frequent_patterns_count.push_back(0);
			std::vector<std::deque<DFS> > tmp;
			dfs_task_queue.push_back(tmp);
		}
		construct_edgelist();
#ifdef ENABLE_LB
		for(int i = 0; i < nthreads; i++) {
			dfs_task_queue_shared.push_back(tmp);
			thread_is_working.push_back(false);
			embeddings_regeneration_level.push_back(0);
		}
		task_split_threshold = 2;
		init_lb(0, nthreads);
#endif
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
#ifdef ENABLE_LB
	void set_regen_level(int tid, int val) {
		embeddings_regeneration_level[tid] = val;
	}
	void try_task_stealing(LocalStatus &status) {
		threads_load_balance(status);
	}
	void activate_thread(int tid) {
		simple_lock.lock();
		thread_is_working[tid] = true;
		simple_lock.unlock();
	}
	void deactivate_thread(int tid) {
		simple_lock.lock();
		thread_is_working[tid] = false;
		simple_lock.unlock();
	}
	bool all_threads_idle() {
		bool all_idle = true;
		simple_lock.lock();
		for(int i = 0; i< num_threads; i++) {
			if(thread_is_working[i] == true) {
				all_idle = false;
				break;
			}
		}
		simple_lock.unlock();
		return all_idle;
	}
	bool thread_working(LocalStatus &status) {
		int thread_id = status.thread_id;
		bool th_is_working;
		simple_lock.lock();
		th_is_working = thread_is_working[thread_id];
		simple_lock.unlock();
		return th_is_working;
	}
#endif
	// edge extension by DFS traversal: recursive call
	void grow(EmbeddingList &emb_list, unsigned dfs_level, LocalStatus &status) {
		unsigned sup = support(emb_list, status);
		if (sup < minimal_support) return;
		if (is_min(status) == false) return; // check if this pattern is canonical: minimal DFSCode
		status.frequent_patterns_count ++;
		// list frequent patterns here!!!
		if(SHOW_OUTPUT) {
			std::cout << status.DFS_CODE.to_string(false) << ": " << sup << std::endl;
			for (auto it = emb_list.begin(); it != emb_list.end(); it++) std::cout << "\t" << it->to_string_all() << std::endl;
		}
		const RMPath &rmpath = status.DFS_CODE.buildRMPath(); // build the right-most path of this pattern
		LabelT minlabel = status.DFS_CODE[0].fromlabel; 
		VeridT maxtoc = status.DFS_CODE[rmpath[0]].to; // right-most vertex
		PatternMap3D new_fwd_root;
		PatternMap2D new_bck_root;
		EdgeList edges;
		status.current_dfs_level = dfs_level;
		// take each embedding in the embedding list, do backward and forward extension
		for(size_t n = 0; n < emb_list.size(); ++n) {
			Embedding *cur = &emb_list[n];
			unsigned emb_size = cur->num_vertices;
			History history(cur);
			// backward
			for(size_t i = rmpath.size() - 1; i >= 1; --i) {
				LabEdge *e = get_backward(*graph, edge_list, history[rmpath[i]], history[rmpath[0]], history);
				if(e) new_bck_root[status.DFS_CODE[rmpath[i]].from][e->elabel].push(emb_size, e, cur);
			}
			// pure forward
			if (get_forward_pure(*graph, edge_list, history[rmpath[0]], minlabel, history, edges)) {
				for (EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					if (emb_size + 1 <= max_size)
						new_fwd_root[maxtoc][(*it)->elabel][graph->getData((*it)->to)].push(emb_size+1, *it, cur);
				}
			}
			// backtracked forward
			for(size_t i = 0; i < rmpath.size(); ++i) {
				if(get_forward_rmpath(*graph, edge_list, history[rmpath[i]], minlabel, history, edges)) {
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
						if (emb_size + 1 <= max_size)
							new_fwd_root[status.DFS_CODE[rmpath[i]].from][(*it)->elabel][graph->getData((*it)->to)].push(emb_size+1, *it, cur);
					} // for it
				} // if
			} // for i
		} // for n
		std::deque<DFS> tmp;
		if(status.dfs_task_queue.size() <= dfs_level) {
			status.dfs_task_queue.push_back(tmp);
		}
		// insert all extended subgraphs into the task queue
		// backward
		for(EmbeddingList_iterator2 to = new_bck_root.begin(); to != new_bck_root.end(); ++to) {
			for(EmbeddingList_iterator1 elabel = to->second.begin(); elabel != to->second.end(); ++elabel) {
				DFS dfs(maxtoc, to->first, (LabelT)-1, elabel->first, (LabelT)-1);
				status.dfs_task_queue[dfs_level].push_back(dfs);
			}
		}
		// forward
		for(EmbeddingList_riterator3 from = new_fwd_root.rbegin(); from != new_fwd_root.rend(); ++from) {
			for(EmbeddingList_iterator2 elabel = from->second.begin(); elabel != from->second.end(); ++elabel) {
				for(EmbeddingList_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					DFS dfs(from->first, maxtoc + 1, (LabelT)-1, elabel->first, tolabel->first);
					status.dfs_task_queue[dfs_level].push_back(dfs);
				}
			}
		}
		// grow to the next level
		while(status.dfs_task_queue[dfs_level].size() > 0) {
			#ifdef ENABLE_LB
			if(nthreads > 1) threads_load_balance(status);
			#endif
			DFS dfs = status.dfs_task_queue[dfs_level].front();
			status.dfs_task_queue[dfs_level].pop_front();
			status.current_dfs_level = dfs_level;
			status.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			if(dfs.is_backward())
				grow(new_bck_root[dfs.to][dfs.elabel], dfs_level + 1, status);
			else
				grow(new_fwd_root[dfs.from][dfs.elabel][dfs.tolabel], dfs_level + 1, status);
			status.DFS_CODE.pop();
		}
		return;
	}
	// This is used for load balancing
	void regenerate_embeddings(EmbeddingList &projected, unsigned dfs_level, LocalStatus &status) {
		for(size_t i = 0; status.dfs_task_queue[dfs_level].size() > 0; i++) {
			status.current_dfs_level = dfs_level;
			#ifdef ENABLE_LB
			if(num_threads > 1) threads_load_balance(status);
			#endif
			DFS dfs = status.dfs_task_queue[dfs_level].front();
			status.dfs_task_queue[dfs_level].pop_front();
			status.DFS_CODE.push(dfs.from, dfs.to, dfs.fromlabel, dfs.elabel, dfs.tolabel);
			EmbeddingList new_root;
			for(size_t n = 0; n < projected.size(); ++n) {
				Embedding *cur = &projected[n];
				unsigned emb_size = cur->num_vertices;
				History history(cur);
				if(dfs.is_backward()) {
					LabEdge *e = get_backward(*graph, edge_list, status.DFS_CODE, history);
					if(e) new_root.push(emb_size, e, cur);
				} else {
					EdgeList edges;
					if(get_forward(*graph, edge_list, status.DFS_CODE, history, edges)) {
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							if (emb_size + 1 <= max_size)
								new_root.push(emb_size+1, *it, cur);
					}
				}
			}
			if (status.embeddings_regeneration_level > dfs_level) {
				regenerate_embeddings(new_root, dfs_level + 1, status);
			} else grow(new_root, dfs_level + 1, status);
			status.DFS_CODE.pop();
		}
	}

protected:
	Graph *graph;
	int nthreads;
	unsigned minimal_support;
	unsigned max_size;
	bool label_flag;
	std::vector<int> frequent_patterns_count;
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue;       //keep the sibling extensions for each level and for each thread
#ifdef ENABLE_LB
	int task_split_threshold;
	std::vector<bool> thread_is_working;
	std::vector<int> embeddings_regeneration_level;
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue_shared;       //keep a separate queue for sharing work
#endif
	//support function for a single large graph, computes the minimum count of a node in the embeddings
	virtual unsigned support(EmbeddingList &projected, LocalStatus &status) {
		Map2D node_id_counts;
		for(EmbeddingList::iterator cur = projected.begin(); cur != projected.end(); ++cur) {
			Embedding *em = &(*cur);
			size_t dfsindex = status.DFS_CODE.size() - 1;
			while(em != NULL) {
				if(status.DFS_CODE[dfsindex].to > status.DFS_CODE[dfsindex].from) {    //forward edge
					node_id_counts[status.DFS_CODE[dfsindex].to][em->edge->to]++;
				}
				if(!em->prev) {
					node_id_counts[status.DFS_CODE[dfsindex].from][em->edge->from]++;
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
	bool is_min(LocalStatus &status) {
		if(status.DFS_CODE.size() == 1) return true;
		status.DFS_CODE.toGraph(status.GRAPH_IS_MIN);
		status.DFS_CODE_IS_MIN.clear();
		PatternMap3D root;
		EdgeList edges;
		for(size_t from = 0; from < status.GRAPH_IS_MIN.size(); ++from) {
			if(get_forward_root(status.GRAPH_IS_MIN, status.GRAPH_IS_MIN[from], edges)) {
			//if(get_forward_root(status.GRAPH_IS_MIN, from, edges)) {
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[status.GRAPH_IS_MIN[from].label][(*it)->elabel][status.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0);
				} // for it
			} // if get_forward_root
		} // for from
		EmbeddingList_iterator3 fromlabel = root.begin();
		EmbeddingList_iterator2 elabel = fromlabel->second.begin();
		EmbeddingList_iterator1 tolabel = elabel->second.begin();
		status.DFS_CODE_IS_MIN.push(0, 1, fromlabel->first, elabel->first, tolabel->first);
		return (project_is_min(status,tolabel->second));
	}

	bool project_is_min(LocalStatus &status, EmbeddingList &projected) {
		const RMPath& rmpath = status.DFS_CODE_IS_MIN.buildRMPath();
		LabelT minlabel         = status.DFS_CODE_IS_MIN[0].fromlabel;
		VeridT maxtoc           = status.DFS_CODE_IS_MIN[rmpath[0]].to;

		// SUBBLOCK 1
		{
			PatternMap1D root;
			bool flg = false;
			VeridT newto = 0;

			for(size_t i = rmpath.size() - 1; !flg  && i >= 1; --i) {
				for(size_t n = 0; n < projected.size(); ++n) {
					Embedding *cur = &projected[n];
					History history(cur);
					LabEdge *e = get_backward(status.GRAPH_IS_MIN, history[rmpath[i]], history[rmpath[0]], history);
					if(e) {
						root[e->elabel].push(0, e, cur);
						newto = status.DFS_CODE_IS_MIN[rmpath[i]].from;
						flg = true;
					} // if e
				} // for n
			} // for i

			if(flg) {
				EmbeddingList_iterator1 elabel = root.begin();
				status.DFS_CODE_IS_MIN.push(maxtoc, newto, (LabelT)-1, elabel->first, (LabelT)-1);
				if(status.DFS_CODE[status.DFS_CODE_IS_MIN.size() - 1] != status.DFS_CODE_IS_MIN[status.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(status, elabel->second);
			}
		} // SUBBLOCK 1

		// SUBBLOCK 2
		{
			bool flg = false;
			VeridT newfrom = 0;
			PatternMap2D root;
			EdgeList edges;

			for(size_t n = 0; n < projected.size(); ++n) {
				Embedding *cur = &projected[n];
				History history(cur);
				if(get_forward_pure(status.GRAPH_IS_MIN, history[rmpath[0]], minlabel, history, edges)) {
					flg = true;
					newfrom = maxtoc;
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
						root[(*it)->elabel][status.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
				} // if get_forward_pure
			} // for n
			for(size_t i = 0; !flg && i < rmpath.size(); ++i) {
				for(size_t n = 0; n < projected.size(); ++n) {
					Embedding *cur = &projected[n];
					History history(cur);
					if(get_forward_rmpath(status.GRAPH_IS_MIN, history[rmpath[i]], minlabel, history, edges)) {
						flg = true;
						newfrom = status.DFS_CODE_IS_MIN[rmpath[i]].from;
						for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it)
							root[(*it)->elabel][status.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, cur);
					} // if get_forward_rmpath
				} // for n
			} // for i

			if(flg) {
				EmbeddingList_iterator2 elabel  = root.begin();
				EmbeddingList_iterator1 tolabel = elabel->second.begin();
				status.DFS_CODE_IS_MIN.push(newfrom, maxtoc + 1, (LabelT)-1, elabel->first, tolabel->first);
				if(status.DFS_CODE[status.DFS_CODE_IS_MIN.size() - 1] != status.DFS_CODE_IS_MIN[status.DFS_CODE_IS_MIN.size() - 1]) return false;
				return project_is_min(status, tolabel->second);
			} // if(flg)
		} // SUBBLOCK 2
		return true;
	} // end project_is_min
};

