// Copyright 2019, University of Texas at Austin
// Authors: Xuhao Chen <cxh@utexas.edu>
#include <deque>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include "embedding.h"
typedef std::deque<DFS> DFSQueue;
typedef galois::InsertBag<DFS> PatternQueue;
typedef galois::substrate::PerThreadStorage<LocalStatus> Status;

class Miner {
public:
	std::vector<LabEdge> edge_list;
	Miner(Graph *g) {
		this->graph = g;
		minimal_support = minsup;
		max_level = k;
		nthreads = numThreads;
		show_output = show;
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
		init_lb();
		#endif
		for(size_t i = 0; i < status.size(); i++) {
			status.getLocal(i)->frequent_patterns_count = 0;
			status.getLocal(i)->thread_id = i;
			#ifdef ENABLE_LB
			status.getLocal(i)->task_split_level = 0;
			status.getLocal(i)->embeddings_regeneration_level = 0;
			status.getLocal(i)->is_running = true;
			#endif
		}
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
		for (auto src : *graph) {
			//auto& src_label = graph->getData(src);
			//auto first = graph->edge_begin(src);
			//auto last = graph->edge_end(src);
			// foe each edge of this vertex
			//for (auto e = first; e != last; ++ e) {
			for (auto e : graph->edges(src)) {
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
	void init_dfscode() {
		int single_edge_dfscodes = 0;
		int num_embeddings = 0;
		// classify each edge into its single-edge pattern accodding to its (src_label, edge_label, dst_label)
		for (auto src : *graph) {
			auto& src_label = graph->getData(src);
			//auto begin = graph->edge_begin(src);
			//auto end = graph->edge_end(src);
			for (auto e : graph->edges(src)) {
				GNode dst = graph->getEdgeDst(e);
				auto elabel = graph->getEdgeData(e);
				auto& dst_label = graph->getData(dst);
				if (src_label <= dst_label) { // when src_label == dst_label (the edge will be added twice since the input graph is symmetrized)
					if (pattern_map.count(src_label) == 0 || pattern_map[src_label].count(elabel) == 0 || pattern_map[src_label][elabel].count(dst_label) == 0)
						single_edge_dfscodes++;
					LabEdge *eptr = &(edge_list[*e]);
					pattern_map[src_label][elabel][dst_label].push(2, eptr, 0); // single-edge embedding: (num_vertices, edge, pointer_to_parent_embedding)
					num_embeddings ++;
				}
			}
		}
		int dfscodes_per_thread = (int) ceil((single_edge_dfscodes * 1.0) / numThreads);
		std::cout << "num_single_edge_patterns = " << single_edge_dfscodes << "\n";
		if (show) std::cout << "dfscodes_per_thread = " << dfscodes_per_thread << std::endl; 
		if (show) std::cout << "num_embeddings = " << num_embeddings << std::endl; 
		// for each single-edge pattern, generate a DFS code and push it into the task queue
		for(EmbeddingList_iterator3 fromlabel = pattern_map.begin(); fromlabel != pattern_map.end(); ++fromlabel) {
			for(EmbeddingList_iterator2 elabel = fromlabel->second.begin(); elabel != fromlabel->second.end(); ++elabel) {
				for(EmbeddingList_iterator1 tolabel = elabel->second.begin(); tolabel != elabel->second.end(); ++tolabel) {
					DFS dfs(0, 1, fromlabel->first, elabel->first, tolabel->first);
					task_queue.push_back(dfs);
				} // for tolabel
			} // for elabel
		} // for fromlabel
	}

	void process() {
		init_dfscode(); // insert single-edge patterns into the queue
		galois::do_all(galois::iterate(task_queue),
			[&](const DFS& dfs) {
				LocalStatus *ls = status.getLocal();
				ls->current_dfs_level = 0;
				std::deque<DFS> tmp;
				ls->dfs_task_queue.push_back(tmp);
				ls->dfs_task_queue[0].push_back(dfs);
				ls->DFS_CODE.push(0, 1, dfs.fromlabel, dfs.elabel, dfs.tolabel);
				grow(pattern_map[dfs.fromlabel][dfs.elabel][dfs.tolabel], 1, *ls);
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
			galois::loopname("FSM-DFS")
		);
	}
	void print_output() {
		size_t total = 0;
		for(int i = 0; i < numThreads; i++)
			total += status.getLocal(i)->frequent_patterns_count;
		std::cout << "\n\tnum_frequent_patterns (minsup=" << minsup << "): " << total << "\n\n";
	}
#ifdef ENABLE_LB
	void init_lb() {
		//this->num_threads = num_threads;
		next_work_request.clear();
		message_queue.clear();
		requested_work_from.clear();
		for(int i = 0; i < num_threads; i++) {
			int p =  (i + 1) % num_threads;
			next_work_request.push_back(p);
			std::deque<int> dq;
			message_queue.push_back(dq);
			std::vector<int> vc;
			requested_work_from.push_back(vc);
		}
		all_idle_work_request_counter = 0;
	}
	void process_work_split_request(int source, LocalStatus &status) {
		int thread_id = status.thread_id;
		if(thread_id == 0 && all_threads_idle()) {
			all_idle_work_request_counter++;
			return;
		}
		if(thread_working(status) == false || can_thread_split_work(status) == false) {
			int buffer[2];
			buffer[0] = RT_WORK_RESPONSE;
			buffer[1] = 0;
			send_msg(buffer, 2, thread_id, source);
			return;
		}
		int length;
		thread_split_work(source, length, status);
		int buffer_size[2];
		buffer_size[0] = RT_WORK_RESPONSE;
		buffer_size[1] = length; // put there length of the split stack split.size()+1;
		send_msg(buffer_size, 2, thread_id, source);
	}
	bool receive_data(int source, int size, LocalStatus &status) {
		int thread_id = status.thread_id;
		if(size == 0) {
			next_work_request[thread_id] = random() % num_threads;
			while(next_work_request[thread_id] == thread_id)
				next_work_request[thread_id] = random() % num_threads;
			requested_work_from[thread_id].erase(requested_work_from[thread_id].begin());
			return false;
		}
		if(requested_work_from[thread_id].size() != 1 || requested_work_from[thread_id][0] != source )
			exit(1);
		next_work_request[thread_id] = random() % num_threads;
		while(next_work_request[thread_id] == thread_id)
			next_work_request[thread_id] = random() % num_threads;
		requested_work_from[thread_id].erase(requested_work_from[thread_id].begin());
		thread_start_working(status);
		return true;
	}
	void recv_msg(int *buffer, int length, int thr, int originating_thr) {
		simple_lock.lock();
		int source = message_queue[thr].front();
		if(originating_thr != source) {
			exit(0);
		}
		message_queue[thr].pop_front(); //take off source
		for(int i = 0; i < length; i++) {
			buffer[i] = message_queue[thr].front();
			message_queue[thr].pop_front();
		}
		simple_lock.unlock();
	}
	void send_msg(int *buffer, int length, int src_thr, int dest_thr) {
		simple_lock.lock();
		message_queue[dest_thr].push_back(src_thr);
		for(int i = 0; i <length; i++)
			message_queue[dest_thr].push_back(buffer[i]);
		simple_lock.unlock();
	}
	bool can_thread_split_work(LocalStatus &status) {
		if(!thread_working(status)) return false;
		status.task_split_level = 0; // start search from level 0 task queue
		while(status.task_split_level < status.current_dfs_level && status.dfs_task_queue[status.task_split_level].size() < task_split_threshold )
			status.task_split_level++;
		if(status.dfs_task_queue.size() > status.task_split_level && status.dfs_task_queue[status.task_split_level].size() >= task_split_threshold )
			return true;
		return false;
	}
	void thread_split_work(int requesting_thread, int &length, LocalStatus &status) {
		for(int i = 0; i < status.task_split_level; i++) {
			if(dfs_task_queue_shared[requesting_thread].size() < (i + 1) ) {
				std::deque<DFS> tmp;
				dfs_task_queue_shared[requesting_thread].push_back(tmp);
			}
			dfs_task_queue_shared[requesting_thread][i].push_back(status.DFS_CODE[i]);
		}
		if(dfs_task_queue_shared[requesting_thread].size() < ( status.task_split_level + 1) ) {
			std::deque<DFS> tmp;
			dfs_task_queue_shared[requesting_thread].push_back(tmp);
		}
		int num_dfs = status.dfs_task_queue[status.task_split_level].size() / 2;
		for(int j = 0; j < num_dfs; j++) {
			DFS dfs = status.dfs_task_queue[status.task_split_level].back();
			dfs_task_queue_shared[requesting_thread][status.task_split_level].push_front(dfs);
			status.dfs_task_queue[status.task_split_level].pop_back();
		}
		embeddings_regeneration_level[requesting_thread] = status.task_split_level;
		length = num_dfs;
	}
	void thread_process_received_data(LocalStatus &status) {
		int thread_id = status.thread_id;
		status.embeddings_regeneration_level = embeddings_regeneration_level[thread_id];
		int num_dfs = dfs_task_queue_shared[thread_id][status.embeddings_regeneration_level].size();
		for(int i = 0; i < status.embeddings_regeneration_level; i++) {
			if(status.dfs_task_queue.size() < (i + 1) ) {
				std::deque<DFS> tmp;
				status.dfs_task_queue.push_back(tmp);
			}
			DFS dfs = dfs_task_queue_shared[thread_id][i].back();
			status.dfs_task_queue[i].push_back(dfs);
			dfs_task_queue_shared[thread_id][i].pop_back();
		}
		if(status.dfs_task_queue.size() < ( status.embeddings_regeneration_level + 1) ) {
			std::deque<DFS> tmp;
			status.dfs_task_queue.push_back(tmp);
		}
		for(int j = 0; j < num_dfs; j++) {
			DFS dfs = dfs_task_queue_shared[thread_id][status.embeddings_regeneration_level].back();
			status.dfs_task_queue[status.embeddings_regeneration_level].push_front(dfs);
			dfs_task_queue_shared[thread_id][status.embeddings_regeneration_level].pop_back();
		}
	}
	void thread_start_working(LocalStatus &status) {
		int thread_id = status.thread_id;
		activate_thread(thread_id);
	}
	void process_request(int source, LocalStatus &status) {
		int thread_id = status.thread_id;
		int recv_buf[2];
		recv_msg(recv_buf, 2, thread_id, source);
		switch(recv_buf[0]) {
			case WORK_REQUEST:
				process_work_split_request(source, status); break;
			case WORK_RESPONSE:
				receive_data(source, recv_buf[1], status); return;
			default: exit(1);
		}
	}
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
	void threads_load_balance(LocalStatus &status) {
		int thread_id = status.thread_id;
		int src = check_request(status);
		if(src != -1) process_request(src, status);
		if(thread_working(status) == false) {
			if(all_threads_idle()) {
				printf("All threads idle!\n");
			} else send_work_request(status); //global load balance
		}
	}
	int check_request(LocalStatus &status) {
		int thread_id = status.thread_id;
		if(message_queue[thread_id].size() > 0 ) {
			simple_lock.lock();
			int source = message_queue[thread_id].front();
			simple_lock.unlock();
			return source;
		}
		return -1;
	}
	void send_work_request(LocalStatus &status) {
		int thread_id = status.thread_id;
		if(!requested_work_from[thread_id].empty()) return;
		int buffer[2];
		buffer[0] = WORK_REQUEST;
		buffer[1] = 0;       // filler
		send_msg(buffer, 2, thread_id, next_work_request[thread_id]);
		requested_work_from[thread_id].push_back(next_work_request[thread_id]);
	}
#endif
	// edge extension by DFS traversal: recursive call
	void grow(EmbeddingList &emb_list, unsigned dfs_level, LocalStatus &status) {
		unsigned sup = support(emb_list, status);
		if (sup < minimal_support) return;
		if (is_min(status) == false) return; // check if this pattern is canonical: minimal DFSCode
		status.frequent_patterns_count ++;
		// list frequent patterns here!!!
		if(show_output) {
			std::cout << status.DFS_CODE.to_string(false) << ": " << sup << std::endl;
			for (auto it = emb_list.begin(); it != emb_list.end(); it++) std::cout << "\t" << it->to_string_all() << std::endl;
		}
		if (dfs_level == max_level) return;
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
					//if (emb_size + 1 <= max_size)
						new_fwd_root[maxtoc][(*it)->elabel][graph->getData((*it)->to)].push(emb_size+1, *it, cur);
				}
			}
			// backtracked forward
			for(size_t i = 0; i < rmpath.size(); ++i) {
				if(get_forward_rmpath(*graph, edge_list, history[rmpath[i]], minlabel, history, edges)) {
					for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
						//if (emb_size + 1 <= max_size)
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
							//if (emb_size + 1 <= max_size)
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
	Status status;
	unsigned minimal_support;
	unsigned max_level;
	bool show_output;
	PatternMap3D pattern_map; // mapping patterns to their embedding list
	PatternQueue task_queue; // task queue holding the DFScodes of patterns
	std::vector<int> frequent_patterns_count;
	std::vector<std::vector<std::deque<DFS> > > dfs_task_queue;       //keep the sibling extensions for each level and for each thread
#ifdef ENABLE_LB
	typedef enum {WORK_REQUEST = 0, WORK_RESPONSE = 1} REQUEST_TYPE;
	galois::substrate::SimpleLock simple_lock;
	int task_split_threshold;
	int all_idle_work_request_counter;
	std::vector<int> next_work_request;
	std::vector<bool> thread_is_working;
	std::vector<std::deque<int> > message_queue;
	std::vector<std::vector<int> > requested_work_from;
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
				for(EdgeList::iterator it = edges.begin(); it != edges.end(); ++it) {
					root[status.GRAPH_IS_MIN[from].label][(*it)->elabel][status.GRAPH_IS_MIN[(*it)->to].label].push(0, *it, 0);
				}
			}
		}
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

