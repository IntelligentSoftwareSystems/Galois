#ifndef BLISS_GRAPH_HH
#define BLISS_GRAPH_HH

#include "abgraph.hh"

namespace bliss {

#ifdef USE_DOMAIN
typedef std::pair<unsigned, Index> IndexEdge;
#else
typedef unsigned IndexEdge;
#endif

#if defined(BLISS_CONSISTENCY_CHECKS)
static bool is_permutation(const unsigned int N, const unsigned int* perm) {
	if(N == 0) return true;
	std::vector<bool> m(N, false);
	for(unsigned int i = 0; i < N; i++) {
		if(perm[i] >= N) return false;
		if(m[perm[i]]) return false;
		m[perm[i]] = true;
	}
	return true;
}
#endif
static bool is_permutation(const std::vector<unsigned int>& perm) {
	const unsigned int N = perm.size();
	if(N == 0)
		return true;
	std::vector<bool> m(N, false);
	for(unsigned int i = 0; i < N; i++) {
		if(perm[i] >= N) return false;
		if(m[perm[i]]) return false;
		m[perm[i]] = true;
	}
	return true;
}

// \brief The class for undirected, vertex colored graphs.
// Multiple edges between vertices are not allowed (i.e., are ignored).
class Graph : public AbstractGraph {
public:
	/**
	 * The possible splitting heuristics.
	 * The selected splitting heuristics affects the computed canonical
	 * labelings; therefore, if you want to compare whether two graphs
	 * are isomorphic by computing and comparing (for equality) their
	 * canonical versions, be sure to use the same splitting heuristics
	 * for both graphs.
	 */
	typedef enum {
		/** First non-unit cell.
		 * Very fast but may result in large search spaces on difficult graphs.
		 * Use for large but easy graphs. */
		shs_f = 0,
		/** First smallest non-unit cell.
		 * Fast, should usually produce smaller search spaces than shs_f. */
		shs_fs,
		/** First largest non-unit cell.
		 * Fast, should usually produce smaller search spaces than shs_f. */
		shs_fl,
		/** First maximally non-trivially connected non-unit cell.
		 * Not so fast, should usually produce smaller search spaces than shs_f,
		 * shs_fs, and shs_fl. */
		shs_fm,
		/** First smallest maximally non-trivially connected non-unit cell.
		 * Not so fast, should usually produce smaller search spaces than shs_f,
		 * shs_fs, and shs_fl. */
		shs_fsm,
		/** First largest maximally non-trivially connected non-unit cell.
		 * Not so fast, should usually produce smaller search spaces than shs_f,
		 * shs_fs, and shs_fl. */
		shs_flm
	} SplittingHeuristic;

	//moved from protected scope by Zhiqiang
	class Vertex {
		public:
			Vertex() { color = 0;}
			~Vertex(){ ; }
#ifdef USE_DOMAIN
			void add_edge(const unsigned other_vertex, Index index) {
				edges.push_back(std::make_pair(other_vertex, index));
#else
			void add_edge(const unsigned other_vertex) {
				edges.push_back(other_vertex);
#endif
			}
			void remove_duplicate_edges(std::vector<bool>& tmp) {
#if defined(BLISS_CONSISTENCY_CHECKS)
				/* Pre-conditions  */
				for(unsigned int i = 0; i < tmp.size(); i++) assert(tmp[i] == false);
#endif
				for(std::vector<IndexEdge>::iterator iter = edges.begin(); iter != edges.end(); ) {
#ifdef USE_DOMAIN
					const unsigned int dest_vertex = iter->first; //cxh
#else
					const unsigned int dest_vertex = *iter;
#endif
					if(tmp[dest_vertex] == true) {
						/* A duplicate edge found! */
						iter = edges.erase(iter);
					} else {
						/* Not seen earlier, mark as seen */
						tmp[dest_vertex] = true;
						iter++;
					}
				}
				/* Clear tmp */
				for(std::vector<IndexEdge>::iterator iter = edges.begin(); iter != edges.end(); iter++) {
#ifdef USE_DOMAIN
					tmp[iter->first] = false;// cxh
#else
					tmp[*iter] = false;
#endif
				}
#if defined(BLISS_CONSISTENCY_CHECKS)
				/* Post-conditions  */
				for(unsigned int i = 0; i < tmp.size(); i++) assert(tmp[i] == false);
#endif
			}
			void sort_edges() { std::sort(edges.begin(), edges.end()); }
			unsigned color;
			//std::vector<unsigned> edges;
			std::vector<IndexEdge> edges; // cxh: add the edge ids from the embedding
			unsigned nof_edges() const {return edges.size(); }
			};
			//added by Zhiqiang
			std::vector<Vertex> & get_vertices_rstream() { return vertices; }
			void sort_edges_rstream() { sort_edges(); }

protected:
	std::vector<Vertex> vertices;
	void sort_edges() {
		for(unsigned int i = 0; i < get_nof_vertices(); i++)
			vertices[i].sort_edges();
	}
	void remove_duplicate_edges() {
		std::vector<bool> tmp(vertices.size(), false);
		for(std::vector<Vertex>::iterator vi = vertices.begin();
				vi != vertices.end();
				vi++)
		{
#if defined(BLISS_EXPENSIVE_CONSISTENCY_CHECKS)
			for(unsigned int i = 0; i < tmp.size(); i++) assert(tmp[i] == false);
#endif
			(*vi).remove_duplicate_edges(tmp);
		}
	}
	// \internal Partition independent invariant.
	// Return the color of the vertex. Time complexity: O(1)
	static unsigned int vertex_color_invariant(const Graph* const g, const unsigned int v) { 
		return g->vertices[v].color;
	}
	/** \internal
	 * Partition independent invariant.
	 * Returns the degree of the vertex.
	 * DUPLICATE EDGES MUST HAVE BEEN REMOVED BEFORE.
	 * Time complexity: O(1).
	 */
	// Return the degree of the vertex. Time complexity: O(1)
	static unsigned int degree_invariant(const Graph* const g, const unsigned int v) {
		return g->vertices[v].nof_edges();
	}
	/** \internal
	 * Partition independent invariant.
	 * Returns 1 if there is an edge from the vertex to itself, 0 if not.
	 * Time complexity: O(k), where k is the number of edges leaving the vertex.
	 */
	// Return 1 if the vertex v has a self-loop, 0 otherwise
	// Time complexity: O(E_v), where E_v is the number of edges leaving v
	static unsigned selfloop_invariant(const Graph* const g, const unsigned int v) {
		const Vertex& vertex = g->vertices[v];
		for(std::vector<IndexEdge>::const_iterator ei = vertex.edges.begin(); ei != vertex.edges.end(); ei++) {
#ifdef USE_DOMAIN
			if(ei->first == v) return 1; // cxh
#else
			if(*ei == v) return 1;
#endif
		}
		return 0;
	}

	// Refine the partition p according to a partition independent invariant
	bool refine_according_to_invariant(unsigned int (*inv)(const Graph* const g, const unsigned int v)) {
		bool refined = false;
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; ) {
			Partition::Cell* const next_cell = cell->next_nonsingleton;
			const unsigned int* ep = p.elements + cell->first;
			for(unsigned int i = cell->length; i > 0; i--, ep++) {
				const unsigned int ival = inv(this, *ep);
				p.invariant_values[*ep] = ival;
				if(ival > cell->max_ival) {
					cell->max_ival = ival;
					cell->max_ival_count = 1;
				}
				else if(ival == cell->max_ival) {
					cell->max_ival_count++;
				}
			}
			Partition::Cell* const last_new_cell = p.zplit_cell(cell, true);
			refined |= (last_new_cell != cell);
			cell = next_cell;
		}
		return refined;
	}
	// Routines needed when refining the partition p into equitable
	// Split the neighbourhood of a cell according to the equitable invariant
	bool split_neighbourhood_of_cell(Partition::Cell* const cell) {
		const bool was_equal_to_first = refine_equal_to_first;
		if(compute_eqref_hash) {
			eqref_hash.update(cell->first);
			eqref_hash.update(cell->length);
		}
		const unsigned int* ep = p.elements + cell->first;
		for(unsigned int i = cell->length; i > 0; i--) {
			const Vertex& v = vertices[*ep++];   
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j != 0; j--) {
#ifdef USE_DOMAIN
				const unsigned int dest_vertex = (ei++)->first; // cxh
#else
				const unsigned int dest_vertex = *ei++;
#endif
				Partition::Cell * const neighbour_cell = p.get_cell(dest_vertex);
				if(neighbour_cell->is_unit())
					continue;
				const unsigned int ival = ++p.invariant_values[dest_vertex];
				if(ival > neighbour_cell->max_ival) {
					neighbour_cell->max_ival = ival;
					neighbour_cell->max_ival_count = 1;
					if(ival == 1) {
						neighbour_heap.insert(neighbour_cell->first);
					}
				}
				else if(ival == neighbour_cell->max_ival) {
					neighbour_cell->max_ival_count++;
				}
			}
		}
		while(!neighbour_heap.is_empty()) {
			const unsigned int start = neighbour_heap.remove();
			Partition::Cell * const neighbour_cell = p.get_cell(p.elements[start]);
			if(compute_eqref_hash) {
				eqref_hash.update(neighbour_cell->first);
				eqref_hash.update(neighbour_cell->length);
				eqref_hash.update(neighbour_cell->max_ival);
				eqref_hash.update(neighbour_cell->max_ival_count);
			}
			Partition::Cell* const last_new_cell = p.zplit_cell(neighbour_cell, true);
			// Update certificate and hash if needed
			const Partition::Cell* c = neighbour_cell;
			while(1) {
				if(in_search) {
					// Build certificate
					cert_add_redundant(CERT_SPLIT, c->first, c->length);
					// No need to continue?
					if(refine_compare_certificate and
							(refine_equal_to_first == false) and
							(refine_cmp_to_best < 0))
						goto worse_exit;
				}
				if(compute_eqref_hash) {
					eqref_hash.update(c->first);
					eqref_hash.update(c->length);
				}
				if(c == last_new_cell) break;
				c = c->next;
			}
		}

		if(refine_compare_certificate and (refine_equal_to_first == false) and (refine_cmp_to_best < 0))
			return true;
		return false;
worse_exit:
		// Clear neighbour heap 
		UintSeqHash rest;
		while(!neighbour_heap.is_empty()) {
			const unsigned int start = neighbour_heap.remove();
			Partition::Cell * const neighbour_cell = p.get_cell(p.elements[start]);
			if(opt_use_failure_recording and was_equal_to_first) {
				rest.update(neighbour_cell->first);
				rest.update(neighbour_cell->length);
				rest.update(neighbour_cell->max_ival);
				rest.update(neighbour_cell->max_ival_count);
			}
			neighbour_cell->max_ival = 0;
			neighbour_cell->max_ival_count = 0;
			p.clear_ivs(neighbour_cell);
		}
		if(opt_use_failure_recording and was_equal_to_first) {
			for(unsigned int i = p.splitting_queue.size(); i > 0; i--) {
				Partition::Cell* const cell = p.splitting_queue.pop_front();
				rest.update(cell->first);
				rest.update(cell->length);
				p.splitting_queue.push_back(cell);
			}
			rest.update(failure_recording_fp_deviation);
			failure_recording_fp_deviation = rest.get_value();
		}
		return true;
	}

	bool split_neighbourhood_of_unit_cell(Partition::Cell* const unit_cell) {
		const bool was_equal_to_first = refine_equal_to_first;
		if(compute_eqref_hash) {
			eqref_hash.update(0x87654321);
			eqref_hash.update(unit_cell->first);
			eqref_hash.update(1);
		}
		const Vertex& v = vertices[p.elements[unit_cell->first]];
		std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
		for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
			const unsigned int dest_vertex = (ei++)->first; // cxh
#else
			const unsigned int dest_vertex = *ei++;
#endif
			Partition::Cell * const neighbour_cell = p.get_cell(dest_vertex);

			if(neighbour_cell->is_unit()) {
				if(in_search) {
					/* Remember neighbour in order to generate certificate */
					neighbour_heap.insert(neighbour_cell->first);
				}
				continue;
			}
			if(neighbour_cell->max_ival_count == 0) {
				neighbour_heap.insert(neighbour_cell->first);
			}
			neighbour_cell->max_ival_count++;

			unsigned int * const swap_position =
				p.elements + neighbour_cell->first + neighbour_cell->length -
				neighbour_cell->max_ival_count;
			*p.in_pos[dest_vertex] = *swap_position;
			p.in_pos[*swap_position] = p.in_pos[dest_vertex];
			*swap_position = dest_vertex;
			p.in_pos[dest_vertex] = swap_position;
		}

		while(!neighbour_heap.is_empty()) {
			const unsigned int start = neighbour_heap.remove();
			Partition::Cell* neighbour_cell =	p.get_cell(p.elements[start]);
#if defined(BLISS_CONSISTENCY_CHECKS)
			if(neighbour_cell->is_unit()) { } else { }
#endif
			if(compute_eqref_hash) {
				eqref_hash.update(neighbour_cell->first);
				eqref_hash.update(neighbour_cell->length);
				eqref_hash.update(neighbour_cell->max_ival_count);
			}

			if(neighbour_cell->length > 1 and neighbour_cell->max_ival_count != neighbour_cell->length) {
				Partition::Cell * const new_cell =
					p.aux_split_in_two(neighbour_cell, neighbour_cell->length - neighbour_cell->max_ival_count);
				unsigned int *ep = p.elements + new_cell->first;
				unsigned int * const lp = p.elements+new_cell->first+new_cell->length;
				while(ep < lp) {
					p.element_to_cell_map[*ep] = new_cell;
					ep++;
				}
				neighbour_cell->max_ival_count = 0;


				if(compute_eqref_hash) {
					/* Update hash */
					eqref_hash.update(neighbour_cell->first);
					eqref_hash.update(neighbour_cell->length);
					eqref_hash.update(0);
					eqref_hash.update(new_cell->first);
					eqref_hash.update(new_cell->length);
					eqref_hash.update(1);
				}

				/* Add cells in splitting_queue */
				if(neighbour_cell->is_in_splitting_queue()) {
					/* Both cells must be included in splitting_queue in order
					   to ensure refinement into equitable partition */
					p.splitting_queue_add(new_cell);
				} else {
					Partition::Cell *min_cell, *max_cell;
					if(neighbour_cell->length <= new_cell->length) {
						min_cell = neighbour_cell;
						max_cell = new_cell;
					} else {
						min_cell = new_cell;
						max_cell = neighbour_cell;
					}
					/* Put the smaller cell in splitting_queue */
					p.splitting_queue_add(min_cell);
					if(max_cell->is_unit()) {
						/* Put the "larger" cell also in splitting_queue */
						p.splitting_queue_add(max_cell);
					}
				}
				/* Update pointer for certificate generation */
				neighbour_cell = new_cell;
			} else {
				/* neighbour_cell->length == 1 ||
				   neighbour_cell->max_ival_count == neighbour_cell->length */
				neighbour_cell->max_ival_count = 0;
			}

			/*
			 * Build certificate if required
			 */
			if(in_search) {
				for(unsigned int i = neighbour_cell->first, j = neighbour_cell->length; j > 0; j--, i++) {
					/* Build certificate */
					cert_add(CERT_EDGE, unit_cell->first, i);
					/* No need to continue? */
					if(refine_compare_certificate and (refine_equal_to_first == false) and (refine_cmp_to_best < 0))
						goto worse_exit;
				}
			} /* if(in_search) */
		} /* while(!neighbour_heap.is_empty()) */

		if(refine_compare_certificate and
				(refine_equal_to_first == false) and
				(refine_cmp_to_best < 0))
			return true;
		return false;

worse_exit:
		/* Clear neighbour heap */
		UintSeqHash rest;
		while(!neighbour_heap.is_empty()) {
			const unsigned int start = neighbour_heap.remove();
			Partition::Cell * const neighbour_cell = p.get_cell(p.elements[start]);
			if(opt_use_failure_recording and was_equal_to_first) {
				rest.update(neighbour_cell->first);
				rest.update(neighbour_cell->length);
				rest.update(neighbour_cell->max_ival_count);
			}
			neighbour_cell->max_ival_count = 0;
		}
		if(opt_use_failure_recording and was_equal_to_first) {
			rest.update(failure_recording_fp_deviation);
			failure_recording_fp_deviation = rest.get_value();
		}
		return true;
	}

	//  Build the initial equitable partition
	void make_initial_equitable_partition() {
		refine_according_to_invariant(&vertex_color_invariant);
		p.splitting_queue_clear();
		//p.print_signature(stderr); fprintf(stderr, "\n");
		refine_according_to_invariant(&selfloop_invariant);
		p.splitting_queue_clear();
		//p.print_signature(stderr); fprintf(stderr, "\n");
		refine_according_to_invariant(&degree_invariant);
		p.splitting_queue_clear();
		//p.print_signature(stderr); fprintf(stderr, "\n");
		refine_to_equitable();
		//p.print_signature(stderr); fprintf(stderr, "\n");
	}
	// \internal
	// \copydoc AbstractGraph::is_equitable() const
	//Check whether the current partition p is equitable.
	//Performance: very slow, use only for debugging purposes.
	bool is_equitable() const {
		const unsigned int N = get_nof_vertices();
		if(N == 0) return true;
		std::vector<unsigned int> first_count = std::vector<unsigned int>(N, 0);
		std::vector<unsigned int> other_count = std::vector<unsigned int>(N, 0);
		for(Partition::Cell *cell = p.first_cell; cell; cell = cell->next) {
			if(cell->is_unit()) continue;
			unsigned int *ep = p.elements + cell->first;
			const Vertex &first_vertex = vertices[*ep++];
			/* Count how many edges lead from the first vertex to
			 * the neighbouring cells */
			for(std::vector<IndexEdge>::const_iterator ei = first_vertex.edges.begin(); ei != first_vertex.edges.end(); ei++) {
#ifdef USE_DOMAIN
				first_count[p.get_cell(ei->first)->first]++; // cxh
#else
				first_count[p.get_cell(*ei)->first]++;
#endif
			}
			/* Count and compare to the edges of the other vertices */
			for(unsigned int i = cell->length; i > 1; i--) {
				const Vertex &vertex = vertices[*ep++];
				for(std::vector<IndexEdge>::const_iterator ei = vertex.edges.begin(); ei != vertex.edges.end(); ei++) {
#ifdef USE_DOMAIN
					other_count[p.get_cell(ei->first)->first]++; // cxh
#else
					other_count[p.get_cell(*ei)->first]++;
#endif
				}
				for(Partition::Cell *cell2 = p.first_cell; cell2; cell2 = cell2->next) {
					if(first_count[cell2->first] != other_count[cell2->first]) {
						/* Not equitable */
						return false;
					}
					other_count[cell2->first] = 0;
				}
			}
			/* Reset first_count */
			for(unsigned int i = 0; i < N; i++) first_count[i] = 0;
		}
		return true;
	}
	/* Splitting heuristics, documented in more detail in graph.cc */
	SplittingHeuristic sh;

	// Find the next cell to be splitted
	Partition::Cell* find_next_cell_to_be_splitted(Partition::Cell* cell) {
		switch(sh) {
			case shs_f:   return sh_first();
			case shs_fs:  return sh_first_smallest();
			case shs_fl:  return sh_first_largest();
			case shs_fm:  return sh_first_max_neighbours();
			case shs_fsm: return sh_first_smallest_max_neighbours();
			case shs_flm: return sh_first_largest_max_neighbours();
			default:      fatal_error("Internal error - unknown splitting heuristics");
						  return 0;
		}
	}
	// \internal
	// A splitting heuristic.
	// Returns the first nonsingleton cell in the current partition.
	Partition::Cell* sh_first() {
		Partition::Cell* best_cell = 0;
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level)
				continue;
			best_cell = cell;
			break;
		}
		return best_cell;
	}
	// \internal A splitting heuristic.
	// Returns the first smallest nonsingleton cell in the current partition.
	Partition::Cell* sh_first_smallest() {
		Partition::Cell* best_cell = 0;
		unsigned int best_size = UINT_MAX;
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level) continue;
			if(cell->length < best_size) {
				best_size = cell->length;
				best_cell = cell;
			}
		}
		return best_cell;
	}
	// \internal A splitting heuristic.
	// Returns the first largest nonsingleton cell in the current partition.
	Partition::Cell* sh_first_largest() {
		Partition::Cell* best_cell = 0;
		unsigned int best_size = 0;
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level)
				continue;
			if(cell->length > best_size) {
				best_size = cell->length;
				best_cell = cell;
			}
		}
		return best_cell;
	}
	// \internal
	// A splitting heuristic.
	// Returns the first nonsingleton cell with max number of neighbouring nonsingleton cells.
	// Assumes that the partition p is equitable.
	// Assumes that the max_ival fields of the cells are all 0.
	Partition::Cell* sh_first_max_neighbours() {
		Partition::Cell* best_cell = 0;
		int best_value = -1;
		KStack<Partition::Cell*> neighbour_cells_visited;
		neighbour_cells_visited.init(get_nof_vertices());
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level)
				continue;
			const Vertex& v = vertices[p.elements[cell->first]];
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
				Partition::Cell * const neighbour_cell = p.get_cell((ei++)->first); // cxh
#else
				Partition::Cell * const neighbour_cell = p.get_cell(*ei++);
#endif
				if(neighbour_cell->is_unit()) continue;
				neighbour_cell->max_ival++;
				if(neighbour_cell->max_ival == 1)
					neighbour_cells_visited.push(neighbour_cell);
			}
			int value = 0;
			while(!neighbour_cells_visited.is_empty()) {
				Partition::Cell* const neighbour_cell = neighbour_cells_visited.pop();
				if(neighbour_cell->max_ival != neighbour_cell->length)
					value++;
				neighbour_cell->max_ival = 0;
			}
			if(value > best_value) {
				best_value = value;
				best_cell = cell;
			}
		}
		return best_cell;
	}
	// \internal A splitting heuristic.
	// Returns the first smallest nonsingleton cell with max number of neighbouring nonsingleton cells.
	// Assumes that the partition p is equitable. Assumes that the max_ival fields of the cells are all 0.
	Partition::Cell* sh_first_smallest_max_neighbours() {
		Partition::Cell* best_cell = 0;
		int best_value = -1;
		unsigned int best_size = UINT_MAX;
		KStack<Partition::Cell*> neighbour_cells_visited;
		neighbour_cells_visited.init(get_nof_vertices());
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level)
				continue;
			const Vertex& v = vertices[p.elements[cell->first]];
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
				Partition::Cell* const neighbour_cell = p.get_cell((ei++)->first); // cxh
#else
				Partition::Cell* const neighbour_cell = p.get_cell(*ei++);
#endif
				if(neighbour_cell->is_unit()) continue;
				neighbour_cell->max_ival++;
				if(neighbour_cell->max_ival == 1)
					neighbour_cells_visited.push(neighbour_cell);
			}
			int value = 0;
			while(!neighbour_cells_visited.is_empty()) {
				Partition::Cell* const neighbour_cell = neighbour_cells_visited.pop();
				if(neighbour_cell->max_ival != neighbour_cell->length)
					value++;
				neighbour_cell->max_ival = 0;
			}
			if((value > best_value) or (value == best_value and cell->length < best_size)) {
				best_value = value;
				best_size = cell->length;
				best_cell = cell;
			}
		}
		return best_cell;
	}
	// \internal A splitting heuristic.
	// Returns the first largest nonsingleton cell with max number of neighbouring nonsingleton cells.
	// Assumes that the partition p is equitable. Assumes that the max_ival fields of the cells are all 0.
	Partition::Cell* sh_first_largest_max_neighbours() {
		Partition::Cell* best_cell = 0;
		int best_value = -1;
		unsigned int best_size = 0;
		KStack<Partition::Cell*> neighbour_cells_visited;
		neighbour_cells_visited.init(get_nof_vertices());
		for(Partition::Cell* cell = p.first_nonsingleton_cell; cell; cell = cell->next_nonsingleton) {
			if(opt_use_comprec and p.cr_get_level(cell->first) != cr_level) continue;
			const Vertex& v = vertices[p.elements[cell->first]];
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
				Partition::Cell* const neighbour_cell = p.get_cell((ei++)->first); // cxh
#else
				Partition::Cell* const neighbour_cell = p.get_cell(*ei++);
#endif
				if(neighbour_cell->is_unit()) continue;
				neighbour_cell->max_ival++;
				if(neighbour_cell->max_ival == 1)
					neighbour_cells_visited.push(neighbour_cell);
			}
			int value = 0;
			while(!neighbour_cells_visited.is_empty()) {
				Partition::Cell* const neighbour_cell = neighbour_cells_visited.pop();
				if(neighbour_cell->max_ival != neighbour_cell->length) value++;
				neighbour_cell->max_ival = 0;
			}
			if((value > best_value) or (value == best_value and cell->length > best_size)) {
				best_value = value;
				best_size = cell->length;
				best_cell = cell;
			}
		}
		return best_cell;
	}
	//Initialize the certificate size and memory
	void initialize_certificate() {
		certificate_index = 0;
		certificate_current_path.clear();
		certificate_first_path.clear();
		certificate_best_path.clear();
	}
	bool is_automorphism(unsigned* const perm) {
		std::set<unsigned int, std::less<unsigned int> > edges1;
		std::set<unsigned int, std::less<unsigned int> > edges2;

#if defined(BLISS_CONSISTENCY_CHECKS)
		if(!is_permutation(get_nof_vertices(), perm))
			_INTERNAL_ERROR();
#endif

		for(unsigned int i = 0; i < get_nof_vertices(); i++) {
			Vertex& v1 = vertices[i];
			edges1.clear();
			for(std::vector<IndexEdge>::iterator ei = v1.edges.begin(); ei != v1.edges.end(); ei++)
#ifdef USE_DOMAIN
				edges1.insert(perm[ei->first]); // cxh
#else
			edges1.insert(perm[*ei]);
#endif
			Vertex& v2 = vertices[perm[i]];
			edges2.clear();
			for(std::vector<IndexEdge>::iterator ei = v2.edges.begin(); ei != v2.edges.end(); ei++)
#ifdef USE_DOMAIN
				edges2.insert(ei->first); // cxh
#else
			edges2.insert(*ei);
#endif
			if(!(edges1 == edges2)) return false;
		}
		return true;
	}

	bool nucr_find_first_component(const unsigned level) {
		cr_component.clear();
		cr_component_elements = 0;
		/* Find first non-discrete cell in the component level */
		Partition::Cell* first_cell = p.first_nonsingleton_cell;
		while(first_cell) {
			if(p.cr_get_level(first_cell->first) == level) break;
			first_cell = first_cell->next_nonsingleton;
		}
		/* The component is discrete, return false */
		if(!first_cell) return false;
		std::vector<Partition::Cell*> component;
		first_cell->max_ival = 1;
		component.push_back(first_cell);
		for(unsigned int i = 0; i < component.size(); i++) {
			Partition::Cell* const cell = component[i];
			const Vertex& v = vertices[p.elements[cell->first]];
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
				const unsigned int neighbour = (ei++)->first; // cxh
#else
				const unsigned int neighbour = *ei++;
#endif 
				Partition::Cell* const neighbour_cell = p.get_cell(neighbour);
				/* Skip unit neighbours */
				if(neighbour_cell->is_unit()) continue;
				/* Already marked to be in the same component? */
				if(neighbour_cell->max_ival == 1) continue;
				/* Is the neighbour at the same component recursion level? */
				if(p.cr_get_level(neighbour_cell->first) != level) continue;
				if(neighbour_cell->max_ival_count == 0)
					neighbour_heap.insert(neighbour_cell->first);
				neighbour_cell->max_ival_count++;
			}
			while(!neighbour_heap.is_empty()) {
				const unsigned int start = neighbour_heap.remove();
				Partition::Cell* const neighbour_cell =
					p.get_cell(p.elements[start]);
				/* Skip saturated neighbour cells */
				if(neighbour_cell->max_ival_count == neighbour_cell->length) {
					neighbour_cell->max_ival_count = 0;
					continue;
				} 
				neighbour_cell->max_ival_count = 0;
				neighbour_cell->max_ival = 1;
				component.push_back(neighbour_cell);
			}
		}
		for(unsigned int i = 0; i < component.size(); i++) {
			Partition::Cell* const cell = component[i];
			cell->max_ival = 0;
			cr_component.push_back(cell->first);
			cr_component_elements += cell->length;
		}
		if(verbstr and verbose_level > 2) {
			fprintf(verbstr, "NU-component with %lu cells and %u vertices\n",
					(long unsigned)cr_component.size(), cr_component_elements);
			fflush(verbstr);
		}
		return true;
	}
	bool nucr_find_first_component(const unsigned int level, std::vector<unsigned int>& component, unsigned int& component_elements, Partition::Cell*& sh_return) {
		component.clear();
		component_elements = 0;
		sh_return = 0;
		unsigned int sh_first  = 0;
		unsigned int sh_size   = 0;
		unsigned int sh_nuconn = 0;

		/* Find first non-discrete cell in the component level */
		Partition::Cell* first_cell = p.first_nonsingleton_cell;
		while(first_cell) {
			if(p.cr_get_level(first_cell->first) == level) break;
			first_cell = first_cell->next_nonsingleton;
		}
		if(!first_cell) {
			/* The component is discrete, return false */
			return false;
		}
		std::vector<Partition::Cell*> comp;
		KStack<Partition::Cell*> neighbours;
		neighbours.init(get_nof_vertices());
		first_cell->max_ival = 1;
		comp.push_back(first_cell);
		for(unsigned int i = 0; i < comp.size(); i++) {
			Partition::Cell* const cell = comp[i];
			const Vertex& v = vertices[p.elements[cell->first]];
			std::vector<IndexEdge>::const_iterator ei = v.edges.begin();
			for(unsigned int j = v.nof_edges(); j > 0; j--) {
#ifdef USE_DOMAIN
				const unsigned int neighbour = (ei++)->first; // cxh
#else
				const unsigned int neighbour = *ei++;
#endif
				Partition::Cell* const neighbour_cell = p.get_cell(neighbour);
				/* Skip unit neighbours */
				if(neighbour_cell->is_unit()) continue;
				/* Is the neighbour at the same component recursion level? */
				//if(p.cr_get_level(neighbour_cell->first) != level)
				//  continue;
				if(neighbour_cell->max_ival_count == 0)
					neighbours.push(neighbour_cell);
				neighbour_cell->max_ival_count++;
			}
			unsigned int nuconn = 1;
			while(!neighbours.is_empty()) {
				Partition::Cell* const neighbour_cell = neighbours.pop();
				//neighbours.pop_back();
				/* Skip saturated neighbour cells */
				if(neighbour_cell->max_ival_count == neighbour_cell->length) {
					neighbour_cell->max_ival_count = 0;
					continue;
				}
				nuconn++;
				neighbour_cell->max_ival_count = 0;
				if(neighbour_cell->max_ival == 0) {
					comp.push_back(neighbour_cell);
					neighbour_cell->max_ival = 1;
				}
			}
			switch(sh) {
				case shs_f:
					if(sh_return == 0 or cell->first <= sh_first) {
						sh_return = cell;
						sh_first = cell->first;
					}
					break;
				case shs_fs:
					if(sh_return == 0 or cell->length < sh_size or
							(cell->length == sh_size and cell->first <= sh_first)) {
						sh_return = cell;
						sh_first = cell->first;
						sh_size = cell->length;
					}
					break;
				case shs_fl:
					if(sh_return == 0 or cell->length > sh_size or
							(cell->length == sh_size and cell->first <= sh_first)) {
						sh_return = cell;
						sh_first = cell->first;
						sh_size = cell->length;
					}
					break;
				case shs_fm:
					if(sh_return == 0 or nuconn > sh_nuconn or
							(nuconn == sh_nuconn and cell->first <= sh_first)) {
						sh_return = cell;
						sh_first = cell->first;
						sh_nuconn = nuconn;
					}
					break;
				case shs_fsm:
					if(sh_return == 0 or
							nuconn > sh_nuconn or
							(nuconn == sh_nuconn and
							 (cell->length < sh_size or
							  (cell->length == sh_size and cell->first <= sh_first)))) {
						sh_return = cell;
						sh_first = cell->first;
						sh_size = cell->length;
						sh_nuconn = nuconn;
					}
					break;
				case shs_flm:
					if(sh_return == 0 or
							nuconn > sh_nuconn or
							(nuconn == sh_nuconn and
							 (cell->length > sh_size or
							  (cell->length == sh_size and cell->first <= sh_first)))) {
						sh_return = cell;
						sh_first = cell->first;
						sh_size = cell->length;
						sh_nuconn = nuconn;
					}
					break;
				default:
					fatal_error("Internal error - unknown splitting heuristics");
					return 0;
			}
		}
		assert(sh_return);
		for(unsigned int i = 0; i < comp.size(); i++) {
			Partition::Cell* const cell = comp[i];
			cell->max_ival = 0;
			component.push_back(cell->first);
			component_elements += cell->length;
		}
		if(verbstr and verbose_level > 2) {
			fprintf(verbstr, "NU-component with %lu cells and %u vertices\n",
					(long unsigned)component.size(), component_elements);
			fflush(verbstr);
		}
		return true;
	}

public:
	// Create a new graph with \a N vertices and no edges.
	Graph(const unsigned nof_vertices = 0) {
		vertices.resize(nof_vertices);
		sh = shs_flm;
	}

	/**
	 * Destroy the graph.
	 */
	~Graph() { ; }

	/**
	 * Read the graph from the file \a fp in a variant of the DIMACS format.
	 * See the <A href="http://www.tcs.hut.fi/Software/bliss/">bliss website</A>
	 * for the definition of the file format.
	 * Note that in the DIMACS file the vertices are numbered from 1 to N while
	 * in this C++ API they are from 0 to N-1.
	 * Thus the vertex n in the file corresponds to the vertex n-1 in the API.
	 *
	 * \param fp      the file stream for the graph file
	 * \param errstr  if non-null, the possible error messages are printed
	 *                in this file stream
	 * \return        a new Graph object or 0 if reading failed for some
	 *                reason
	 */
	static Graph* read_dimacs(FILE* const fp, FILE* const errstr = stderr) { return NULL; }

	/**
	 * Write the graph to a file in a variant of the DIMACS format.
	 * See the <A href="http://www.tcs.hut.fi/Software/bliss/">bliss website</A>
	 * for the definition of the file format.
	 */
	void write_dimacs(FILE* const fp) {}

	// \copydoc AbstractGraph::write_dot(FILE * const fp)
	void write_dot(FILE* const fp) {}

	// \copydoc AbstractGraph::write_dot(const char * const file_name)
	void write_dot(const char* const file_name) {}

	// \copydoc AbstractGraph::is_automorphism(const std::vector<unsigned int>& perm) const
	bool is_automorphism(const std::vector<unsigned>& perm) const {
		if(!(perm.size() == get_nof_vertices() and is_permutation(perm)))
			return false;
		std::set<unsigned, std::less<unsigned> > edges1;
		std::set<unsigned, std::less<unsigned> > edges2;
		for(unsigned i = 0; i < get_nof_vertices(); i++) {
			const Vertex& v1 = vertices[i];
			edges1.clear();
			for(std::vector<IndexEdge>::const_iterator ei = v1.edges.begin(); ei != v1.edges.end(); ei++)
#ifdef USE_DOMAIN
				edges1.insert(perm[ei->first]); // cxh
#else
			edges1.insert(perm[*ei]);
#endif
			const Vertex& v2 = vertices[perm[i]];
			edges2.clear();
			for(std::vector<IndexEdge>::const_iterator ei = v2.edges.begin(); ei != v2.edges.end(); ei++)
#ifdef USE_DOMAIN
				edges2.insert(ei->first); // cxh
#else
			edges2.insert(*ei);
#endif
			if(!(edges1 == edges2)) return false;
		}
		return true;
	}
	// \copydoc AbstractGraph::get_hash()
	virtual unsigned get_hash() {
		remove_duplicate_edges();
		sort_edges();
		UintSeqHash h;
		h.update(get_nof_vertices());
		/* Hash the color of each vertex */
		for(unsigned int i = 0; i < get_nof_vertices(); i++) {
			h.update(vertices[i].color);
		}
		/* Hash the edges */
		for(unsigned int i = 0; i < get_nof_vertices(); i++) {
			Vertex &v = vertices[i];
			for(std::vector<IndexEdge>::const_iterator ei = v.edges.begin(); ei != v.edges.end(); ei++) {
#ifdef USE_DOMAIN
				const unsigned int dest_i = ei->first; // cxh
#else
				const unsigned int dest_i = *ei;
#endif
				if(dest_i < i) continue;
				h.update(i);
				h.update(dest_i);
			}
		}
		return h.get_value();
	}
	// Return the number of vertices in the graph.
	unsigned int get_nof_vertices() const {return vertices.size(); }

	// \copydoc AbstractGraph::permute(const unsigned int* const perm) const
	Graph* permute(const unsigned* perm) const {
#if defined(BLISS_CONSISTENCY_CHECKS)
		if(!is_permutation(get_nof_vertices(), perm))
			_INTERNAL_ERROR();
#endif
		Graph* const g = new Graph(get_nof_vertices());
		for(unsigned i = 0; i < get_nof_vertices(); i++) {
			const Vertex& v = vertices[i];
			Vertex& permuted_v = g->vertices[perm[i]];
			permuted_v.color = v.color;
			for(std::vector<IndexEdge>::const_iterator ei = v.edges.begin(); ei != v.edges.end(); ei++) {
#ifdef USE_DOMAIN
				const unsigned dest_v = ei->first; //cxh
				permuted_v.add_edge(perm[dest_v], ei->second);
#else
				const unsigned dest_v = *ei;
				permuted_v.add_edge(perm[dest_v]);
#endif
			}
			permuted_v.sort_edges();
		}
		return g;
	}
	Graph* permute(const std::vector<unsigned>& perm) const {
#if defined(BLISS_CONSISTENCY_CHECKS)
#endif
		Graph* const g = new Graph(get_nof_vertices());
		for(unsigned int i = 0; i < get_nof_vertices(); i++) {
			const Vertex& v = vertices[i];
			Vertex& permuted_v = g->vertices[perm[i]];
			permuted_v.color = v.color;
			for(std::vector<IndexEdge>::const_iterator ei = v.edges.begin(); ei != v.edges.end(); ei++) {
#ifdef USE_DOMAIN
				const unsigned dest_v = ei->first; // cxh
				permuted_v.add_edge(perm[dest_v], ei->second);
#else
				const unsigned dest_v = *ei;
				permuted_v.add_edge(perm[dest_v]);
#endif
			}
			permuted_v.sort_edges();
		}
		return g;
	}
	// Add a new vertex with color \a color in the graph and return its index.
	unsigned add_vertex(const unsigned color = 0) {
		const unsigned int vertex_num = vertices.size();
		vertices.resize(vertex_num + 1);
		vertices.back().color = color;
		return vertex_num;
	}
	/**
	 * Add an edge between vertices \a v1 and \a v2.
	 * Duplicate edges between vertices are ignored but try to avoid introducing
	 * them in the first place as they are not ignored immediately but will
	 * consume memory and computation resources for a while.
	 */
	void add_edge(const unsigned vertex1, const unsigned vertex2, Index index) {
		//printf("Adding edge (%u -> %u)\n", vertex1, vertex2);
#ifdef USE_DOMAIN
		vertices[vertex1].add_edge(vertex2, index);
		vertices[vertex2].add_edge(vertex1, std::make_pair(index.second, index.first));
#else
		vertices[vertex1].add_edge(vertex2);
		vertices[vertex2].add_edge(vertex1);
#endif
	}
	// Change the color of the vertex \a vertex to \a color.
	void change_color(const unsigned vertex, const unsigned color) {
		vertices[vertex].color = color;
	}

	/**
	 * Compare this graph with the graph \a other.
	 * Returns 0 if the graphs are equal, and a negative (positive) integer
	 * if this graph is "smaller than" ("greater than", resp.) than \a other.
	 */
	int cmp(Graph& other) {
		/* Compare the numbers of vertices */
		if(get_nof_vertices() < other.get_nof_vertices())
			return -1;
		if(get_nof_vertices() > other.get_nof_vertices())
			return 1;
		/* Compare vertex colors */
		for(unsigned i = 0; i < get_nof_vertices(); i++) {
			if(vertices[i].color < other.vertices[i].color)
				return -1;
			if(vertices[i].color > other.vertices[i].color)
				return 1;
		}
		/* Compare vertex degrees */
		remove_duplicate_edges();
		other.remove_duplicate_edges();
		for(unsigned i = 0; i < get_nof_vertices(); i++) {
			if(vertices[i].nof_edges() < other.vertices[i].nof_edges())
				return -1;
			if(vertices[i].nof_edges() > other.vertices[i].nof_edges())
				return 1;
		}
		/* Compare edges */
		for(unsigned i = 0; i < get_nof_vertices(); i++) {
			Vertex &v1 = vertices[i];
			Vertex &v2 = other.vertices[i];
			v1.sort_edges();
			v2.sort_edges();
			std::vector<IndexEdge>::const_iterator ei1 = v1.edges.begin();
			std::vector<IndexEdge>::const_iterator ei2 = v2.edges.begin();
			while(ei1 != v1.edges.end()) {
#ifdef USE_DOMAIN
				if(ei1->first < ei2->first) return -1; // cxh
				if(ei1->first > ei2->first) return 1; // cxh
#else
				if(*ei1 < *ei2) return -1;
				if(*ei1 > *ei2) return 1;
#endif
				ei1++;
				ei2++;
			}
		}
		return 0;
	}
	/**
	 * Set the splitting heuristic used by the automorphism and canonical
	 * labeling algorithm.
	 * The selected splitting heuristics affects the computed canonical
	 * labelings; therefore, if you want to compare whether two graphs
	 * are isomorphic by computing and comparing (for equality) their
	 * canonical versions, be sure to use the same splitting heuristics
	 * for both graphs.
	 */
	void set_splitting_heuristic(const SplittingHeuristic shs) {sh = shs; }
};

}

#endif
