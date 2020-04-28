#ifndef BLISS_AB_GRAPH_HH
#define BLISS_AB_GRAPH_HH
#include <set>
#include <list>
#include <cstdio>
#include <vector>
#include <cassert>
#include <climits>
#include <algorithm>

namespace bliss {
	class AbstractGraph;
}

#include "kstack.hh"
#include "kqueue.hh"
#include "heap.hh"
#include "orbit.hh"
#include "partition.hh"
#include "bignum.hh"
#include "uintseqhash.hh"

namespace bliss {

void fatal_error(const char* fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr,"Bliss fatal error: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\nAborting!\n");
	va_end(ap);
	exit(1);
}

#define _INTERNAL_ERROR() fatal_error("%s:%d: internal error",__FILE__,__LINE__)
#define _OUT_OF_MEMORY() fatal_error("%s:%d: out of memory",__FILE__,__LINE__)

typedef std::pair<unsigned,unsigned> Index;

class TreeNode {
//friend class AbstractGraph;
public:
	unsigned int split_cell_first;

	int split_element;
	static const int SPLIT_START = -1;
	static const int SPLIT_END   = -2;
	Partition::BacktrackPoint partition_bt_point;
	unsigned int certificate_index;
	static const char NO = -1;
	static const char MAYBE = 0;
	static const char YES = 1;
	/* First path stuff */
	bool fp_on;
	bool fp_cert_equal;
	char fp_extendable;
	/* Best path stuff */
	bool in_best_path;
	int cmp_to_best_path;
	unsigned int failure_recording_ival;
	/* Component recursion related data */
	unsigned int cr_cep_stack_size;
	unsigned int cr_cep_index;
	unsigned int cr_level;
	bool needs_long_prune;
	unsigned int long_prune_begin;
	std::set<unsigned int, std::less<unsigned int> > long_prune_redundant;
	UintSeqHash eqref_hash;
	unsigned int subcertificate_length;
};

typedef struct {
	unsigned int splitting_element;
	unsigned int certificate_index;
	unsigned int subcertificate_length;
	UintSeqHash eqref_hash;
} PathInfo;

// \brief Statistics returned by the bliss search algorithm.
class Stats {
	friend class AbstractGraph;
	/** \internal The size of the automorphism group. */
	BigNum group_size;
	/** \internal An approximation (due to possible overflows) of
	 * the size of the automorphism group. */
	long double group_size_approx;
	/** \internal The number of nodes in the search tree. */
	long unsigned int nof_nodes;
	/** \internal The number of leaf nodes in the search tree. */
	long unsigned int nof_leaf_nodes;
	/** \internal The number of bad nodes in the search tree. */
	long unsigned int nof_bad_nodes;
	/** \internal The number of canonical representative updates. */
	long unsigned int nof_canupdates;
	/** \internal The number of generator permutations. */
	long unsigned int nof_generators;
	/** \internal The maximal depth of the search tree. */
	unsigned long int max_level;
	/** */
	void reset() {
		group_size.assign(1);
		group_size_approx = 1.0;
		nof_nodes = 0;
		nof_leaf_nodes = 0;
		nof_bad_nodes = 0;
		nof_canupdates = 0;
		nof_generators = 0;
		max_level = 0;
	}
	public:
	Stats() { reset(); }
	/** Print the statistics. */
	size_t print(FILE* const fp) const {
		size_t r = 0;
		r += fprintf(fp, "Nodes:          %lu\n", nof_nodes);
		r += fprintf(fp, "Leaf nodes:     %lu\n", nof_leaf_nodes);
		r += fprintf(fp, "Bad nodes:      %lu\n", nof_bad_nodes);
		r += fprintf(fp, "Canrep updates: %lu\n", nof_canupdates);
		r += fprintf(fp, "Generators:     %lu\n", nof_generators);
		r += fprintf(fp, "Max level:      %lu\n", max_level);
		r += fprintf(fp, "|Aut|:          ")+group_size.print(fp)+fprintf(fp, "\n");
		fflush(fp);
		return r;
	}
	/** An approximation (due to possible overflows/rounding errors) of
	 * the size of the automorphism group. */
	long double get_group_size_approx() const {return group_size_approx;}
	/** The number of nodes in the search tree. */
	long unsigned int get_nof_nodes() const {return nof_nodes;}
	/** The number of leaf nodes in the search tree. */
	long unsigned int get_nof_leaf_nodes() const {return nof_leaf_nodes;}
	/** The number of bad nodes in the search tree. */
	long unsigned int get_nof_bad_nodes() const {return nof_bad_nodes;}
	/** The number of canonical representative updates. */
	long unsigned int get_nof_canupdates() const {return nof_canupdates;}
	/** The number of generator permutations. */
	long unsigned int get_nof_generators() const {return nof_generators;}
	/** The maximal depth of the search tree. */
	unsigned long int get_max_level() const {return max_level;}
};

// \brief An abstract base class for different types of graphs.
class AbstractGraph {
	friend class Partition;
public:
	//AbstractGraph();
	// Constructor and destructor routines for the abstract graph class
	AbstractGraph() {
	// Initialize stuff
	first_path_labeling = 0;
	first_path_labeling_inv = 0;
	best_path_labeling = 0;
	best_path_labeling_inv = 0;
	first_path_automorphism = 0;
	best_path_automorphism = 0;
	in_search = false;
	// Default value for using "long prune"
	opt_use_long_prune = true;
	// Default value for using failure recording
	opt_use_failure_recording = true;
	// Default value for using component recursion
	opt_use_comprec = true;
	verbose_level = 0;
	verbstr = stdout;
	report_hook = 0;
	report_user_param = 0;
	}
	//virtual ~AbstractGraph();
	virtual ~AbstractGraph() {
	if(first_path_labeling) {
		free(first_path_labeling); first_path_labeling = 0; }
	if(first_path_labeling_inv) {
		free(first_path_labeling_inv); first_path_labeling_inv = 0; }
	if(best_path_labeling) {
		free(best_path_labeling); best_path_labeling = 0; }
	if(best_path_labeling_inv) {
		free(best_path_labeling_inv); best_path_labeling_inv = 0; }
	if(first_path_automorphism) {
		free(first_path_automorphism); first_path_automorphism = 0; }
	if(best_path_automorphism) {
		free(best_path_automorphism); best_path_automorphism = 0; }
	report_hook = 0;
	report_user_param = 0;
	}

	//Set the verbose output level for the algorithms.
	// \param level  the level of verbose output, 0 means no verbose output
	//void set_verbose_level(const unsigned int level);
void set_verbose_level(const unsigned int level) {
	verbose_level = level;
}

	/**
	 * Set the file stream for the verbose output.
	 * \param fp  the file stream; if null, no verbose output is written
	 */
	//void set_verbose_file(FILE * const fp);
void set_verbose_file(FILE* const fp) {
	verbstr = fp;
}
	/**
	 * Add a new vertex with color \a color in the graph and return its index.
	 */
	virtual unsigned int add_vertex(const unsigned int color = 0) = 0;

	/**
	 * Add an edge between vertices \a source and \a target.
	 * Duplicate edges between vertices are ignored but try to avoid introducing
	 * them in the first place as they are not ignored immediately but will
	 * consume memory and computation resources for a while.
	 */
	virtual void add_edge(const unsigned int source, const unsigned int target, Index index) = 0;

	/**
	 * Change the color of the vertex \a vertex to \a color.
	 */
	virtual void change_color(const unsigned int vertex, const unsigned int color) = 0;

	/**
	 * Check whether \a perm is an automorphism of this graph.
	 * Unoptimized, mainly for debugging purposes.
	 */
	//virtual bool is_automorphism(const std::vector<unsigned int>& perm) const;

virtual bool is_automorphism(const std::vector<unsigned int>& perm) const {
	_INTERNAL_ERROR();
	return false;
}

	/** Activate/deactivate failure recording.
	 * May not be called during the search, i.e. from an automorphism reporting
	 * hook function.
	 * \param active  if true, activate failure recording, deactivate otherwise
	 */
	void set_failure_recording(const bool active) {assert(!in_search); opt_use_failure_recording = active;}

	/** Activate/deactivate component recursion.
	 * The choice affects the computed canonical labelings;
	 * therefore, if you want to compare whether two graphs are isomorphic by
	 * computing and comparing (for equality) their canonical versions,
	 * be sure to use the same choice for both graphs.
	 * May not be called during the search, i.e. from an automorphism reporting
	 * hook function.
	 * \param active  if true, activate component recursion, deactivate otherwise
	 */
	void set_component_recursion(const bool active) {assert(!in_search); opt_use_comprec = active;}

	/**
	 * Return the number of vertices in the graph.
	 */
	virtual unsigned int get_nof_vertices() const = 0;

	/**
	 * Return a new graph that is the result of applying the permutation \a perm
	 * to this graph. This graph is not modified.
	 * \a perm must contain N=this.get_nof_vertices() elements and be a bijection
	 * on {0,1,...,N-1}, otherwise the result is undefined or a segfault.
	 */
	virtual AbstractGraph* permute(const unsigned* const perm) const = 0;
	virtual AbstractGraph* permute(const std::vector<unsigned int>& perm) const = 0;

	/**
	 * Find a set of generators for the automorphism group of the graph.
	 * The function \a hook (if non-null) is called each time a new generator
	 * for the automorphism group is found.
	 * The first argument \a user_param for the hook is the
	 * \a hook_user_param given below,
	 * the second argument \a n is the length of the automorphism (equal to
	 * get_nof_vertices()) and
	 * the third argument \a aut is the automorphism
	 * (a bijection on {0,...,get_nof_vertices()-1}).
	 * The memory for the automorphism \a aut will be invalidated immediately
	 * after the return from the hook function;
	 * if you want to use the automorphism later, you have to take a copy of it.
	 * Do not call any member functions in the hook.
	 * The search statistics are copied in \a stats.
	 */
	//void find_automorphisms(Stats& stats, void (*hook)(void* user_param, unsigned int n, const unsigned int* aut), void* hook_user_param);
void find_automorphisms(Stats& stats, void (*hook)(void *user_param, unsigned int n, const unsigned int *aut), void *user_param) {
	report_hook = hook;
	report_user_param = user_param;
	search(false, stats);
	if(first_path_labeling) {
		free(first_path_labeling);
		first_path_labeling = 0;
	}
	if(best_path_labeling) {
		free(best_path_labeling);
		best_path_labeling = 0;
	}
}
	/**
	 * Otherwise the same as find_automorphisms() except that
	 * a canonical labeling of the graph (a bijection on
	 * {0,...,get_nof_vertices()-1}) is returned.
	 * The memory allocated for the returned canonical labeling will remain
	 * valid only until the next call to a member function with the exception
	 * that constant member functions (for example, bliss::Graph::permute()) can
	 * be called without invalidating the labeling.
	 * To compute the canonical version of an undirected graph, call this
	 * function and then bliss::Graph::permute() with the returned canonical
	 * labeling.
	 * Note that the computed canonical version may depend on the applied version
	 * of bliss as well as on some other options (for instance, the splitting
	 * heuristic selected with bliss::Graph::set_splitting_heuristic()).
	 */
	//const unsigned int* canonical_form(Stats& stats, void (*hook)(void* user_param, unsigned int n, const unsigned int* aut), void* hook_user_param);
const unsigned * canonical_form(Stats& stats, void (*hook)(void *user_param, unsigned int n, const unsigned int *aut), void *user_param) {
	report_hook = hook;
	report_user_param = user_param;
	search(true, stats);
	return best_path_labeling;
}
	/**
	 * Write the graph to a file in a variant of the DIMACS format.
	 * See the <A href="http://www.tcs.hut.fi/Software/bliss/">bliss website</A>
	 * for the definition of the file format.
	 * Note that in the DIMACS file the vertices are numbered from 1 to N while
	 * in this C++ API they are from 0 to N-1.
	 * Thus the vertex n in the file corresponds to the vertex n-1 in the API.
	 * \param fp  the file stream where the graph is written
	 */
	virtual void write_dimacs(FILE * const fp) = 0;

	/**
	 * Write the graph to a file in the graphviz dotty format.
	 * \param fp  the file stream where the graph is written
	 */
	virtual void write_dot(FILE * const fp) = 0;

	/**
	 * Write the graph in a file in the graphviz dotty format.
	 * Do nothing if the file cannot be written.
	 * \param file_name  the name of the file to which the graph is written
	 */
	virtual void write_dot(const char * const file_name) = 0;

	/**
	 * Get a hash value for the graph.
	 * \return  the hash value
	 */ 
	virtual unsigned int get_hash() = 0;

	/**
	 * Disable/enable the "long prune" method.
	 * The choice affects the computed canonical labelings;
	 * therefore, if you want to compare whether two graphs are isomorphic by
	 * computing and comparing (for equality) their canonical versions,
	 * be sure to use the same choice for both graphs.
	 * May not be called during the search, i.e. from an automorphism reporting
	 * hook function.
	 * \param active  if true, activate "long prune", deactivate otherwise
	 */
	void set_long_prune_activity(const bool active) {
		assert(!in_search);
		opt_use_long_prune = active;
	}

protected:
	/** \internal
	 * How much verbose output is produced (0 means none) */
	unsigned int verbose_level;
	/** \internal
	 * The output stream for verbose output. */
	FILE *verbstr;

protected:
	/** \internal
	 * The ordered partition used in the search algorithm. */
	Partition p;

	/** \internal
	 * Whether the search for automorphisms and a canonical labeling is
	 * in progress.
	 */
	bool in_search;

	/** \internal
	 * Is failure recording in use?
	 */
	bool opt_use_failure_recording;
	/* The "tree-specific" invariant value for the point when current path
	 * got different from the first path */
	unsigned int failure_recording_fp_deviation;

	/** \internal
	 * Is component recursion in use?
	 */
	bool opt_use_comprec;

	unsigned int refine_current_path_certificate_index;
	bool refine_compare_certificate;
	bool refine_equal_to_first;
	unsigned int refine_first_path_subcertificate_end;
	int refine_cmp_to_best;
	unsigned int refine_best_path_subcertificate_end;
	static const unsigned int CERT_SPLIT = 0; //UINT_MAX;
	static const unsigned int CERT_EDGE  = 1; //UINT_MAX-1;
	/** \internal
	 * Add a triple (v1,v2,v3) in the certificate.
	 * May modify refine_equal_to_first and refine_cmp_to_best.
	 * May also update eqref_hash and failure_recording_fp_deviation. */
	//void cert_add(const unsigned int v1, const unsigned int v2, const unsigned int v3);
// Certificate building
void cert_add(const unsigned int v1, const unsigned int v2, const unsigned int v3) {
	if(refine_compare_certificate) {
		if(refine_equal_to_first) {
			/* So far equivalent to the first path... */
			unsigned int index = certificate_current_path.size();
			if(index >= refine_first_path_subcertificate_end) {
				refine_equal_to_first = false;
			} else if(certificate_first_path[index] != v1) {
				refine_equal_to_first = false;
			} else if(certificate_first_path[++index] != v2) {
				refine_equal_to_first = false;
			} else if(certificate_first_path[++index] != v3) {
				refine_equal_to_first = false;
			} if(opt_use_failure_recording and !refine_equal_to_first) {
				/* We just became different from the first path,
				 * remember the deviation point tree-specific invariant
				 * for the use of failure recording */
				UintSeqHash h;
				h.update(v1);
				h.update(v2);
				h.update(v3);
				h.update(index);
				h.update(eqref_hash.get_value());
				failure_recording_fp_deviation = h.get_value();
			}
		}
		if(refine_cmp_to_best == 0) {
			/* So far equivalent to the current best path... */
			unsigned int index = certificate_current_path.size();
			if(index >= refine_best_path_subcertificate_end) {
				refine_cmp_to_best = 1;
			} else if(v1 > certificate_best_path[index]) {
				refine_cmp_to_best = 1;
			} else if(v1 < certificate_best_path[index]) {
				refine_cmp_to_best = -1;
			} else if(v2 > certificate_best_path[++index]) {
				refine_cmp_to_best = 1;
			} else if(v2 < certificate_best_path[index]) {
				refine_cmp_to_best = -1;
			} else if(v3 > certificate_best_path[++index]) {
				refine_cmp_to_best = 1;
			} else if(v3 < certificate_best_path[index]) {
				refine_cmp_to_best = -1;
			}
		}
		if((refine_equal_to_first == false) and (refine_cmp_to_best < 0))
			return;
	}
	/* Update the current path certificate */
	certificate_current_path.push_back(v1);
	certificate_current_path.push_back(v2);
	certificate_current_path.push_back(v3);
}
	/** \internal
	 * Add a redundant triple (v1,v2,v3) in the certificate.
	 * Can also just dicard the triple.
	 * May modify refine_equal_to_first and refine_cmp_to_best.
	 * May also update eqref_hash and failure_recording_fp_deviation. */
	//void cert_add_redundant(const unsigned int x, const unsigned int y, const unsigned int z);
void cert_add_redundant(const unsigned int v1, const unsigned int v2, const unsigned int v3) {
	return cert_add(v1, v2, v3);
}
	/**\internal
	 * Is the long prune method in use?
	 */
	bool opt_use_long_prune;
	/**\internal
	 * Maximum amount of memory (in megabytes) available for
	 * the long prune method
	 */
	static const unsigned int long_prune_options_max_mem = 50;
	/**\internal
	 * Maximum amount of automorphisms stored for the long prune method;
	 * less than this is stored if the memory limit above is reached first
	 */
	static const unsigned int long_prune_options_max_stored_auts = 100;

	unsigned int long_prune_max_stored_autss;
	std::vector<std::vector<bool> *> long_prune_fixed;
	std::vector<std::vector<bool> *> long_prune_mcrs;
	std::vector<bool> long_prune_temp;
	unsigned int long_prune_begin;
	unsigned int long_prune_end;
	/** \internal
	 * Initialize the "long prune" data structures.
	 */
	//void long_prune_init();
	/** \internal
	 * Release the memory allocated for "long prune" data structures.
	 */
	//void long_prune_deallocate();
	//void long_prune_add_automorphism(const unsigned int *aut);
	//std::vector<bool>& long_prune_get_fixed(const unsigned int index);
	//std::vector<bool>& long_prune_allocget_fixed(const unsigned int index);
	//std::vector<bool>& long_prune_get_mcrs(const unsigned int index);
	//std::vector<bool>& long_prune_allocget_mcrs(const unsigned int index);
	/** \internal
	 * Swap the i:th and j:th stored automorphism information;
	 * i and j must be "in window, i.e. in [long_prune_begin,long_prune_end[
	 */
	//void long_prune_swap(const unsigned int i, const unsigned int j);
//Long prune code
void long_prune_init() {
	const unsigned int N = get_nof_vertices();
	long_prune_temp.clear();
	long_prune_temp.resize(N);
	/* Of how many automorphisms we can store information in
	   the predefined, fixed amount of memory? */
	const unsigned int nof_fitting_in_max_mem =
		(long_prune_options_max_mem * 1024 * 1024) / (((N * 2) / 8)+1);
	long_prune_max_stored_autss = long_prune_options_max_stored_auts;
	/* Had some problems with g++ in using (a<b)?a:b when constants involved,
	   so had to make this in a stupid way... */
	if(nof_fitting_in_max_mem < long_prune_options_max_stored_auts)
		long_prune_max_stored_autss = nof_fitting_in_max_mem;
	long_prune_deallocate();
	long_prune_fixed.resize(N, 0);
	long_prune_mcrs.resize(N, 0);
	long_prune_begin = 0;
	long_prune_end = 0;
}

void long_prune_deallocate() {
	while(!long_prune_fixed.empty()) {
		delete long_prune_fixed.back();
		long_prune_fixed.pop_back();
	}
	while(!long_prune_mcrs.empty()) {
		delete long_prune_mcrs.back();
		long_prune_mcrs.pop_back();
	}
}

void long_prune_swap(const unsigned int i, const unsigned int j) {
	const unsigned int real_i = i % long_prune_max_stored_autss;
	const unsigned int real_j = j % long_prune_max_stored_autss;
	std::vector<bool>* tmp = long_prune_fixed[real_i];
	long_prune_fixed[real_i] = long_prune_fixed[real_j];
	long_prune_fixed[real_j] = tmp;
	tmp = long_prune_mcrs[real_i];
	long_prune_mcrs[real_i] = long_prune_mcrs[real_j];
	long_prune_mcrs[real_j] = tmp;
}

std::vector<bool>& long_prune_allocget_fixed(const unsigned int index) {
	const unsigned int i = index % long_prune_max_stored_autss;
	if(!long_prune_fixed[i])
		long_prune_fixed[i] = new std::vector<bool>(get_nof_vertices());
	return *long_prune_fixed[i];
}

std::vector<bool>& long_prune_get_fixed(const unsigned int index) {
	return *long_prune_fixed[index % long_prune_max_stored_autss];
}

std::vector<bool>& long_prune_allocget_mcrs(const unsigned int index) {
	const unsigned int i = index % long_prune_max_stored_autss;
	if(!long_prune_mcrs[i])
		long_prune_mcrs[i] = new std::vector<bool>(get_nof_vertices());
	return *long_prune_mcrs[i];
}

std::vector<bool>& long_prune_get_mcrs(const unsigned int index) {
	return *long_prune_mcrs[index % long_prune_max_stored_autss];
}

void long_prune_add_automorphism(const unsigned int* aut) {
	if(long_prune_max_stored_autss == 0) return;
	const unsigned int N = get_nof_vertices();
	/* If the buffer of stored auts is full, remove the oldest aut */
	if(long_prune_end - long_prune_begin == long_prune_max_stored_autss) {
		long_prune_begin++;
	}
	long_prune_end++;
	std::vector<bool>& fixed = long_prune_allocget_fixed(long_prune_end-1);
	std::vector<bool>& mcrs = long_prune_allocget_mcrs(long_prune_end-1);
	/* Mark nodes that are (i) fixed or (ii) minimal orbit representatives
	 * under the automorphism 'aut' */
	for(unsigned int i = 0; i < N; i++) {
		fixed[i] = (aut[i] == i);
		if(long_prune_temp[i] == false) {
			mcrs[i] = true;
			unsigned int j = aut[i];
			while(j != i) {
				long_prune_temp[j] = true;
				j = aut[j];
			}
		} else {
			mcrs[i] = false;
		}
		/* Clear the temp array on-the-fly... */
		long_prune_temp[i] = false;
	}
}

	/*
	 * Data structures and routines for refining the partition p into equitable
	 */
	Heap neighbour_heap;
	virtual bool split_neighbourhood_of_unit_cell(Partition::Cell *) = 0;
	virtual bool split_neighbourhood_of_cell(Partition::Cell * const) = 0;
	//void refine_to_equitable();
	//void refine_to_equitable(Partition::Cell * const unit_cell);
	//void refine_to_equitable(Partition::Cell * const unit_cell1, Partition::Cell * const unit_cell2);
void refine_to_equitable() {
	/* Start refinement from all cells -> push 'em all in the splitting queue */
	for(Partition::Cell* cell = p.first_cell; cell; cell = cell->next)
		p.splitting_queue_add(cell);
	do_refine_to_equitable();
}

void refine_to_equitable(Partition::Cell* const unit_cell) {
	p.splitting_queue_add(unit_cell);
	do_refine_to_equitable();
}

void refine_to_equitable(Partition::Cell* const unit_cell1, Partition::Cell* const unit_cell2) {
	p.splitting_queue_add(unit_cell1);
	p.splitting_queue_add(unit_cell2);
	do_refine_to_equitable();
}
	/** \internal
	 * \return false if it was detected that the current certificate
	 *         is different from the first and/or best (whether this is checked
	 *         depends on in_search and refine_compare_certificate flags.
	 */
	//bool do_refine_to_equitable();
bool do_refine_to_equitable() {
	eqref_hash.reset();
	while(!p.splitting_queue_is_empty()) {
		Partition::Cell* const cell = p.splitting_queue_pop();
		if(cell->is_unit()) {
			if(in_search) {
				const unsigned int index = cell->first;
				if(first_path_automorphism) {
					/* Build the (potential) automorphism on-the-fly */
					first_path_automorphism[first_path_labeling_inv[index]] =
						p.elements[index];
				}
				if(best_path_automorphism) {
					/* Build the (potential) automorphism on-the-fly */
					best_path_automorphism[best_path_labeling_inv[index]] =
						p.elements[index];
				}
			}
			const bool worse = split_neighbourhood_of_unit_cell(cell);
			if(in_search and worse) goto worse_exit;
		}
		else {
			const bool worse = split_neighbourhood_of_cell(cell);
			if(in_search and worse) goto worse_exit;
		}
	}
	return true;
worse_exit:
	/* Clear splitting_queue */
	p.splitting_queue_clear();
	return false;
}
	unsigned int eqref_max_certificate_index;
	/** \internal
	 * Whether eqref_hash is updated during equitable refinement process.
	 */
	bool compute_eqref_hash;
	UintSeqHash eqref_hash;
	/** \internal
	 * Check whether the current partition p is equitable.
	 * Performance: very slow, use only for debugging purposes.
	 */
	virtual bool is_equitable() const = 0;

	unsigned int *first_path_labeling;
	unsigned int *first_path_labeling_inv;
	Orbit         first_path_orbits;
	unsigned int *first_path_automorphism;
	unsigned int *best_path_labeling;
	unsigned int *best_path_labeling_inv;
	Orbit         best_path_orbits;
	unsigned int *best_path_automorphism;

	//void update_labeling(unsigned int * const lab);
/** \internal
 * Assign the labeling induced by the current partition 'this.p' to
 * \a labeling.
 * That is, if the partition is [[2,0],[1]],
 * then \a labeling will map 0 to 1, 1 to 2, and 2 to 0.
 */
void update_labeling(unsigned int* const labeling) {
	const unsigned int N = get_nof_vertices();
	unsigned int* ep = p.elements;
	for(unsigned int i = 0; i < N; i++, ep++)
		labeling[*ep] = i;
}
	//void update_labeling_and_its_inverse(unsigned int * const lab, unsigned int * const lab_inv);
/** \internal
 * The same as update_labeling() except that the inverse of the labeling
 * is also produced and assigned to \a labeling_inv.
 */
void update_labeling_and_its_inverse(unsigned int* const labeling, unsigned int* const labeling_inv) {
	const unsigned int N = get_nof_vertices();
	unsigned int* ep = p.elements;
	unsigned int* clip = labeling_inv;
	for(unsigned int i = 0; i < N; ) {
		labeling[*ep] = i;
		i++;
		*clip = *ep;
		ep++;
		clip++;
	}
}
	void update_orbit_information(Orbit &o, const unsigned int *perm) {
		const unsigned int N = get_nof_vertices();
		for(unsigned int i = 0; i < N; i++)
			if(perm[i] != i) o.merge_orbits(i, perm[i]);
	}
	//void reset_permutation(unsigned int *perm);
	/* Mainly for debugging purposes */
	//virtual bool is_automorphism(unsigned int* const perm);

// \internal
// Reset the permutation \a perm to the identity permutation.
void reset_permutation(unsigned int* perm) {
	const unsigned int N = get_nof_vertices();
	for(unsigned int i = 0; i < N; i++, perm++)
		*perm = i;
}

virtual bool is_automorphism(unsigned int* const perm) {
	_INTERNAL_ERROR();
	return false;
}
	std::vector<unsigned int> certificate_current_path;
	std::vector<unsigned int> certificate_first_path;
	std::vector<unsigned int> certificate_best_path;
	unsigned int certificate_index;
	virtual void initialize_certificate() = 0;
	virtual void remove_duplicate_edges() = 0;
	virtual void make_initial_equitable_partition() = 0;
	virtual Partition::Cell* find_next_cell_to_be_splitted(Partition::Cell *cell) = 0;
	//void search(const bool canonical, Stats &stats);
#include "search.h"
	void (*report_hook)(void *user_param, unsigned int n, const unsigned int *aut);
	void *report_user_param;
	/*
	 *
	 * Nonuniform component recursion (NUCR)
	 *
	 */

	/** The currently traversed component */
	unsigned int cr_level;

	/** \internal
	 * The "Component End Point" data structure
	 */
	class CR_CEP {
		public:
			/** At which level in the search was this CEP created */
			unsigned int creation_level;
			/** The current component has been fully traversed when the partition has
			 * this many discrete cells left */
			unsigned int discrete_cell_limit;
			/** The component to be traversed after the current one */
			unsigned int next_cr_level;
			/** The next component end point */
			unsigned int next_cep_index;
			bool first_checked;
			bool best_checked;
	};
	/** \internal
	 * A stack for storing Component End Points
	 */
	std::vector<CR_CEP> cr_cep_stack;

	/** \internal
	 * Find the first non-uniformity component at the component recursion
	 * level \a level.
	 * The component is stored in \a cr_component.
	 * If no component is found, \a cr_component is empty.
	 * Returns false if all the cells in the component recursion level \a level
	 * were discrete.
	 * Modifies the max_ival and max_ival_count fields of Partition:Cell
	 * (assumes that they are 0 when called and
	 *  quarantees that they are 0 when returned).
	 */
	virtual bool nucr_find_first_component(const unsigned int level) = 0;
	virtual bool nucr_find_first_component(const unsigned int level,
			std::vector<unsigned int>& component,
			unsigned int& component_elements,
			Partition::Cell*& sh_return) = 0;
	/** \internal
	 * The non-uniformity component found by nucr_find_first_component()
	 * is stored here.
	 */
	std::vector<unsigned int> cr_component;
	/** \internal
	 * The number of vertices in the component \a cr_component
	 */
	unsigned int cr_component_elements;
};

// Assumes that the elements in the cell are sorted according to their invariant values.
Partition::Cell* Partition::split_cell(Partition::Cell* const original_cell) {
  Partition::Cell* cell = original_cell;
  const bool original_cell_was_in_splitting_queue =
    original_cell->in_splitting_queue;
  Partition::Cell* largest_new_cell = 0;

  while(true) {
      unsigned int* ep = elements + cell->first;
      const unsigned int* const lp = ep + cell->length;
      const unsigned int ival = invariant_values[*ep];
      invariant_values[*ep] = 0;
      element_to_cell_map[*ep] = cell;
      in_pos[*ep] = ep;
      ep++;
      while(ep < lp) {
	  const unsigned int e = *ep;
	  if(invariant_values[e] != ival)
	    break;
	  invariant_values[e] = 0;
	  in_pos[e] = ep;
	  ep++;
	  element_to_cell_map[e] = cell;
	}
      if(ep == lp) break;
      Partition::Cell* const new_cell = aux_split_in_two(cell, (ep - elements) - cell->first);
      if(graph and graph->compute_eqref_hash) {
	  graph->eqref_hash.update(new_cell->first);
	  graph->eqref_hash.update(new_cell->length);
	  graph->eqref_hash.update(ival);
	}
      /* Add cells in splitting_queue */
      assert(!new_cell->is_in_splitting_queue());
      if(original_cell_was_in_splitting_queue)
	{
	  /* In this case, all new cells are inserted in splitting_queue */
	  assert(cell->is_in_splitting_queue());
	  splitting_queue_add(new_cell);
	}
      else
	{
	  /* Otherwise, we can omit one new cell from splitting_queue */
	  assert(!cell->is_in_splitting_queue());
	  if(largest_new_cell == 0) {
	    largest_new_cell = cell;
	  } else {
	    assert(!largest_new_cell->is_in_splitting_queue());
	    if(cell->length > largest_new_cell->length) {
	      splitting_queue_add(largest_new_cell);
	      largest_new_cell = cell;
	    } else {
	      splitting_queue_add(cell);
	    }
	  }
	}
      /* Process the rest of the cell */
      cell = new_cell;
    }

  
  if(original_cell == cell) {
    /* All the elements in cell had the same invariant value */
    return cell;
  }

  /* Add cells in splitting_queue */
  if(!original_cell_was_in_splitting_queue) {
      /* Also consider the last new cell */
      assert(largest_new_cell);
      if(cell->length > largest_new_cell->length) {
	  splitting_queue_add(largest_new_cell);
	  largest_new_cell = cell;
	} else {
	  splitting_queue_add(cell);
	}
      if(largest_new_cell->is_unit()) {
	  /* Needed in certificate computation */
	  splitting_queue_add(largest_new_cell);
	}
    }
  return cell;
}

}

#endif
