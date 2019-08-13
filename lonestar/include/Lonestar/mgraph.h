#ifndef __MGRAPH_HPP__
#define __MGRAPH_HPP__
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "core.h"
#include "common_types.h"

struct MEdge {
	IndexT src;
	IndexT dst;
	ValueT elabel;
	MEdge() : src(0), dst(0), elabel(0) {}
	MEdge(IndexT from, IndexT to, ValueT el) :
		src(from), dst(to), elabel(el) {}
	std::string to_string() const {
		std::stringstream ss;
		ss << "e(" << src << "," << dst << "," << elabel << ")";
		return ss.str();
	}
};
typedef std::vector<MEdge> MEdgeList;

class MGraph {
public:
	//MEdgeList el;
	MGraph() : need_relabel_edges(false), need_dag(false), symmetrize_(false), directed_(false) {}
	MGraph(bool relabel_edges) : need_relabel_edges(relabel_edges), need_dag(false), symmetrize_(false), directed_(false) {}
	MGraph(bool relabel_edges, bool dag) : need_relabel_edges(relabel_edges), need_dag(dag), symmetrize_(false), directed_(false) {}
	void clean() {
		el.clear();
		delete[] rowptr_;
		delete[] colidx_;
		delete[] weight_;
		rank.clear();
		degrees.clear();
		labels_.clear();
		vertices.clear();
	}
	IndexT * out_rowptr() const { return rowptr_; }
	IndexT * out_colidx() const { return colidx_; }
	ValueT * labels() { return labels_.data(); }
	ValueT get_label(int n) { return labels_[n]; }
	IndexT get_offset(int n) { return rowptr_[n]; }
	IndexT get_dest(int n) { return colidx_[n]; }
	ValueT get_weight(int n) { return weight_[n]; }
	int get_core() { return core; }
	int out_degree(int n) const { return rowptr_[n+1] - rowptr_[n]; }
	bool directed() const { return directed_; }
	int num_vertices() const { return num_vertices_; }
	int num_edges() const { return num_edges_; }

	void read_txt(const char *filename, bool symmetrize = true) {
		std::ifstream is;
		is.open(filename, std::ios::in);
		char line[1024];
		std::vector<std::string> result;
		std::set<std::pair<IndexT, IndexT> > edge_set;
		//clear();
		while(true) {
			unsigned pos = is.tellg();
			if(!is.getline(line, 1024)) break;
			result.clear();
			split(line, result);
			if(result.empty()) {
			} else if(result[0] == "t") {
				if(!labels_.empty()) {   // use as delimiter
					is.seekg(pos, std::ios_base::beg);
					break;
				} else { }
			} else if(result[0] == "v" && result.size() >= 3) {
				unsigned id = atoi(result[1].c_str());
				labels_.resize(id + 1);
				labels_[id] = atoi(result[2].c_str());
			} else if(result[0] == "e" && result.size() >= 4) {
				IndexT src    = atoi(result[1].c_str());
				IndexT dst    = atoi(result[2].c_str());
				ValueT elabel = atoi(result[3].c_str());
				assert(labels_.size() > src && labels_.size() > dst);
				if (src == dst) continue; // remove self-loop
				if (edge_set.find(std::pair<IndexT, IndexT>(src, dst)) == edge_set.end()) {
					edge_set.insert(std::pair<IndexT, IndexT>(src, dst));
					el.push_back(MEdge(src, dst, elabel));
					if(symmetrize) {
						edge_set.insert(std::pair<IndexT, IndexT>(dst, src));
						el.push_back(MEdge(dst, src, elabel));
					}
				}
			}
		}
		is.close();
		num_vertices_ = labels_.size();
		int num_labels = count_unique_labels();
		std::cout << "Number of unique vertex label values: " << num_labels << std::endl;
		num_edges_ = el.size();
		if (!directed_) symmetrize_ = false; // no need to symmetrize undirected graph
		MakeGraphFromEL();
	}
	void read_adj(const char *filename) {
		FILE* fd = fopen(filename, "r");
		assert(fd != NULL);
		char buf[2048];
		int size = 0, maxsize = 0;
		while (fgets(buf, 2048, fd) != NULL) {
			int len = strlen(buf);
			size += len;
			if (buf[len-1] == '\n') {
				maxsize = std::max(size, maxsize);
				size = 0;
			}
		}
		fclose(fd);

		std::ifstream is;
		is.open(filename, std::ios::in);
		//char line[1024];
		char*line = new char[maxsize+1];
		std::vector<std::string> result;
		while(is.getline(line, maxsize+1)) {
			result.clear();
			split(line, result);
			IndexT src = atoi(result[0].c_str());
			labels_.resize(src + 1);
			labels_[src] = atoi(result[1].c_str());
			ValueT elabel = 0;
			std::set<std::pair<IndexT, ValueT> > neighbors;
			for(size_t i = 2; i < result.size(); i++) {
				IndexT dst = atoi(result[i].c_str());
				if (src == dst) continue; // remove self-loop
#ifdef USE_ELABEL
				elabel = atoi(result[i].c_str());
#endif
				neighbors.insert(std::pair<IndexT, ValueT>(dst, elabel)); // remove redundant edge
			}
			for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
				el.push_back(MEdge(src, it->first, it->second));
		}
		is.close();
		num_vertices_ = labels_.size();
		int num_labels = count_unique_labels();
		std::cout << "Number of unique vertex label values: " << num_labels << std::endl;
		num_edges_ = el.size();
		if (!directed_) symmetrize_ = false; // no need to symmetrize undirected graph
		MakeGraphFromEL();
	}
	void read_mtx(const char *filename, bool symmetrize = false, bool needs_weights = false) {
		std::ifstream in;
		in.open(filename, std::ios::in);
		std::string start, object, format, field, symmetry, line;
		in >> start >> object >> format >> field >> symmetry >> std::ws;
		if (start != "%%MatrixMarket") {
			std::cout << ".mtx file did not start with %%MatrixMarket" << std::endl;
			std::exit(-21);
		}
		if ((object != "matrix") || (format != "coordinate")) {
			std::cout << "only allow matrix coordinate format for .mtx" << std::endl;
			std::exit(-22);
		}
		if (field == "complex") {
			std::cout << "do not support complex weights for .mtx" << std::endl;
			std::exit(-23);
		}
		bool read_weights;
		if (field == "pattern") {
			read_weights = false;
		} else if ((field == "real") || (field == "double") || (field == "integer")) {
			read_weights = true;
		} else {
			std::cout << "unrecognized field type for .mtx" << std::endl;
			std::exit(-24);
		}
		bool undirected;
		if (symmetry == "symmetric") {
			undirected = true;
		} else if ((symmetry == "general") || (symmetry == "skew-symmetric")) {
			undirected = false;
		} else {
			std::cout << "unsupported symmetry type for .mtx" << std::endl;
			std::exit(-25);
		}
		while (true) {
			char c = in.peek();
			if (c == '%') { in.ignore(200, '\n');
			} else { break; }
		}
		int64_t m, n, nonzeros;
		in >> m >> n >> nonzeros >> std::ws;
		if (m != n) {
			std::cout << m << " " << n << " " << nonzeros << std::endl;
			std::cout << "matrix must be square for .mtx" << std::endl;
			std::exit(-26);
		}
		while (std::getline(in, line)) {
			std::istringstream edge_stream(line);
			int u;
			edge_stream >> u;
			if (read_weights) {
				int v;
				edge_stream >> v;
				int w = 1;
				el.push_back(MEdge(u - 1, v - 1, w));
				if (symmetrize)
					el.push_back(MEdge(v - 1, u - 1, w));
			} else {
				int v;
				edge_stream >> v;
				el.push_back(MEdge(u - 1, v - 1, 1));
				if (symmetrize)
					el.push_back(MEdge(v - 1, u - 1, 1));
			}
		}
		in.close();
		labels_.resize(m);
		directed_ = !undirected;
		if (undirected) symmetrize_ = false; // no need to symmetrize undirected graph
		for (int i = 0; i < m; i ++) { labels_[i] = rand() % 10 + 1; }
		num_vertices_ = m;
		num_edges_ = el.size();
		MakeGraphFromEL();
	}
	void read_gr(Graph& g) {
		num_vertices_ = g.size();
		//degrees.resize(num_vertices_);
		//std::fill(degrees.begin(), degrees.end(), 0);
		//std::cout << "Assume the input graph is clean and symmetric (.csgr)\n";
		for (auto it = g.begin(); it != g.end(); it ++) {
			GNode src = *it;
			for (auto e : g.edges(src)) {
				GNode dst = g.getEdgeDst(e);
				el.push_back(MEdge(src, dst, 1));
				//degrees[src] ++;
			}
		}
		assert(el.size() == g.sizeEdges());
		num_edges_ = el.size();
		labels_.resize(num_vertices_);
		for (int i = 0; i < num_vertices_; i ++) { labels_[i] = g.getData(i); }
		MakeGraphFromEL();
	}
	void print_graph() {
		if (directed_) std::cout << "directed graph\n";
		else std::cout << "undirected graph\n";
		for (int n = 0; n < num_vertices_; n ++) {
			IndexT row_begin = rowptr_[n];
			IndexT row_end = rowptr_[n+1];
			std::cout << "vertex " << n << ": label = " << labels_[n] << " edgelist = [ ";
			for (IndexT offset = row_begin; offset < row_end; offset ++) {
				IndexT dst = colidx_[offset];
				std::cout << dst << " ";
			}
			std::cout << "]" << std::endl;
		}
	}

private:
	MEdgeList el;
	bool need_relabel_edges;
	bool need_dag;
	bool symmetrize_; // whether to symmetrize a directed graph
	bool directed_;
	int num_vertices_;
	int num_edges_;
	IndexT *rowptr_;
	IndexT *colidx_;
	ValueT *weight_;
	int core;
	//int *in_rowptr_;
	//int *in_colidx_;
	std::vector<int> rank;
	std::vector<IndexT> degrees;
	std::vector<ValueT> labels_;
	std::vector<std::vector<MEdge> > vertices;

	int count_unique_labels() {
		std::set<int> s;
		int res = 0;
		for (size_t i = 0; i < labels_.size(); i++) {
			if (s.find(labels_[i]) == s.end()) {
				s.insert(labels_[i]);
				res++;
			}
		}
		return res;
	}
	void CountDegrees(const MEdgeList &el) {
		degrees.resize(num_vertices_);
		std::fill(degrees.begin(), degrees.end(), 0);
		for (auto it = el.begin(); it < el.end(); it++) {
			MEdge e = *it;
			degrees[e.src] ++;
			if (symmetrize_) degrees[e.dst] ++;
		}
	}
	void MakeCSRFromEL() {
		CountDegrees(el);
		core = *(std::max_element(degrees.begin(), degrees.end()));
		std::vector<int> offsets = PrefixSum(degrees);
		num_edges_ = offsets[num_vertices_];
		weight_ = new ValueT[num_edges_];
		colidx_ = new IndexT[num_edges_];
		rowptr_ = new IndexT[num_vertices_+1]; 
		for (int i = 0; i < num_vertices_+1; i ++) rowptr_[i] = offsets[i];
		for (auto it = el.begin(); it < el.end(); it++) {
			MEdge e = *it;
			weight_[offsets[e.src]] = e.elabel;
			colidx_[offsets[e.src]++] = e.dst;
			if (symmetrize_) {
				weight_[offsets[e.dst]] = e.elabel;
				colidx_[offsets[e.dst]++] = e.src;
			}
		}
	}
	//computing degeneracy ordering and core value
	void ord_core() {
		rank.resize(num_vertices_);
		unsigned *d0 = (unsigned *)calloc(num_vertices_, sizeof(unsigned));
		IndexT *cd0 = (IndexT*)malloc((num_vertices_ + 1)*sizeof(IndexT));
		IndexT *adj0 = (IndexT*)malloc(2*num_edges_*sizeof(IndexT));
		for (int i = 0; i < num_edges_; i ++) {
			d0[el[i].src]++;
			d0[el[i].dst]++;
		}
		cd0[0] = 0;
		for (int i = 1; i < num_vertices_ + 1; i ++) {
			cd0[i] = cd0[i-1] + d0[i-1];
			d0[i-1] = 0;
		}
		for (int i = 0; i < num_edges_; i ++) {
			adj0[ cd0[el[i].src] + d0[ el[i].src]++] = el[i].dst;
			adj0[ cd0[el[i].dst] + d0[ el[i].dst]++] = el[i].src;
		}
		bheap heap;
		heap.mkheap(num_vertices_, d0);
		int r = 0;
		for (int i = 0; i < num_vertices_; i ++) {
			keyvalue kv = heap.popmin();
			rank[kv.key] = num_vertices_ - (++r);
			for (IndexT j = cd0[kv.key]; j < cd0[kv.key + 1]; j ++) {
				heap.update(adj0[j]);
			}
		}
		free(d0);
		free(cd0);
		free(adj0);
	}
	// relabel vertices by descending degree order (do not apply to weighted graphs)
	void DegreeRanking() {
		std::cout << " Relabeling vertices by descending degree order\n";
		typedef std::pair<int, IndexT> degree_node_p;
		std::vector<degree_node_p> degree_id_pairs(num_vertices_);
		for (IndexT n = 0; n < num_vertices_; n++)
			degree_id_pairs[n] = std::make_pair(out_degree(n), n);
		std::sort(degree_id_pairs.begin(), degree_id_pairs.end(), std::greater<degree_node_p>());
		degrees.resize(num_vertices_);
		std::fill(degrees.begin(), degrees.end(), 0);
		std::vector<IndexT> new_ids(num_vertices_);
		for (IndexT n = 0; n < num_vertices_; n++) {
			degrees[n] = degree_id_pairs[n].first;
			new_ids[degree_id_pairs[n].second] = n;
		}
		std::vector<IndexT> offsets = PrefixSum(degrees);
		IndexT *index = new IndexT[num_vertices_+1];
		IndexT *neighs = new IndexT[num_edges_];
		for (IndexT i = 0; i < num_vertices_+1; i++) index[i] = offsets[i];
		for (IndexT u = 0; u < num_vertices_; u++) {
			for (IndexT offset = get_offset(u); offset < get_offset(u+1); offset++) {
				IndexT v = get_dest(offset);
				neighs[offsets[new_ids[u]]++] = new_ids[v];
			}
			std::sort(neighs+index[new_ids[u]], neighs+index[new_ids[u]+1]);
		}
		delete rowptr_;
		delete colidx_;
		rowptr_ = index;
		colidx_ = neighs;
	}
	void ConstructDAG() {
		std::cout << "Constructing DAG\n";
		MEdgeList new_el;
		int count = 0;
		for (int i = 0; i < num_edges_; i ++) {
			IndexT from = el[i].src;
			IndexT to = el[i].dst;
			if (degrees[from] < degrees[to] || (degrees[from] == degrees[to] && from < to)) {
				new_el.push_back(el[i]);
				count ++;
			}
		}
		el = new_el;
		assert(count == el.size());
		num_edges_ = count;
	}
	void RelabelEdges() {
		std::cout << "Relabeling edges\n";
		ord_core();
		//for (int i = 0; i < num_vertices_; i ++)
		//	std::cout << i << " --> " << rank[i] << "\n";
		for (int i = 0; i < num_edges_; i ++) {
			int source = rank[el[i].src];
			int target = rank[el[i].dst];
			if (source < target) {
				int tmp = source;
				source = target;
				target = tmp;
			}
			el[i].src = source;
			el[i].dst = target;
		}
		std::vector<ValueT> new_labels(num_vertices_);
		for (int i = 0; i < num_vertices_; i ++)
			new_labels[rank[i]] = labels_[i];
		for (int i = 0; i < num_vertices_; i ++)
			labels_[i] = new_labels[i];
	}
	void MakeCSR(bool transpose) {
		degrees.resize(num_vertices_);
		std::fill(degrees.begin(), degrees.end(), 0);
		for (int i = 0; i < num_vertices_; i ++)
			degrees[i] = vertices[i].size();
		core = *(std::max_element(degrees.begin(), degrees.end()));
		//printf("core value (max truncated degree) = %u\n", core);
		std::vector<int> offsets = PrefixSum(degrees);
		assert(num_edges_ == offsets[num_vertices_]);
		weight_ = new ValueT[num_edges_];
		colidx_ = new IndexT[num_edges_];
		rowptr_ = new IndexT[num_vertices_+1]; 
		for (int i = 0; i < num_vertices_+1; i ++) rowptr_[i] = offsets[i];
		for (int i = 0; i < num_vertices_; i ++) {
			for (auto it = vertices[i].begin(); it < vertices[i].end(); it ++) {
				MEdge e = *it;
				assert(i == e.src);
				if (symmetrize_ || (!symmetrize_ && !transpose)) {
					weight_[offsets[e.src]] = e.elabel;
					colidx_[offsets[e.src]++] = e.dst;
				}
				if (symmetrize_ || (!symmetrize_ && transpose)) {
					weight_[offsets[e.dst]] = e.elabel;
					colidx_[offsets[e.dst]++] = e.src;
				}
			}
		}
	}
	static bool compare_id(MEdge a, MEdge b) { return (a.dst < b.dst); }
	void SquishGraph(bool remove_selfloops = true, bool remove_redundents = true) {
		std::vector<MEdge> neighbors;
		for (int i = 0; i < num_vertices_; i++)
			vertices.push_back(neighbors);
		//assert(num_edges_ == el.size());
		for (int i = 0; i < num_edges_; i ++)
			vertices[el[i].src].push_back(el[i]);
		el.clear();
		printf("Sorting the neighbor lists...");
		for (int i = 0; i < num_vertices_; i ++)
			std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
		printf(" Done\n");
		//remove self loops
		int num_selfloops = 0;
		if(remove_selfloops) {
			printf("Removing self loops...");
			for(int i = 0; i < num_vertices_; i ++) {
				for(unsigned j = 0; j < vertices[i].size(); j ++) {
					if(i == vertices[i][j].dst) {
						vertices[i].erase(vertices[i].begin()+j);
						num_selfloops ++;
						j --;
					}
				}
			}
			printf(" %d selfloops are removed\n", num_selfloops);
			num_edges_ -= num_selfloops;
		}
		// remove redundent
		int num_redundents = 0;
		if(remove_redundents) {
			printf("Removing redundent edges...");
			for (int i = 0; i < num_vertices_; i ++) {
				for (unsigned j = 1; j < vertices[i].size(); j ++) {
					if (vertices[i][j].dst == vertices[i][j-1].dst) {
						vertices[i].erase(vertices[i].begin()+j);
						num_redundents ++;
						j --;
					}
				}
			}
			printf(" %d redundent edges are removed\n", num_redundents);
			num_edges_ -= num_redundents;
		}
		if(need_dag) {
			int num_dag = 0;
			std::cout << "Constructing DAG...";
			degrees.resize(num_vertices_);
			for (int i = 0; i < num_vertices_; i ++)
				degrees[i] = vertices[i].size();
			for (int i = 0; i < num_vertices_; i ++) {
				for (unsigned j = 0; j < vertices[i].size(); j ++) {
					int to = vertices[i][j].dst;
					if (degrees[to] < degrees[i] || (degrees[to] == degrees[i] && to < i)) {
						vertices[i].erase(vertices[i].begin()+j);
						num_dag ++;
						j --;
					}
				}
			}
			printf(" %d dag edges are removed\n", num_dag);
			num_edges_ -= num_dag;
		}
	}
	void MakeGraphFromEL() {
		if (need_relabel_edges) RelabelEdges();
		//if (need_dag) ConstructDAG();
		//MakeCSRFromEL();
		SquishGraph();
		MakeCSR(false);
		//if (!need_relabel_edges) DegreeRanking();
	}
	static std::vector<int> PrefixSum(const std::vector<int> &vec) {
		std::vector<int> sums(vec.size() + 1);
		int total = 0;
		for (size_t n=0; n < vec.size(); n++) {
			sums[n] = total;
			total += vec[n];
		}
		sums[vec.size()] = total;
		return sums;
	}
	inline void split(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
		std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
		std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
		while (std::string::npos != pos || std::string::npos != lastPos) {
			tokens.push_back(str.substr(lastPos, pos - lastPos));
			lastPos = str.find_first_not_of(delimiters, pos);
			pos = str.find_first_of(delimiters, lastPos);
		}
	}
};
#endif
