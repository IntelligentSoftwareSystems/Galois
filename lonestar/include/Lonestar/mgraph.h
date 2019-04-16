#ifndef __MGRAPH_HPP__
#define __MGRAPH_HPP__
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

struct MEdge {
	int src;
	int dst;
	int elabel;
	//unsigned id;
	MEdge() : src(0), dst(0), elabel(0) {}
	MEdge(int from, int to, int el) :
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
	MGraph() : symmetrize_(false), directed_(false) {}
	MGraph(bool sym) : symmetrize_(sym), directed_(false) {}
	int * out_rowptr() const { return rowptr_; }
	int * out_colidx() const { return colidx_; }
	int * labels() { return labels_.data(); }
	int get_label(int n) { return labels_[n]; }
	int get_offset(int n) { return rowptr_[n]; }
	int get_dest(int n) { return colidx_[n]; }
	int get_weight(int n) { return weight_[n]; }
	int out_degree(int n) const { return rowptr_[n+1] - rowptr_[n]; }
	bool directed() const { return directed_; }
	int num_vertices() const { return num_vertices_; }
	int num_edges() const { return num_edges_; }

	void read_txt(const char *filename) {
		std::ifstream is;
		is.open(filename, std::ios::in);
		char line[1024];
		std::vector<std::string> result;
		std::set<std::pair<int,int> > edge_set;
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
				int src    = atoi(result[1].c_str());
				int dst    = atoi(result[2].c_str());
				int elabel = atoi(result[3].c_str());
				if(labels_.size() <= src || labels_.size() <= dst) {
					std::cerr << "Format Error:  define vertex lists before edges, from: " << src 
						<< "; to: " << dst << "; vertex count: " << labels_.size() << std::endl;
					exit(1);
				}
				if (src == dst) continue; // remove self-loop
				if (edge_set.find(std::pair<int, int>(src, dst)) == edge_set.end()) {
					edge_set.insert(std::pair<int, int>(src, dst));
					el.push_back(MEdge(src, dst, elabel));
					if(directed_ == false) {
						edge_set.insert(std::pair<int, int>(dst, src));
						el.push_back(MEdge(dst, src, elabel));
					}
				}
			}
		}
		is.close();
		num_vertices_ = labels_.size();
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
			int src = atoi(result[0].c_str());
			labels_.resize(src + 1);
			labels_[src] = atoi(result[1].c_str());
			int elabel = 0;
			std::set<std::pair<int, int> > neighbors;
			for(size_t i = 2; i < result.size(); i++) {
				int dst = atoi(result[i].c_str());
				if (src == dst) continue; // remove self-loop
#ifdef USE_ELABEL
				elabel = atoi(result[i].c_str());
#endif
				neighbors.insert(std::pair<int, int>(dst, elabel)); // remove redundant edge
			}
			for (std::set<std::pair<int, int> >::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
				el.push_back(MEdge(src, it->first, it->second));
		}
		is.close();
		num_vertices_ = labels_.size();
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
				if (undirected || symmetrize)
					el.push_back(MEdge(v - 1, u - 1, w));
			} else {
				int v;
				edge_stream >> v;
				el.push_back(MEdge(u - 1, v - 1, 1));
				if (undirected || symmetrize)
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
	void print_graph() {
		if (directed_) std::cout << "directed graph\n";
		else std::cout << "undirected graph\n";
		for (int n = 0; n < num_vertices_; n ++) {
			int row_begin = rowptr_[n];
			int row_end = rowptr_[n+1];
			std::cout << "vertex " << n << ": label = " << labels_[n] << " edgelist = [ ";
			for (int offset = row_begin; offset < row_end; offset ++) {
				int dst = colidx_[offset];
				std::cout << dst << " ";
			}
			std::cout << "]" << std::endl;
		}
	}

private:
	bool symmetrize_; // whether to symmetrize a directed graph
	bool directed_;
	int num_vertices_;
	int num_edges_;
	int *rowptr_;
	int *colidx_;
	int *weight_;
	//int *in_rowptr_;
	//int *in_colidx_;
	std::vector<int> labels_;
	MEdgeList el;
	std::vector<std::vector<MEdge> > vertices;

	std::vector<int> CountDegrees(const MEdgeList &el, bool transpose) {
		std::vector<int> degrees(num_vertices_, 0);
		for (auto it = el.begin(); it < el.end(); it++) {
			MEdge e = *it;
			if (symmetrize_ || (!symmetrize_ && !transpose))
				degrees[e.src] ++;
			if (symmetrize_ || (!symmetrize_ && transpose))
				degrees[e.dst] ++;
		}
		return degrees;
	}
	void MakeCSRFromEL(bool transpose) {
		std::vector<int> degrees = CountDegrees(el, transpose);
		std::vector<int> offsets = PrefixSum(degrees);
		num_edges_ = offsets[num_vertices_];
		weight_ = new int[num_edges_];
		colidx_ = new int[num_edges_];
		rowptr_ = new int[num_vertices_+1]; 
		for (int i = 0; i < num_vertices_+1; i ++) rowptr_[i] = offsets[i];
		for (auto it = el.begin(); it < el.end(); it++) {
			MEdge e = *it;
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
	void MakeCSR(bool transpose) {
		std::vector<int> degrees(num_vertices_);
		for (int i = 0; i < num_vertices_; i ++)
			degrees[i] = vertices[i].size();
		std::vector<int> offsets = PrefixSum(degrees);
		assert(num_edges_ == offsets[num_vertices_]);
		weight_ = new int[num_edges_];
		colidx_ = new int[num_edges_];
		rowptr_ = new int[num_vertices_+1]; 
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
	}
	void MakeGraphFromEL() {
		//MakeCSRFromEL(false);
		SquishGraph();
		MakeCSR(false);
	}
	static std::vector<int> PrefixSum(const std::vector<int> &degrees) {
		std::vector<int> sums(degrees.size() + 1);
		int total = 0;
		for (size_t n=0; n < degrees.size(); n++) {
			sums[n] = total;
			total += degrees[n];
		}
		sums[degrees.size()] = total;
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
