#pragma once
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "pangolin/types.h"

struct MEdge {
  IndexT src;
  IndexT dst;
  ValueT elabel;
  MEdge() : src(0), dst(0), elabel(0) {}
  MEdge(IndexT from, IndexT to, ValueT el) : src(from), dst(to), elabel(el) {}
  std::string to_string() const {
    std::stringstream ss;
    ss << "e(" << src << "," << dst << "," << elabel << ")";
    return ss.str();
  }
};
typedef std::vector<MEdge> MEdgeList;

class MGraph {
public:
  // MEdgeList el;
  MGraph() : need_dag(false), symmetrize_(false), directed_(false) {}
  MGraph(bool dag) : need_dag(dag), symmetrize_(false), directed_(false) {}
  void clean() {
    el.clear();
    delete[] rowptr_;
    delete[] colidx_;
    delete[] weight_;
    degrees.clear();
    labels_.clear();
    vertices.clear();
  }
  IndexT* out_rowptr() const { return rowptr_; }
  IndexT* out_colidx() const { return colidx_; }
  ValueT* labels() { return labels_.data(); }
  ValueT get_label(IndexT n) { return labels_[n]; }
  IndexT get_offset(IndexT n) { return rowptr_[n]; }
  IndexT get_dest(IndexT n) { return colidx_[n]; }
  ValueT get_weight(IndexT n) { return weight_[n]; }
  unsigned get_max_degree() { return max_degree; }
  unsigned out_degree(IndexT n) const { return rowptr_[n + 1] - rowptr_[n]; }
  bool directed() const { return directed_; }
  size_t num_vertices() const { return num_vertices_; }
  size_t num_edges() const { return num_edges_; }

  void read_txt(const char* filename, bool symmetrize = true) {
    std::ifstream is;
    is.open(filename, std::ios::in);
    char line[1024];
    std::vector<std::string> result;
    std::set<std::pair<IndexT, IndexT>> edge_set;
    // clear();
    while (true) {
      unsigned pos = is.tellg();
      if (!is.getline(line, 1024))
        break;
      result.clear();
      split(line, result);
      if (result.empty()) {
      } else if (result[0] == "t") {
        if (!labels_.empty()) { // use as delimiter
          is.seekg(pos, std::ios_base::beg);
          break;
        } else {
        }
      } else if (result[0] == "v" && result.size() >= 3) {
        unsigned id = atoi(result[1].c_str());
        labels_.resize(id + 1);
        labels_[id] = atoi(result[2].c_str());
      } else if (result[0] == "e" && result.size() >= 4) {
        IndexT src    = atoi(result[1].c_str());
        IndexT dst    = atoi(result[2].c_str());
        ValueT elabel = atoi(result[3].c_str());
        assert(labels_.size() > src && labels_.size() > dst);
        if (src == dst)
          continue; // remove self-loop
        if (edge_set.find(std::pair<IndexT, IndexT>(src, dst)) ==
            edge_set.end()) {
          edge_set.insert(std::pair<IndexT, IndexT>(src, dst));
          el.push_back(MEdge(src, dst, elabel));
          if (symmetrize) {
            edge_set.insert(std::pair<IndexT, IndexT>(dst, src));
            el.push_back(MEdge(dst, src, elabel));
          }
        }
      }
    }
    is.close();
    num_vertices_   = labels_.size();
    auto num_labels = count_unique_labels();
    std::cout << "Number of unique vertex label values: " << num_labels
              << std::endl;
    num_edges_ = el.size();
    if (!directed_)
      symmetrize_ = false; // no need to symmetrize undirected graph
    MakeGraphFromEL();
  }
  void read_adj(const char* filename) {
    FILE* fd = fopen(filename, "r");
    assert(fd != NULL);
    char buf[2048];
    unsigned size = 0, maxsize = 0;
    while (fgets(buf, 2048, fd) != NULL) {
      auto len = strlen(buf);
      size += len;
      if (buf[len - 1] == '\n') {
        maxsize = std::max(size, maxsize);
        size    = 0;
      }
    }
    fclose(fd);

    std::ifstream is;
    is.open(filename, std::ios::in);
    // char line[1024];
    char* line = new char[maxsize + 1];
    std::vector<std::string> result;
    while (is.getline(line, maxsize + 1)) {
      result.clear();
      split(line, result);
      IndexT src = atoi(result[0].c_str());
      labels_.resize(src + 1);
      labels_[src]  = atoi(result[1].c_str());
      ValueT elabel = 0;
      std::set<std::pair<IndexT, ValueT>> neighbors;
      for (size_t i = 2; i < result.size(); i++) {
        IndexT dst = atoi(result[i].c_str());
        if (src == dst)
          continue; // remove self-loop
        // elabel = atoi(result[i].c_str());
        neighbors.insert(
            std::pair<IndexT, ValueT>(dst, elabel)); // remove redundant edge
      }
      for (auto it = neighbors.begin(); it != neighbors.end(); ++it)
        el.push_back(MEdge(src, it->first, it->second));
    }
    is.close();
    num_vertices_   = labels_.size();
    auto num_labels = count_unique_labels();
    std::cout << "Number of unique vertex label values: " << num_labels
              << std::endl;
    num_edges_ = el.size();
    if (!directed_)
      symmetrize_ = false; // no need to symmetrize undirected graph
    MakeGraphFromEL();
  }
  void read_mtx(const char* filename, bool symmetrize = false) {
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
    } else if ((field == "real") || (field == "double") ||
               (field == "integer")) {
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
      if (c == '%') {
        in.ignore(200, '\n');
      } else {
        break;
      }
    }
    size_t m, n, nonzeros;
    in >> m >> n >> nonzeros >> std::ws;
    if (m != n) {
      std::cout << m << " " << n << " " << nonzeros << std::endl;
      std::cout << "matrix must be square for .mtx" << std::endl;
      std::exit(-26);
    }
    while (std::getline(in, line)) {
      std::istringstream edge_stream(line);
      IndexT u;
      edge_stream >> u;
      if (read_weights) {
        IndexT v;
        edge_stream >> v;
        el.push_back(MEdge(u - 1, v - 1, 1));
        if (symmetrize)
          el.push_back(MEdge(v - 1, u - 1, 1));
      } else {
        IndexT v;
        edge_stream >> v;
        el.push_back(MEdge(u - 1, v - 1, 1));
        if (symmetrize)
          el.push_back(MEdge(v - 1, u - 1, 1));
      }
    }
    in.close();
    labels_.resize(m);
    directed_ = !undirected;
    if (undirected)
      symmetrize_ = false; // no need to symmetrize undirected graph
    for (size_t i = 0; i < m; i++) {
      labels_[i] = rand() % 10 + 1;
    }
    num_vertices_ = m;
    num_edges_    = el.size();
    MakeGraphFromEL();
  }
  void read_gr(Graph& g) {
    num_vertices_ = g.size();
    for (auto it = g.begin(); it != g.end(); it++) {
      GNode src = *it;
      for (auto e : g.edges(src)) {
        GNode dst = g.getEdgeDst(e);
        el.push_back(MEdge(src, dst, 1));
      }
    }
    assert(el.size() == g.sizeEdges());
    num_edges_ = el.size();
    labels_.resize(num_vertices_);
    for (size_t i = 0; i < num_vertices_; i++) {
      labels_[i] = g.getData(i);
    }
    MakeGraphFromEL();
  }
  void print_graph() {
    if (directed_)
      std::cout << "directed graph\n";
    else
      std::cout << "undirected graph\n";
    for (size_t n = 0; n < num_vertices_; n++) {
      IndexT row_begin = rowptr_[n];
      IndexT row_end   = rowptr_[n + 1];
      std::cout << "vertex " << n << ": label = " << labels_[n]
                << " edgelist = [ ";
      for (IndexT offset = row_begin; offset < row_end; offset++) {
        IndexT dst = colidx_[offset];
        std::cout << dst << " ";
      }
      std::cout << "]" << std::endl;
    }
  }

private:
  MEdgeList el;
  bool need_dag;
  bool symmetrize_; // whether to symmetrize a directed graph
  bool directed_;
  size_t num_vertices_;
  size_t num_edges_;
  IndexT* rowptr_;
  IndexT* colidx_;
  ValueT* weight_;
  unsigned max_degree;
  std::vector<IndexT> degrees;
  std::vector<ValueT> labels_;
  std::vector<std::vector<MEdge>> vertices;

  unsigned count_unique_labels() {
    std::set<ValueT> s;
    unsigned res = 0;
    for (size_t i = 0; i < labels_.size(); i++) {
      if (s.find(labels_[i]) == s.end()) {
        s.insert(labels_[i]);
        res++;
      }
    }
    return res;
  }
  void CountDegrees(const MEdgeList& el) {
    degrees.resize(num_vertices_);
    std::fill(degrees.begin(), degrees.end(), 0);
    for (auto it = el.begin(); it < el.end(); it++) {
      MEdge e = *it;
      degrees[e.src]++;
      if (symmetrize_)
        degrees[e.dst]++;
    }
  }
  void MakeCSR(bool transpose) {
    degrees.resize(num_vertices_);
    std::fill(degrees.begin(), degrees.end(), 0);
    for (size_t i = 0; i < num_vertices_; i++)
      degrees[i] = vertices[i].size();
    max_degree = *(std::max_element(degrees.begin(), degrees.end()));

    std::vector<IndexT> offsets(degrees.size() + 1);
    IndexT total = 0;
    for (size_t n = 0; n < degrees.size(); n++) {
      offsets[n] = total;
      total += degrees[n];
    }
    offsets[degrees.size()] = total;

    assert(num_edges_ == offsets[num_vertices_]);
    weight_ = new ValueT[num_edges_];
    colidx_ = new IndexT[num_edges_];
    rowptr_ = new IndexT[num_vertices_ + 1];
    for (size_t i = 0; i < num_vertices_ + 1; i++)
      rowptr_[i] = offsets[i];
    for (size_t i = 0; i < num_vertices_; i++) {
      for (auto it = vertices[i].begin(); it < vertices[i].end(); it++) {
        MEdge e = *it;
        assert(i == e.src);
        if (symmetrize_ || (!symmetrize_ && !transpose)) {
          weight_[offsets[e.src]]   = e.elabel;
          colidx_[offsets[e.src]++] = e.dst;
        }
        if (symmetrize_ || (!symmetrize_ && transpose)) {
          weight_[offsets[e.dst]]   = e.elabel;
          colidx_[offsets[e.dst]++] = e.src;
        }
      }
    }
  }
  static bool compare_id(MEdge a, MEdge b) { return (a.dst < b.dst); }
  void SquishGraph(bool remove_selfloops  = true,
                   bool remove_redundents = true) {
    std::vector<MEdge> neighbors;
    for (size_t i = 0; i < num_vertices_; i++)
      vertices.push_back(neighbors);
    // assert(num_edges_ == el.size());
    for (size_t i = 0; i < num_edges_; i++)
      vertices[el[i].src].push_back(el[i]);
    el.clear();
    printf("Sorting the neighbor lists...");
    for (size_t i = 0; i < num_vertices_; i++)
      std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
    printf(" Done\n");
    // remove self loops
    int num_selfloops = 0;
    if (remove_selfloops) {
      printf("Removing self loops...");
      for (size_t i = 0; i < num_vertices_; i++) {
        for (unsigned j = 0; j < vertices[i].size(); j++) {
          if (i == vertices[i][j].dst) {
            vertices[i].erase(vertices[i].begin() + j);
            num_selfloops++;
            j--;
          }
        }
      }
      printf(" %d selfloops are removed\n", num_selfloops);
      num_edges_ -= num_selfloops;
    }
    // remove redundent
    int num_redundents = 0;
    if (remove_redundents) {
      printf("Removing redundent edges...");
      for (size_t i = 0; i < num_vertices_; i++) {
        for (unsigned j = 1; j < vertices[i].size(); j++) {
          if (vertices[i][j].dst == vertices[i][j - 1].dst) {
            vertices[i].erase(vertices[i].begin() + j);
            num_redundents++;
            j--;
          }
        }
      }
      printf(" %d redundent edges are removed\n", num_redundents);
      num_edges_ -= num_redundents;
    }
    if (need_dag) {
      int num_dag = 0;
      std::cout << "Constructing DAG...";
      degrees.resize(num_vertices_);
      for (size_t i = 0; i < num_vertices_; i++)
        degrees[i] = vertices[i].size();
      for (size_t i = 0; i < num_vertices_; i++) {
        for (unsigned j = 0; j < vertices[i].size(); j++) {
          IndexT to = vertices[i][j].dst;
          if (degrees[to] < degrees[i] ||
              (degrees[to] == degrees[i] && to < i)) {
            vertices[i].erase(vertices[i].begin() + j);
            num_dag++;
            j--;
          }
        }
      }
      printf(" %d dag edges are removed\n", num_dag);
      num_edges_ -= num_dag;
    }
  }
  void MakeGraphFromEL() {
    SquishGraph();
    MakeCSR(false);
  }
  inline void split(const std::string& str, std::vector<std::string>& tokens,
                    const std::string& delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos     = str.find_first_of(delimiters, lastPos);
    }
  }
};
