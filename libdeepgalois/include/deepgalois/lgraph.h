#ifndef __LGRAPH_HPP__
#define __LGRAPH_HPP__

// defines the Learning Graph (LGraph) data structure
#include <set>
#include <string>

namespace deepgalois {

typedef unsigned IndexT;
typedef float ValueT;

/**
 * Learning graph.
 *
 * Provides basic accesors and such; nothing special. Just a CSR.
 * Ultimatly becomes an LC_CSR.
 *
 * @todo remove this intermediate step if using edgelists
 */
class LGraph {
public:
  LGraph() : directed_(false) {}
  void clean() {
    delete[] rowptr_;
    delete[] colidx_;
  }
  bool directed() const { return directed_; }
  size_t num_vertices() const { return num_vertices_; }
  size_t num_edges() const { return num_edges_; }
  IndexT* out_rowptr() const { return rowptr_; }
  IndexT* out_colidx() const { return colidx_; }
  unsigned out_degree(IndexT n) const { return rowptr_[n + 1] - rowptr_[n]; }
  IndexT get_offset(IndexT n) { return rowptr_[n]; }
  IndexT get_dest(IndexT n) { return colidx_[n]; }

  void read_edgelist(const char* filename, bool symmetrize = false, bool add_self_loop = false) {
    std::ifstream in;
    std::string line;
    in.open(filename, std::ios::in);
    size_t m, n;
    in >> m >> n >> std::ws;
    num_vertices_ = m;
    num_edges_    = 0;
    std::cout << "num_vertices " << num_vertices_ << "\n";
    std::vector<std::set<IndexT> > vertices(m);
    for (size_t i = 0; i < n; i++) {
      std::set<IndexT> neighbors;
      if (add_self_loop) neighbors.insert(i);
      vertices.push_back(neighbors);
    }
    while (std::getline(in, line)) {
      std::istringstream edge_stream(line);
      IndexT u, v;
      edge_stream >> u;
      edge_stream >> v;
      vertices[u].insert(v);
      if (symmetrize) vertices[v].insert(u);
    }
    in.close();
	for (size_t i = 0; i < n; i++) num_edges_ += vertices[i].size();
	std::cout << "num_edges " << num_edges_ << "\n";
    MakeCSR(vertices);
  }

private:
  bool directed_;
  size_t num_vertices_;
  size_t num_edges_;
  IndexT* rowptr_;
  IndexT* colidx_;

  void MakeCSR(std::vector<std::set<IndexT> > vertices) {
    std::vector<IndexT> degrees;
    degrees.resize(num_vertices_);
    std::fill(degrees.begin(), degrees.end(), 0);
    for (size_t i = 0; i < num_vertices_; i++)
      degrees[i] = vertices[i].size();
    std::vector<IndexT> offsets(degrees.size() + 1);
    IndexT total = 0;
    for (size_t n = 0; n < degrees.size(); n++) {
      offsets[n] = total;
      total += degrees[n];
    }
    offsets[degrees.size()] = total;
    degrees.clear();
    assert(num_edges_ == offsets[num_vertices_]);
    colidx_ = new IndexT[num_edges_];
    rowptr_ = new IndexT[num_vertices_ + 1];
    for (size_t i = 0; i < num_vertices_ + 1; i++)
      rowptr_[i] = offsets[i];
    for (size_t i = 0; i < num_vertices_; i++) {
      for (auto dst : vertices[i])
          colidx_[offsets[i]++] = dst;
    }
  }
};

} // namespace
#endif
