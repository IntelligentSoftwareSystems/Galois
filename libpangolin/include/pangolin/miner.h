#ifndef MINER_HPP_
#define MINER_HPP_
#include "pangolin/scan.h"
#include "pangolin/util.h"
#include "pangolin/embedding_queue.h"
#include "bliss/uintseqhash.hh"
#define CHUNK_SIZE 1

template <typename ElementTy, typename EmbeddingTy, bool enable_dag>
class Miner {
  typedef EmbeddingQueue<EmbeddingTy> EmbeddingQueueTy;

public:
  Miner(unsigned max_sz, int nt) : max_size(max_sz), num_threads(nt) {
    // std::cout << "max_size = " << max_sz << std::endl;
    // std::cout << "num_threads = " << nt << std::endl;
  }
  virtual ~Miner() {}
  inline void insert(EmbeddingQueueTy& queue, bool debug = false);
  inline unsigned intersect(unsigned a, unsigned b) {
    return intersect_merge(a, b);
  }
  inline unsigned intersect_dag(unsigned a, unsigned b) {
    return intersect_dag_merge(a, b);
  }
  // unsigned read_graph(std::string filename);
  unsigned read_graph(std::string filetype, std::string filename) {
    max_degree = util::read_graph(graph, filetype, filename, enable_dag);
    graph.degree_counting();
    degrees = graph.degrees.data();
    // std::cout << "Input graph: num_vertices " << graph.size() << " num_edges
    // "
    //          << graph.sizeEdges() << "\n";
    // util::print_graph(graph);
    // convert_to_gbbs(filename);
    return max_degree;
  }
  void convert_to_gbbs(std::string filename) {
    printf("writing gbbs file\n");
    std::ofstream outfile;
    outfile.open(filename + ".gbbs");
    outfile << "AdjacencyGraph"
            << "\n";
    auto m   = graph.size();
    auto nnz = graph.sizeEdges();
    outfile << m << "\n";
    outfile << nnz << "\n";
    size_t offset = 0;
    for (size_t i = 0; i < m; i++) {
      outfile << offset << "\n";
      offset += graph.get_degree(i);
    }
    for (size_t i = 0; i < m; i++) {
      for (auto e : graph.edges(i)) {
        auto v = graph.getEdgeDst(e);
        outfile << v << "\n";
      }
    }
    outfile.close();
    exit(0);
  }
  unsigned read_pattern(std::string filename, std::string filetype = "gr",
                        bool symmetric = false) {
    unsigned max_deg = util::read_graph(pattern, filetype, filename, false);
    pattern.degree_counting();
    auto nv = pattern.size();
    auto ne = pattern.sizeEdges();
    std::cout << "Pattern graph: num_vertices " << nv << " num_edges " << ne
              << "\n";
    if (symmetric) {
      if (nv == 4) {
        if (ne == 12) {
          std::cout << "Input pattern: 4-clique, please use kcl\n";
          exit(1);
        } else if (ne == 10) {
          std::cout << "Input pattern: diamond\n";
          return 4;
        } else if (ne == 8) {
          if (max_deg == 3) {
            std::cout << "Input pattern: tailed-triangle\n";
            return 3;
          } else {
            std::cout << "Input pattern: 4-cycle\n";
            return 2;
          }
        } else if (ne == 6) {
          if (max_deg == 3) {
            std::cout << "Input pattern: 3-star\n";
            return 1;
          } else {
            std::cout << "Input pattern: 4-path\n";
            return 0;
          }
        } else {
          std::cout << "Error: the number of edges is invalid\n";
          exit(1);
        }
      } else if (nv == 5) {
        std::cout << "5-motif currently not supported\n";
        exit(1);
      } else {
        std::cout << "pattern size currently not supported\n";
        exit(1);
      }
    } else {
      if (nv == 4) {
        if (ne == 6) {
          std::cout << "Input pattern: 4-clique, please use kcl\n";
          exit(1);
        } else if (ne == 5) {
          std::cout << "Input pattern: diamond\n";
          return 4;
        } else if (ne == 4) {
          if (max_deg == 2) {
            std::cout << "Input pattern: tailed-triangle\n";
            return 3;
          } else {
            // assert(max_deg == 1);
            std::cout << "Input pattern: 4-cycle\n";
            return 2;
          }
        } else if (ne == 3) {
          if (max_deg == 3) {
            std::cout << "Input pattern: 3-star\n";
            return 1;
          } else {
            // assert(max_deg == 2);
            std::cout << "Input pattern: 4-path\n";
            return 0;
          }
        } else {
          std::cout << "Error: the unmber of edges is invalid\n";
          exit(1);
        }
      } else if (nv == 5) {
      } else {
        std::cout << "pattern size currently not supported\n";
        exit(1);
      }
    }
    return 0;
  }

protected:
  PangolinGraph graph;
  PangolinGraph pattern;
  unsigned max_size;
  int num_threads;
  unsigned max_degree;
  uint32_t* degrees;

  inline bool is_automorphism_dag(unsigned n, const EmbeddingTy& emb,
                                  unsigned idx, VertexId dst) {
    // if (dst <= emb.get_vertex(0)) return true;
    for (unsigned i = 0; i < n; ++i)
      if (dst == emb.get_vertex(i))
        return true;
    for (unsigned i = 0; i < idx; ++i)
      if (is_connected_dag(dst, emb.get_vertex(i)))
        return true;
    // for (unsigned i = idx+1; i < n; ++i) if (dst < emb.get_vertex(i)) return
    // true;
    return false;
  }
  inline bool is_vertexInduced_automorphism(unsigned n, const EmbeddingTy& emb,
                                            unsigned idx, VertexId dst) {
    // unsigned n = emb.size();
    // the new vertex id should be larger than the first vertex id
    if (dst <= emb.get_vertex(0))
      return true;
    // the new vertex should not already exist in the embedding
    for (unsigned i = 1; i < n; ++i)
      if (dst == emb.get_vertex(i))
        return true;
    // the new vertex should not already be extended by any previous vertex in
    // the embedding
    for (unsigned i = 0; i < idx; ++i)
      if (is_connected(emb.get_vertex(i), dst))
        return true;
    // the new vertex id should be larger than any vertex id after its source
    // vertex in the embedding
    for (unsigned i = idx + 1; i < n; ++i)
      if (dst < emb.get_vertex(i))
        return true;
    return false;
  }
  unsigned get_degree(PangolinGraph* g, VertexId vid) {
    return std::distance(g->edge_begin(vid), g->edge_end(vid));
  }
  inline unsigned intersect_merge(unsigned src, unsigned dst) {
    unsigned count = 0;
    for (auto e : graph.edges(dst)) {
      GNode dst_dst = graph.getEdgeDst(e);
      for (auto e1 : graph.edges(src)) {
        GNode to = graph.getEdgeDst(e1);
        if (dst_dst == to) {
          count += 1;
          break;
        }
        if (to > dst_dst)
          break;
      }
    }
    return count;
  }
  inline unsigned intersect_dag_merge(unsigned p, unsigned q) {
    unsigned count = 0;
    auto p_start   = graph.edge_begin(p);
    auto p_end     = graph.edge_end(p);
    auto q_start   = graph.edge_begin(q);
    auto q_end     = graph.edge_end(q);
    auto p_it      = p_start;
    auto q_it      = q_start;
    int a;
    int b;
    while (p_it < p_end && q_it < q_end) {
      a     = graph.getEdgeDst(p_it);
      b     = graph.getEdgeDst(q_it);
      int d = a - b;
      if (d <= 0)
        p_it++;
      if (d >= 0)
        q_it++;
      if (d == 0)
        count++;
    }
    return count;
  }
  inline unsigned intersect_search(unsigned a, unsigned b) {
    if (degrees[a] == 0 || degrees[b] == 0)
      return 0;
    unsigned count  = 0;
    unsigned lookup = a;
    unsigned search = b;
    if (degrees[a] > degrees[b]) {
      lookup = b;
      search = a;
    }
    auto begin = graph.edge_begin(search);
    auto end   = graph.edge_end(search);
    for (auto e : graph.edges(lookup)) {
      GNode key = graph.getEdgeDst(e);
      if (binary_search(key, begin, end))
        count++;
    }
    return count;
  }
  inline bool is_all_connected_except(unsigned dst, unsigned pos,
                                      const EmbeddingTy& emb) {
    unsigned n         = emb.size();
    bool all_connected = true;
    for (unsigned i = 0; i < n; ++i) {
      if (i == pos)
        continue;
      unsigned from = emb.get_vertex(i);
      if (!is_connected(from, dst)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  inline bool is_all_connected_except_dag(unsigned dst, unsigned pos,
                                          const EmbeddingTy& emb) {
    unsigned n         = emb.size();
    bool all_connected = true;
    for (unsigned i = 0; i < n; ++i) {
      if (i == pos)
        continue;
      unsigned from = emb.get_vertex(i);
      if (!is_connected_dag(dst, from)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  inline bool is_all_connected(unsigned dst, const EmbeddingTy& emb,
                               unsigned end, unsigned start = 0) {
    assert(start >= 0 && end > 0);
    bool all_connected = true;
    for (unsigned i = start; i < end; ++i) {
      unsigned from = emb.get_vertex(i);
      if (!is_connected(from, dst)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  inline bool is_all_connected_dag(unsigned dst, const EmbeddingTy& emb,
                                   unsigned end, unsigned start = 0) {
    assert(start >= 0 && end > 0);
    bool all_connected = true;
    for (unsigned i = start; i < end; ++i) {
      unsigned from = emb.get_vertex(i);
      if (!is_connected_dag(dst, from)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  inline bool is_all_connected_dag(unsigned dst,
                                   const std::vector<VertexId>& emb,
                                   unsigned end, unsigned start = 0) {
    assert(start >= 0 && end > 0);
    bool all_connected = true;
    for (unsigned i = start; i < end; ++i) {
      unsigned from = emb[i];
      if (!is_connected_dag(dst, from)) {
        all_connected = false;
        break;
      }
    }
    return all_connected;
  }
  // check if vertex a is connected to vertex b in a undirected graph
  inline bool is_connected(unsigned a, unsigned b) {
    if (degrees[a] == 0 || degrees[b] == 0)
      return false;
    unsigned key    = a;
    unsigned search = b;
    if (degrees[a] < degrees[b]) {
      key    = b;
      search = a;
    }
    auto begin = graph.edge_begin(search);
    auto end   = graph.edge_end(search);
    // return serial_search(key, begin, end);
    return binary_search(key, begin, end);
  }
  inline int is_connected_dag(unsigned key, unsigned search) {
    if (degrees[search] == 0)
      return false;
    auto begin = graph.edge_begin(search);
    auto end   = graph.edge_end(search);
    // return serial_search(key, begin, end);
    return binary_search(key, begin, end);
  }
  inline bool serial_search(unsigned key, PangolinGraph::edge_iterator begin,
                            PangolinGraph::edge_iterator end) {
    for (auto offset = begin; offset != end; ++offset) {
      unsigned d = graph.getEdgeDst(offset);
      if (d == key)
        return true;
      if (d > key)
        return false;
    }
    return false;
  }
  inline bool binary_search(unsigned key, PangolinGraph::edge_iterator begin,
                            PangolinGraph::edge_iterator end) {
    auto l = begin;
    auto r = end - 1;
    while (r >= l) {
      auto mid       = l + (r - l) / 2;
      unsigned value = graph.getEdgeDst(mid);
      if (value == key)
        return true;
      if (value < key)
        l = mid + 1;
      else
        r = mid - 1;
    }
    return false;
  }
  inline int binary_search(unsigned key, PangolinGraph::edge_iterator begin,
                           int length) {
    if (length < 1)
      return -1;
    int l = 0;
    int r = length - 1;
    while (r >= l) {
      int mid        = l + (r - l) / 2;
      unsigned value = graph.getEdgeDst(begin + mid);
      if (value == key)
        return mid;
      if (value < key)
        l = mid + 1;
      else
        r = mid - 1;
    }
    return -1;
  }
  inline void gen_adj_matrix(unsigned n, const std::vector<bool>& connected,
                             Matrix& a) {
    unsigned l = 0;
    for (unsigned i = 1; i < n; i++)
      for (unsigned j = 0; j < i; j++)
        if (connected[l++])
          a[i][j] = a[j][i] = 1;
  }
  // calculate the trace of a given n*n matrix
  inline MatType trace(unsigned n, Matrix matrix) {
    MatType tr = 0;
    for (unsigned i = 0; i < n; i++) {
      tr += matrix[i][i];
    }
    return tr;
  }
  // matrix mutiplication, both a and b are n*n matrices
  inline Matrix product(unsigned n, const Matrix& a, const Matrix& b) {
    Matrix c(n, std::vector<MatType>(n));
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        c[i][j] = 0;
        for (unsigned k = 0; k < n; ++k) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return c;
  }
  // calculate the characteristic polynomial of a n*n matrix A
  inline void char_polynomial(unsigned n, Matrix& A, std::vector<MatType>& c) {
    // n is the size (num_vertices) of a graph
    // A is the adjacency matrix (n*n) of the graph
    Matrix C;
    C = A;
    for (unsigned i = 1; i <= n; i++) {
      if (i > 1) {
        for (unsigned j = 0; j < n; j++)
          C[j][j] += c[n - i + 1];
        C = product(n, A, C);
      }
      c[n - i] -= trace(n, C) / i;
    }
  }
  inline void get_connectivity(unsigned n, unsigned idx, VertexId dst,
                               const EmbeddingTy& emb,
                               std::vector<bool>& connected) {
    connected.push_back(true); // 0 and 1 are connected
    for (unsigned i = 2; i < n; i++)
      for (unsigned j = 0; j < i; j++)
        if (is_connected(emb.get_vertex(i), emb.get_vertex(j)))
          connected.push_back(true);
        else
          connected.push_back(false);
    for (unsigned j = 0; j < n; j++) {
      if (j == idx)
        connected.push_back(true);
      else if (is_connected(emb.get_vertex(j), dst))
        connected.push_back(true);
      else
        connected.push_back(false);
    }
  }
  // eigenvalue based approach to find the pattern id for a given embedding
  inline unsigned find_motif_pattern_id_eigen(unsigned n, unsigned idx,
                                              VertexId dst,
                                              const EmbeddingTy& emb) {
    std::vector<bool> connected;
    get_connectivity(n, idx, dst, emb, connected);
    Matrix A(n + 1, std::vector<MatType>(n + 1, 0));
    gen_adj_matrix(n + 1, connected, A);
    std::vector<MatType> c(n + 1, 0);
    char_polynomial(n + 1, A, c);
    bliss::UintSeqHash h;
    for (unsigned i = 0; i < n + 1; ++i)
      h.update((unsigned)c[i]);
    return h.get_value();
  }

  // unsigned orientation(PangolinGraph &og, PangolinGraph &g);
};

#endif // MINER_HPP_
