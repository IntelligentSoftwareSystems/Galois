#ifndef VERTEX_MINER_H
#define VERTEX_MINER_H
#include "pangolin/miner.h"
#include "pangolin/ptypes.h"
#include "pangolin/quick_pattern.h"
#include "pangolin/canonical_graph.h"
#include "pangolin/BfsMining/embedding_list.h"

template <typename ElementTy, typename EmbeddingTy, typename API,
          bool enable_dag = false, bool is_single = true,
          bool use_wedge = false, bool use_match_order = false>
class VertexMiner : public Miner<ElementTy, EmbeddingTy, enable_dag> {
  typedef EmbeddingList<ElementTy, EmbeddingTy> EmbeddingListTy;

public:
  VertexMiner(unsigned max_sz, int nt, unsigned nb)
      : Miner<ElementTy, EmbeddingTy, enable_dag>(max_sz, nt), num_blocks(nb) {}
  virtual ~VertexMiner() {}
  void init_emb_list() {
    this->emb_list.init(this->graph, this->max_size, enable_dag);
  }
  bool is_single_pattern() { return npatterns == 1; }
  int get_num_patterns() { return npatterns; }
  void set_num_patterns(int np = 1) {
    npatterns = np;
    accumulators.resize(npatterns);
    for (int i = 0; i < npatterns; i++)
      accumulators[i].reset();
    if (!is_single)
      for (auto i = 0; i < this->num_threads; i++)
        qp_localmaps.getLocal(i)->clear();
  }
  void clean() {
    is_wedge.clear();
    accumulators.clear();
    qp_map.clear();
    cg_map.clear();
    for (auto i = 0; i < this->num_threads; i++)
      qp_localmaps.getLocal(i)->clear();
    for (auto i = 0; i < this->num_threads; i++)
      cg_localmaps.getLocal(i)->clear();
    this->emb_list.clean();
  }
  void initialize(std::string pattern_filename) {
    galois::on_each([&](unsigned tid, unsigned) {
      auto& local_counters = *(counters.getLocal(tid));
      local_counters.resize(npatterns);
      std::fill(local_counters.begin(), local_counters.end(), 0);
    });
    init_emb_list();
    if (use_match_order) {
      if (pattern_filename == "") {
        std::cout << "need specify pattern file name using -p\n";
        exit(1);
      }
      // unsigned pid = this->read_pattern(pattern_filename);
      // unsigned pid = this->read_pattern(pattern_filename, "gr", true);
      // std::cout << "pattern id = " << pid << "\n";
      // set_input_pattern(pid);
    }
  }
  void set_input_pattern(unsigned GALOIS_UNUSED(pid)) {
    // input_pid = pid;
  }
  virtual void print_output() {}

  // extension for vertex-induced motif
  inline void extend_vertex_multi(unsigned level, size_t chunk_begin,
                                  size_t chunk_end) {
    auto cur_size = this->emb_list.size();
    size_t begin = 0, end = cur_size;
    if (level == 1) {
      begin    = chunk_begin;
      end      = chunk_end;
      cur_size = end - begin;
      // std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end "
      //          << chunk_end << "\n";
    }
    // std::cout << "\t number of current embeddings in level " << level << ": "
    // << cur_size << "\n";
    UintList num_new_emb(cur_size); // TODO: for large graph, wo need UlongList
    // UlongList num_new_emb(cur_size);
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          auto& local_counters  = *(counters.getLocal());
          unsigned n            = level + 1;
          StrQpMapFreq* qp_lmap = nullptr;
          if (n >= 4)
            qp_lmap = qp_localmaps.getLocal();
          EmbeddingTy emb(n);
          get_embedding(level, pos, emb);
          if (n < this->max_size - 1)
            num_new_emb[pos - begin] = 0;
          if (n == 3 && this->max_size == 4)
            emb.set_pid(this->emb_list.get_pid(pos));
          for (unsigned i = 0; i < n; ++i) {
            if (!API::toExtend(n, emb, i))
              continue;
            auto src = emb.get_vertex(i);
            for (auto e : this->graph.edges(src)) {
              auto dst = this->graph.getEdgeDst(e);
              if (API::toAdd(n, this->graph, emb, i, dst)) {
                if (n < this->max_size - 1) {
                  num_new_emb[pos - begin]++;
                } else { // do reduction
                  if (n < 4) {
                    unsigned pid =
                        this->find_motif_pattern_id(n, i, dst, emb, pos);
                    local_counters[pid] += 1;
                  } else
                    quick_reduce(n, i, dst, emb, qp_lmap);
                }
              }
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-alloc"));
    if (level == this->max_size - 2) {
      galois::on_each([&](unsigned tid, unsigned) {
        auto& local_counters = *(counters.getLocal(tid));
        for (int i = 0; i < this->npatterns; i++)
          this->accumulators[i] += local_counters[i];
      });
      return;
    }

    UlongList indices = parallel_prefix_sum<unsigned, Ulong>(num_new_emb);
    num_new_emb.clear();
    Ulong new_size = indices.back();
    this->emb_list.add_level(new_size);
    if (use_wedge && level == 1 && this->max_size == 4) {
      is_wedge.resize(this->emb_list.size());
      std::fill(is_wedge.begin(), is_wedge.end(), 0);
    }
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          auto start = indices[pos - begin];
          auto n     = emb.size();
          for (unsigned i = 0; i < n; ++i) {
            if (!API::toExtend(n, emb, i))
              continue;
            auto src = emb.get_vertex(i);
            for (auto e : this->graph.edges(src)) {
              GNode dst = this->graph.getEdgeDst(e);
              if (API::toAdd(n, this->graph, emb, i, dst)) {
                if (!is_single && n == 2 && this->max_size == 4)
                  this->emb_list.set_pid(start, this->find_motif_pattern_id(
                                                    n, i, dst, emb, start));
                this->emb_list.set_idx(level + 1, start, pos);
                this->emb_list.set_vid(level + 1, start++, dst);
              }
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-insert"));
    indices.clear();
  }

  // extension for vertex-induced clique
  inline void extend_vertex_single(unsigned level, size_t chunk_begin,
                                   size_t chunk_end) {
    auto cur_size = this->emb_list.size();
    size_t begin = 0, end = cur_size;
    if (level == 1) {
      begin    = chunk_begin;
      end      = chunk_end;
      cur_size = end - begin;
      // std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end "
      //          << chunk_end << "\n";
    }
    // std::cout << "\t number of current embeddings in level " << level << ": "
    //          << cur_size << "\n";
    UintList num_new_emb(cur_size);
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          auto& local_counters = *(counters.getLocal());
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          auto vid                 = this->emb_list.get_vid(level, pos);
          num_new_emb[pos - begin] = 0;
          for (auto e : this->graph.edges(vid)) {
            GNode dst = this->graph.getEdgeDst(e);
            if (API::toAdd(level + 1, this->graph, emb, level, dst)) {
              if (level < this->max_size - 2) {
                num_new_emb[pos - begin]++;
              } else {
                local_counters[0] += 1;
              }
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-alloc"));

    if (level == this->max_size - 2) {
      galois::on_each([&](unsigned tid, unsigned) {
        auto& local_counters = *(counters.getLocal(tid));
        for (int i = 0; i < this->npatterns; i++)
          this->accumulators[0] += local_counters[0];
      });
      return;
    }

    UlongList indices = parallel_prefix_sum<unsigned, Ulong>(num_new_emb);
    num_new_emb.clear();
    Ulong new_size = indices.back();
    std::cout << "\t number of new embeddings: " << new_size << "\n";
    this->emb_list.add_level(new_size);
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          auto vid   = this->emb_list.get_vid(level, pos);
          auto start = indices[pos - begin];
          for (auto e : this->graph.edges(vid)) {
            GNode dst = this->graph.getEdgeDst(e);
            if (API::toAdd(level + 1, this->graph, emb, level, dst)) {
              this->emb_list.set_idx(level + 1, start, pos);
              this->emb_list.set_vid(level + 1, start++, dst);
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-insert"));
    indices.clear();
  }

  inline void extend_single_ordered(unsigned level, size_t chunk_begin,
                                    size_t chunk_end) {
    auto cur_size = this->emb_list.size();
    size_t begin = 0, end = cur_size;
    if (level == 1) {
      begin    = chunk_begin;
      end      = chunk_end;
      cur_size = end - begin;
      // std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end " <<
      // chunk_end << "\n";
    }
    // std::cout << "\t number of embeddings in level " << level << ": " <<
    // cur_size << "\n";
    UintList num_new_emb(cur_size);

    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          auto& local_counters = *(counters.getLocal());
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          num_new_emb[pos - begin] = 0;
          auto id                  = API::getExtendableVertex(level + 1);
          auto src                 = emb.get_vertex(id);
          for (auto e : this->graph.edges(src)) {
            auto dst = this->graph.getEdgeDst(e);
            if (API::toAdd(level + 1, this->graph, emb, src, dst)) {
              if (level < this->max_size - 2) {
                num_new_emb[pos - begin]++;
              } else {
                local_counters[0] += 1;
              }
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-alloc"));

    if (level == this->max_size - 2) {
      galois::on_each([&](unsigned tid, unsigned) {
        auto& local_counters = *(counters.getLocal(tid));
        for (int i = 0; i < this->npatterns; i++)
          this->accumulators[0] += local_counters[0];
      });
      return;
    }

    UlongList indices = parallel_prefix_sum<unsigned, Ulong>(num_new_emb);
    num_new_emb.clear();
    Ulong new_size = indices.back();
    // std::cout << "number of new embeddings: " << new_size << "\n";
    this->emb_list.add_level(new_size);

    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          auto start = indices[pos - begin];
          auto id    = API::getExtendableVertex(level + 1);
          auto src   = emb.get_vertex(id);
          // std::cout << "current embedding: " << emb << "\n";
          // std::cout << "extending vertex " << src << "\n";
          for (auto e : this->graph.edges(src)) {
            auto dst = this->graph.getEdgeDst(e);
            if (API::toAdd(level + 1, this->graph, emb, src, dst)) {
              this->emb_list.set_idx(level + 1, start, pos);
              this->emb_list.set_vid(level + 1, start++, dst);
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-insert"));
    indices.clear();
  }

  inline void extend_ordered(unsigned level, size_t chunk_begin,
                             size_t chunk_end) {
    auto cur_size = this->emb_list.size();
    size_t begin = 0, end = cur_size;
    if (level == 1) {
      begin    = chunk_begin;
      end      = chunk_end;
      cur_size = end - begin;
      // std::cout << "\t chunk_begin = " << chunk_begin << ", chunk_end "
      //          << chunk_end << "\n";
    }
    // std::cout << "\t number of current embeddings in level " << level << ": "
    //          << cur_size << "\n";
    UintList num_new_emb(cur_size);
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          num_new_emb[pos - begin] = 0;
          // std::cout << "current embedding: " << emb << "\n";
          for (auto q_edge : this->pattern.edges(level + 1)) {
            VertexId q_dst   = this->pattern.getEdgeDst(q_edge);
            VertexId q_order = q_dst;
            if (q_order < level + 1) {
              VertexId d_vertex = emb.get_vertex(q_order);
              for (auto d_edge : this->graph.edges(d_vertex)) {
                GNode d_dst = this->graph.getEdgeDst(d_edge);
                if (API::toAddOrdered(level + 1, this->graph, emb, q_order,
                                      d_dst, this->pattern)) {
                  if (level < this->max_size - 2)
                    num_new_emb[pos - begin]++;
                  else
                    accumulators[0] += 1;
                }
              }
              break;
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-alloc"));

    if (level == this->max_size - 2)
      return;
    UlongList indices = parallel_prefix_sum<unsigned, Ulong>(num_new_emb);
    num_new_emb.clear();
    Ulong new_size = indices.back();
    std::cout << "\t number of new embeddings: " << new_size << "\n";
    this->emb_list.add_level(new_size);
    galois::do_all(
        galois::iterate(begin, end),
        [&](const size_t& pos) {
          EmbeddingTy emb(level + 1);
          get_embedding(level, pos, emb);
          auto start = indices[pos - begin];
          for (auto q_edge : this->pattern.edges(level + 1)) {
            VertexId q_dst   = this->pattern.getEdgeDst(q_edge);
            VertexId q_order = q_dst;
            if (q_order < level + 1) {
              VertexId d_vertex = emb.get_vertex(q_order);
              for (auto d_edge : this->graph.edges(d_vertex)) {
                GNode d_dst = this->graph.getEdgeDst(d_edge);
                if (API::toAddOrdered(level + 1, this->graph, emb, q_order,
                                      d_dst, this->pattern)) {
                  this->emb_list.set_idx(level + 1, start, pos);
                  this->emb_list.set_vid(level + 1, start++, d_dst);
                }
              }
              break;
            }
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("Extending-insert"));
    indices.clear();
  }

  // quick pattern reduction
  inline void quick_reduce(unsigned n, unsigned i, VertexId dst,
                           const EmbeddingTy& emb, StrQpMapFreq* qp_lmap) {
    std::vector<bool> connected;
    this->get_connectivity(n, i, dst, emb, connected);
    StrQPattern qp(n + 1, connected);
    if (qp_lmap->find(qp) != qp_lmap->end()) {
      (*qp_lmap)[qp] += 1;
      qp.clean();
    } else
      (*qp_lmap)[qp] = 1;
  }
  // canonical pattern reduction
  inline void canonical_reduce() {
    for (auto i = 0; i < this->num_threads; i++)
      cg_localmaps.getLocal(i)->clear();
    galois::do_all(
        galois::iterate(qp_map),
        [&](auto& element) {
          StrCgMapFreq* cg_map = cg_localmaps.getLocal();
          StrCPattern cg(element.first);
          if (cg_map->find(cg) != cg_map->end())
            (*cg_map)[cg] += element.second;
          else
            (*cg_map)[cg] = element.second;
          cg.clean();
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::loopname("CanonicalReduce"));
    qp_map.clear();
  }
  inline void merge_qp_map() {
    qp_map.clear();
    for (unsigned i = 0; i < qp_localmaps.size(); i++) {
      StrQpMapFreq qp_lmap = *qp_localmaps.getLocal(i);
      for (auto element : qp_lmap) {
        if (qp_map.find(element.first) != qp_map.end())
          qp_map[element.first] += element.second;
        else
          qp_map[element.first] = element.second;
      }
    }
  }
  inline void merge_cg_map() {
    cg_map.clear();
    for (unsigned i = 0; i < cg_localmaps.size(); i++) {
      StrCgMapFreq cg_lmap = *cg_localmaps.getLocal(i);
      for (auto element : cg_lmap) {
        if (cg_map.find(element.first) != cg_map.end())
          cg_map[element.first] += element.second;
        else
          cg_map[element.first] = element.second;
      }
    }
  }

  // Utilities
  Ulong get_total_count() { return accumulators[0].reduce(); }
  void printout_motifs() {
    std::cout << std::endl;
    if (accumulators.size() == 2) {
      std::cout << "\ttriangles " << accumulators[0].reduce() << std::endl;
      std::cout << "\twedges    " << accumulators[1].reduce() << std::endl;
    } else if (accumulators.size() == 6) {
      std::cout << "\t4-paths --> " << accumulators[0].reduce() << std::endl;
      std::cout << "\t3-stars --> " << accumulators[1].reduce() << std::endl;
      std::cout << "\t4-cycles --> " << accumulators[2].reduce() << std::endl;
      std::cout << "\ttailed-triangles --> " << accumulators[3].reduce()
                << std::endl;
      std::cout << "\tdiamonds --> " << accumulators[4].reduce() << std::endl;
      std::cout << "\t4-cliques --> " << accumulators[5].reduce() << std::endl;
    } else {
      if (this->max_size < 9) {
        std::cout << std::endl;
        for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
          std::cout << "{" << it->first << "} --> " << it->second << std::endl;
      } else {
        std::cout << std::endl;
        for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
          std::cout << it->first << " --> " << it->second << std::endl;
      }
    }
    // std::cout << std::endl;
  }
  void tc_vertex_solver() { // vertex parallel
    galois::do_all(
        galois::iterate(this->graph.begin(), this->graph.end()),
        [&](const GNode& src) {
          for (auto e : this->graph.edges(src)) {
            auto dst = this->graph.getEdgeDst(e);
            accumulators[0] += this->intersect(src, dst);
          }
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("TC"));
  }

  void tc_solver() { // edge parallel
    galois::do_all(
        galois::iterate((size_t)0, this->emb_list.size()),
        [&](const size_t& id) {
          auto src = this->emb_list.get_idx(1, id);
          auto dst = this->emb_list.get_vid(1, id);
          auto num = this->intersect_dag(src, dst);
          accumulators[0] += num;
        },
        galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
        galois::loopname("TC"));
  }

  void solver() {
    size_t num          = this->emb_list.size();
    size_t chunk_length = (num - 1) / num_blocks + 1;
    // std::cout << "number of single-edge embeddings: " << num << "\n";
    for (size_t cid = 0; cid < num_blocks; cid++) {
      size_t chunk_begin = cid * chunk_length;
      size_t chunk_end   = std::min((cid + 1) * chunk_length, num);
      // size_t cur_size    = chunk_end - chunk_begin;
      // std::cout << "Processing the " << cid << " chunk (" << cur_size
      //          << " edges) of " << num_blocks << " blocks\n";
      unsigned level = 1;
      while (1) {
        // this->emb_list.printout_embeddings(level);
        if (use_match_order) {
          extend_single_ordered(level, chunk_begin, chunk_end);
        } else {
          if (is_single_pattern())
            extend_vertex_single(level, chunk_begin, chunk_end);
          else
            extend_vertex_multi(level, chunk_begin, chunk_end);
        }
        if (level == this->max_size - 2)
          break;
        level++;
      }
      this->emb_list.reset_level();
    }
    if (this->max_size >= 5 && !is_single_pattern()) {
      merge_qp_map();
      canonical_reduce();
      merge_cg_map();
    }
  }

private:
  unsigned num_blocks;
  StrQpMapFreq qp_map; // quick patterns map for counting the frequency
  StrCgMapFreq cg_map; // canonical graph map for couting the frequency
  LocalStrQpMapFreq qp_localmaps; // quick patterns local map for each thread
  LocalStrCgMapFreq cg_localmaps; // canonical graph local map for each thread
  std::vector<BYTE> is_wedge;     // indicate a 3-vertex embedding is a wedge or
                                  // chain (v0-cntered or v1-centered)

  inline void get_embedding(unsigned level, size_t pos, EmbeddingTy& emb) {
    auto vid = this->emb_list.get_vid(level, pos);
    auto idx = this->emb_list.get_idx(level, pos);
    ElementTy ele(vid);
    emb.set_element(level, ele);
    // backward constructing the embedding
    for (unsigned l = 1; l < level; l++) {
      auto u = this->emb_list.get_vid(level - l, idx);
      ElementTy ele(u);
      emb.set_element(level - l, ele);
      idx = this->emb_list.get_idx(level - l, idx);
    }
    ElementTy ele0(idx);
    emb.set_element(0, ele0);
  }

protected:
  int npatterns;
  galois::substrate::PerThreadStorage<std::vector<Ulong>> counters;
  std::vector<UlongAccu> accumulators;
  EmbeddingListTy emb_list;

  inline unsigned find_motif_pattern_id(unsigned n, unsigned idx, VertexId dst,
                                        const EmbeddingTy& emb,
                                        unsigned pos = 0) {
    unsigned pid = 0;
    if (n == 2) { // count 3-motifs
      pid = 1;    // 3-chain
      if (idx == 0) {
        if (this->is_connected(emb.get_vertex(1), dst))
          pid = 0; // triangle
        else if (use_wedge && this->max_size == 4)
          is_wedge[pos] = 1; // wedge; used for 4-motif
      }
    } else if (n == 3) { // count 4-motifs
      unsigned num_edges = 1;
      pid                = emb.get_pid();
      if (pid == 0) { // extending a triangle
        for (unsigned j = idx + 1; j < n; j++)
          if (this->is_connected(emb.get_vertex(j), dst))
            num_edges++;
        pid = num_edges + 2; // p3: tailed-triangle; p4: diamond; p5: 4-clique
      } else {               // extending a 3-chain
        std::vector<bool> connected(3, false);
        connected[idx] = true;
        for (unsigned j = idx + 1; j < n; j++) {
          if (this->is_connected(emb.get_vertex(j), dst)) {
            num_edges++;
            connected[j] = true;
          }
        }
        if (num_edges == 1) {
          pid             = 0; // p0: 3-path
          unsigned center = 1;
          if (use_wedge) {
            if (is_wedge[pos])
              center = 0;
          } else
            center = this->is_connected(emb.get_vertex(1), emb.get_vertex(2))
                         ? 1
                         : 0;
          if (idx == center)
            pid = 1; // p1: 3-star
        } else if (num_edges == 2) {
          pid             = 2; // p2: 4-cycle
          unsigned center = 1;
          if (use_wedge) {
            if (is_wedge[pos])
              center = 0;
          } else
            center = this->is_connected(emb.get_vertex(1), emb.get_vertex(2))
                         ? 1
                         : 0;
          if (connected[center])
            pid = 3; // p3: tailed-triangle
        } else {
          pid = 4; // p4: diamond
        }
      }
    } else { // count 5-motif and beyond
      pid = this->find_motif_pattern_id_eigen(n, idx, dst, emb);
    }
    return pid;
  }
};

#endif // VERTEX_MINER_HPP_
