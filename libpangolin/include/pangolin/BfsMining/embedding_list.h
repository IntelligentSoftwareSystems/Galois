#ifndef EMBEDDING_LIST_H_
#define EMBEDDING_LIST_H_
#include "pangolin/gtypes.h"
#include "pangolin/base_embedding.h"
#include "pangolin/vertex_embedding.h"
#include "pangolin/edge_embedding.h"

// Embedding list: SoA structure
template <typename ElementType, typename EmbeddingType>
class EmbeddingList {
public:
  EmbeddingList() {}
  ~EmbeddingList() {}
  void init(PangolinGraph& graph, unsigned max_size = 2, bool is_dag = false);
  VertexId get_vid(unsigned level, size_t id) const {
    return vid_lists[level][id];
  }
  IndexTy get_idx(unsigned level, size_t id) const {
    return idx_lists[level][id];
  }
  BYTE get_his(unsigned level, size_t id) const { return his_lists[level][id]; }
  unsigned get_pid(size_t id) const { return pid_list[id]; }
  void set_vid(unsigned level, size_t id, VertexId vid) {
    vid_lists[level][id] = vid;
  }
  void set_idx(unsigned level, size_t id, IndexTy idx) {
    idx_lists[level][id] = idx;
  }
  void set_his(unsigned level, size_t id, BYTE lab) {
    his_lists[level][id] = lab;
  }
  void set_pid(size_t id, unsigned pid) { pid_list[id] = pid; }
  size_t size() const { return vid_lists[last_level].size(); }
  size_t size(unsigned level) const { return vid_lists[level].size(); }
  VertexList get_vid_list(unsigned level) { return vid_lists[level]; }
  UintList get_idx_list(unsigned level) { return idx_lists[level]; }
  ByteList get_his_list(unsigned level) { return his_lists[level]; }
  void remove_tail(size_t idx) {
    vid_lists[last_level].erase(vid_lists[last_level].begin() + idx,
                                vid_lists[last_level].end());
    if (std::is_same<ElementType, LabeledElement>::value)
      his_lists[last_level].erase(his_lists[last_level].begin() + idx,
                                  his_lists[last_level].end());
  }
  void add_level(Ulong size) {
    last_level++;
    assert(last_level < max_level);
    vid_lists[last_level].resize(size);
    idx_lists[last_level].resize(size);
    if (std::is_same<ElementType, LabeledElement>::value)
      his_lists[last_level].resize(size);
    if (std::is_same<EmbeddingType, VertexEmbedding>::value ||
        std::is_same<EmbeddingType, EdgeEmbedding>::value) // multi-pattern
      pid_list.resize(size);
  }
  void reset_level() {
    for (size_t i = 2; i <= last_level; i++) {
      vid_lists[i].clear();
      idx_lists[i].clear();
    }
    last_level = 1;
  }
  void printout_embeddings(int level, bool verbose = false) {
    std::cout << "Number of embeddings in level " << level << ": " << size()
              << std::endl;
    if (verbose) {
      for (size_t pos = 0; pos < size(); pos++) {
        EmbeddingType emb(last_level + 1);
        get_embedding(last_level, pos, emb);
        std::cout << emb << "\n";
      }
    }
  }
  void clean() {
    pid_list.clear();
    for (size_t i = 0; i < vid_lists.size(); i++) {
      if (std::is_same<ElementType, LabeledElement>::value)
        his_lists[i].clear();
      idx_lists[i].clear();
      vid_lists[i].clear();
    }
    his_lists.clear();
    idx_lists.clear();
    vid_lists.clear();
  }

private:
  UintList pid_list;
  ByteLists his_lists;
  IndexLists idx_lists;
  VertexLists vid_lists;
  unsigned last_level;
  unsigned max_level;
  void get_embedding(unsigned level, size_t pos, EmbeddingType& emb) {
    auto vid    = get_vid(level, pos);
    IndexTy idx = get_idx(level, pos);
    BYTE his    = 0;
    if (std::is_same<ElementType, LabeledElement>::value)
      his = get_his(level, pos);
    ElementType ele(vid, 0, 0, his);
    emb.set_element(level, ele);
    for (unsigned l = 1; l < level; l++) {
      vid = get_vid(level - l, idx);
      if (std::is_same<ElementType, LabeledElement>::value)
        his = get_his(level - l, idx);
      ElementType ele(vid, 0, 0, his);
      emb.set_element(level - l, ele);
      idx = get_idx(level - l, idx);
    }
    ElementType ele0(idx, 0, 0, 0);
    emb.set_element(0, ele0);
  }
};

#endif // EMBEDDING_LIST_HPP_
