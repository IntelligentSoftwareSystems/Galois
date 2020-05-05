#include "pangolin/base_embedding.h"

std::ostream& operator<<(std::ostream& strm, const BaseEmbedding& emb) {
  if (emb.empty()) {
    strm << "(empty)";
    return strm;
  }
  strm << "(";
  for (unsigned index = 0; index < emb.size() - 1; ++index)
    std::cout << emb.get_vertex(index) << ", ";
  std::cout << emb.get_vertex(emb.size() - 1);
  strm << ")";
  return strm;
}
