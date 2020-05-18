#include <iostream>
#include "galois/Bag.h"

// Embedding queue: AoS structure
// print out the embeddings in the task queue
template <typename EmbeddingTy>
class EmbeddingQueue : public galois::InsertBag<EmbeddingTy> {
public:
  void printout_embeddings(int level, bool verbose = false) {
    int num_embeddings = std::distance(this->begin(), this->end());
    std::cout << "Number of embeddings in level " << level << ": "
              << num_embeddings << std::endl;
    if (verbose)
      for (auto emb : *this)
        std::cout << emb << "\n";
  }
  void clean() {
    for (auto emb : *this)
      emb.clean();
    this->clear();
  }
};
