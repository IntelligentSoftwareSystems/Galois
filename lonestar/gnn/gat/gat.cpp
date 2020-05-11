// Graph Attension Networks (GAT)
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"

const char* name = "Graph Attention Networks (GAT)";
const char* desc = "Graph Attention Networks on an undirected graph: <https://arxiv.org/pdf/1710.10903.pdf>";
const char* url  = 0;

// math: h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
// where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and node :math:`j`:
// .. math:: \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
//                e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)
/*
namespace deepgalois {
 
// define aggregator here
class AppAggregator: public Aggregator {
public:
  emb_t applyEdge(VertexID, VertexID u, emb_t in) {
    auto ilen = get_in_feat_len();
    return &in[ilen*u];
  }

  emb_t applyVertex(VertexID v, emb_t in, emb_t accum) {
    auto n = get_num_samples();
    auto ilen = get_in_feat_len();
    auto olen = get_out_feat_len();
    emb_t a, b, c;
  }
};

}
//*/
#include "engine.h"
