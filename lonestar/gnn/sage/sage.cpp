// GraphSAGE: <https://arxiv.org/pdf/1706.02216.pdf>
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"

const char* name = "GraphSAGE";
const char* desc = "GraphSAGE on an undirected graph: <https://arxiv.org/pdf/1706.02216.pdf>";
const char* url  = 0;

// define aggregator here
// .. math::
//      h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
//      \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)
//
//      h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
//      (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)
//
//      h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

namespace deepgalois {
 
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
    math::mvmul(CblasTrans, olen, ilen, 1.0, W, &accum[v*ilen], 0.0, a); // a = W * accum[v]; [olen x ilen] * [ilen x 1] = [olen x 1]
    math::mvmul(CblasTrans, olen, ilen, 1.0, Q, &in[v*ilen], 0.0, b);    // b = Q * in; [olen x ilen] * [ilen x 1] = [olen x 1] 
    math::vadd_cpu(olen, a, b, c); // c = a + b; [olen x 1]
    return c; // the feature vector to update h[v]
  }
/*
  emb_t applyVertex(emb_t in, emb_t accum) {
    auto n = get_num_samples();
    auto ilen = get_in_feat_len();
    auto olen = get_out_feat_len();
    emb_t a, b, c;
    math::matmul(n, olen, ilen, accum, W, a); // a = accum * W; [n x ilen] * [ilen x olen] = [n x olen]
    math::matmul(n, olen, ilen, in, Q, b);    // b = in * Q; [n x ilen] * [ilen x olen] = [n x olen] 
    math::vadd(n*olen, a, b, c); // c = a + b; [n x olen]
    return c; // all the feature vectors to update the entire h
  }
*/
  //void update_all(size_t len, Graph& g, const emb_t in, emb_t out) {
  //}
};

}
#include "engine.h"
