// Graph Attension Networks (GAT)
// Xuhao Chen <cxh@utexas.edu>
#include "lonestargnn.h"

const char* name = "Graph Attention Networks (GAT)";
const char* desc = "Graph Attention Networks on an undirected graph: <https://arxiv.org/pdf/1710.10903.pdf>";
const char* url  = 0;

// define aggregator here

// math: h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}
// where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and node :math:`j`:
// .. math:: \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})
//                e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

#include "engine.h"
