// GraphSAGE 
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


#include "engine.h"
