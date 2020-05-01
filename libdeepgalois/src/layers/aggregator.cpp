#include "deepgalois/layers/aggregator.h"
#include "deepgalois/math_functions.hh"

#ifdef CPU_ONLY
void deepgalois::update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  #ifndef GALOIS_USE_DIST
  galois::do_all(galois::iterate(size_t(0), g.size()),[&](const auto src) {
  #else
  auto& rangeObj = g.allNodesRange();
  galois::do_all(galois::iterate(rangeObj), [&](const auto src) {
  #endif
    // zero out the output data
    math::clear_cpu(len , &out[src * len]);
    float_t a = 0.0;
    float_t b = 0.0;
    // get normalization factor if needed
    if (norm) a = norm_factor[src];
    // gather neighbors' embeddings
    for (auto e = g.edge_begin(src); e != g.edge_end(src); e++) {
      const auto dst = g.getEdgeDst(e);
      if (norm) {
        // normalize b as well
        b = a * norm_factor[dst];
        //float_t* neighbor = new float_t[len]; // this is super slow
        vec_t neighbor(len);
        // scale the neighbor's data using the normalization factor
        math::scale(len, b, &in[dst * len], &neighbor[0]);
        // use scaled data to update; out[src] += in[dst]
        math::vadd_cpu(len, &out[src * len], &neighbor[0],  &out[src * len]);
      } else {
        // add embeddings from neighbors together; out[src] += in[dst]
        math::vadd_cpu(len, &out[src * len], &in[dst * len], &out[src * len]);
      }
    }
  }, galois::steal(), galois::no_stats(), galois::loopname("update_all"));
}

void deepgalois::update_all_csrmm(size_t len, Graph& g, const float_t* in, float_t* out,
                                  bool norm, const float_t* norm_factor) {
  galois::StatTimer Tcsrmm("CSRMM-MKL");
  //galois::gPrint("csrmm mkl\n");
  Tcsrmm.start();
  unsigned n = g.size();
  math::clear_cpu(n*len, out);
  math::csrmm_cpu(n, len, n, g.sizeEdges(), 1.0, norm_factor, 
            (const int*)g.row_start_ptr(), (const int*)g.edge_dst_ptr(), in, 0.0, out);
  Tcsrmm.stop();
}
#endif
