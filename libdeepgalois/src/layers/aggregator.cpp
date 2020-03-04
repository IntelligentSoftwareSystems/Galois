#include "deepgalois/layers/aggregator.h"
#include "deepgalois/math_functions.hh"

#ifdef CPU_ONLY
void deepgalois::update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  galois::do_all(galois::iterate(g), [&](const GNode src) {
   // zero out this node's out values
   deepgalois::math::clear(len, &out[src * len]);
   float_t a = 0.0;
   float_t b = 0.0;

   // get normalization factor if needed
   if (norm) a = norm_factor[src];

   // gather neighbors' embeddings
   for (const auto e : g.edges(src)) {
     const auto dst = g.getEdgeDst(e);

     if (norm) {
       // normalize b as well
       b = a * norm_factor[dst];
       vec_t neighbor(len);
       // scale the neighbor's data  using the normalization
       // factor
       deepgalois::math::mul_scalar(len, b, &in[dst * len], &neighbor[0]);
       // use scaled data to update
       deepgalois::math::vadd(len, &out[src * len], &neighbor[0],
            &out[src * len]); // out[src] += in[dst]
     } else
       // add embeddings from neighbors together
       deepgalois::math::vadd(len, &out[src * len],
                              &in[dst * len],
                              &out[src * len]); // out[src] += in[dst]
   }
 }, galois::steal(), galois::no_stats(), galois::loopname("update_all"));
}
#endif
