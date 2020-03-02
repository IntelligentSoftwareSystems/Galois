#include "aggregator.h"
#include "math_functions.hh"

void update_all(size_t len, Graph& g, const float_t* in, float_t* out,
                bool norm, const float_t* norm_factor) {
  galois::do_all(galois::iterate(g.begin(), g.end()),
                 [&](const auto& src) {
                   clear(len, &out[src * len]);
                   float_t a = 0.0, b = 0.0;
                   if (norm)
                     a = norm_factor[src];
                   // gather neighbors' embeddings
                   for (const auto e : g.edges(src)) {
                     const auto dst = g.getEdgeDst(e);
                     if (norm) {
                       b = a * norm_factor[dst];
                       vec_t neighbor(len);
                       mul_scalar(len, b, &in[dst * len], &neighbor[0]);
                       vadd(len, &out[src * len], &neighbor[0],
                            &out[src * len]); // out[src] += in[dst]
                     } else
                       vadd(len, &out[src * len], &in[dst * len],
                            &out[src * len]); // out[src] += in[dst]
                   }
                 },
                 galois::chunk_size<CHUNK_SIZE>(), galois::steal(),
                 galois::loopname("update_all"));
}
