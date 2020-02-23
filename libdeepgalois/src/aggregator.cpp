#include "types.h"
#include "gtypes.h"
#include "aggregator.h"
#include "math_functions.hh"

void update_all(Graph &g, const tensor_t &in, tensor_t &out, bool norm, const vec_t &norm_factor) {
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		clear(out[src]); // TODO: vectorize clear
		float_t a = 0.0, b = 0.0;
		if (norm) a = norm_factor[src];
		// gather neighbors' embeddings
		for (const auto e : g.edges(src)) {
			const auto dst = g.getEdgeDst(e);
			if (norm) {
				b = a * norm_factor[dst];
				vec_t neighbor = in[dst];
				mul_scalar(b, neighbor);
				vadd(out[src], neighbor, out[src]); // out[src] += in[dst]
			} else vadd(out[src], in[dst], out[src]); // out[src] += in[dst]
		}
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("update_all"));
}

void update_all(Graph &g, const vec_t &in, tensor_t &out, bool norm, const vec_t &norm_factor) {
	size_t len = out[0].size();
	galois::do_all(galois::iterate(g.begin(), g.end()), [&](const auto& src) {
		clear(out[src]);
		float_t a = 0.0, b = 0.0;
		if (norm) a = norm_factor[src];
		// gather neighbors' embeddings
		for (const auto e : g.edges(src)) {
			const auto dst = g.getEdgeDst(e);
			if (norm) {
				b = a * norm_factor[dst];
				vec_t neighbor(len);
				mul_scalar(len, b, &in[dst*len], neighbor.data());
				vadd(out[src], neighbor, out[src]); // out[src] += in[dst]
			} else vadd(len, out[src].data(), &in[dst*len], out[src].data()); // out[src] += in[dst]
		}
	}, galois::chunk_size<CHUNK_SIZE>(), galois::steal(), galois::loopname("update_all"));
}

