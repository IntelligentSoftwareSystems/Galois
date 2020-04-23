#pragma once
#include <vector>
#include "pangolin/gtypes.h"

inline std::vector<IndexT> PrefixSum(const std::vector<IndexT> &vec) {
	std::vector<IndexT> sums(vec.size() + 1);
	IndexT total = 0;
	for (size_t n=0; n < vec.size(); n++) {
		sums[n] = total;
		total += vec[n];
	}
	sums[vec.size()] = total;
	return sums;
}

#ifdef LARGE_SIZE
template <typename InTy = unsigned, typename OutTy = unsigned>
inline std::vector<OutTy> prefix_sum(const std::vector<InTy> &in) {
	std::vector<OutTy> sums(in.size() + 1);
	OutTy total = 0;
	for (size_t n = 0; n < in.size(); n++) {
		sums[n] = total;
		total += (OutTy)in[n];
	}
	sums[in.size()] = total;
	return sums;
}

template <typename InTy = unsigned, typename OutTy = unsigned>
inline std::vector<OutTy> parallel_prefix_sum(const std::vector<InTy> &in) {
    const size_t block_size = 1<<20;
    const size_t num_blocks = (in.size() + block_size - 1) / block_size;
    std::vector<OutTy> local_sums(num_blocks);
	// count how many bits are set on each thread
	galois::do_all(galois::iterate((size_t)0, num_blocks), [&](const size_t& block) {
		OutTy lsum = 0;
		size_t block_end = std::min((block + 1) * block_size, in.size());
		for (size_t i=block * block_size; i < block_end; i++)
			lsum += in[i];
		local_sums[block] = lsum;
	});
	std::vector<OutTy> bulk_prefix(num_blocks+1);
	OutTy total = 0;
	for (size_t block=0; block < num_blocks; block++) {
		bulk_prefix[block] = total;
		total += local_sums[block];
	}
	bulk_prefix[num_blocks] = total;
	std::vector<OutTy> prefix(in.size() + 1);
	galois::do_all(galois::iterate((size_t)0, num_blocks), [&](const size_t& block) {
		OutTy local_total = bulk_prefix[block];
		size_t block_end = std::min((block + 1) * block_size, in.size());
		for (size_t i=block * block_size; i < block_end; i++) {
			prefix[i] = local_total;
			local_total += in[i];
		}
	});
	prefix[in.size()] = bulk_prefix[num_blocks];
	return prefix;
}

#else
template <typename InTy = unsigned, typename OutTy = unsigned>
inline galois::gstl::Vector<OutTy> parallel_prefix_sum(const galois::gstl::Vector<InTy> &in) {
	galois::gstl::Vector<OutTy> sums(in.size() + 1);
	OutTy total = 0;
	for (size_t n = 0; n < in.size(); n++) {
		sums[n] = total;
		total += (OutTy)in[n];
	}
	sums[in.size()] = total;
	return sums;
} 
#endif

