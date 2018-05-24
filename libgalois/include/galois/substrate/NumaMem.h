#ifndef GALOIS_SUBSTRATE_NUMAMEM
#define GALOIS_SUBSTRATE_NUMAMEM

#include <cstddef>
#include <memory>
#include <vector>

namespace galois {
namespace substrate {

namespace internal {
struct largeFreer {
  size_t bytes;
  void operator()(void* ptr) const;
};
}//namespace internal

typedef std::unique_ptr<void, internal::largeFreer> LAptr;

LAptr largeMallocLocal(size_t bytes); // fault in locally
LAptr largeMallocFloating(size_t bytes); // leave numa mapping undefined
// fault in interleaved mapping
LAptr largeMallocInterleaved(size_t bytes, unsigned numThreads);
// fault in block interleaved mapping
LAptr largeMallocBlocked(size_t bytes, unsigned numThreads);

// fault in specified regions for each thread (threadRanges)
template<typename RangeArrayTy>
LAptr largeMallocSpecified(size_t bytes, uint32_t numThreads,
                           RangeArrayTy& threadRanges, size_t elementSize);


} // namespace substrate
} // namespace galois

#endif //GALOIS_SUBSTRATE_NUMAMEM
