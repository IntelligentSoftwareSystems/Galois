#ifndef GALOIS_SUBSTRATE_PAGEALLOC_H
#define GALOIS_SUBSTRATE_PAGEALLOC_H

#include <cstddef>

namespace galois {
namespace substrate {

//size of pages
size_t allocSize();

//allocate contiguous pages, optionally faulting them in
void* allocPages(unsigned num, bool preFault);

//free page range
void freePages(void* ptr, unsigned num);

} // namespace substrate
} // namespace galois

#endif //GALOIS_SUBSTRATE_PAGEALLOC_H
