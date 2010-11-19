#ifndef _FIXEDREQUESTHEAP_H_
#define _FIXEDREQUESTHEAP_H_

/**
 * @class FixedRequestHeap
 * @brief Always grabs the same size, regardless of the request size.
 */

namespace Hoard {

template <size_t RequestSize,
	  class SuperHeap>
class FixedRequestHeap : public SuperHeap {
public:
  inline void * malloc (size_t) {
    return SuperHeap::malloc (RequestSize);
  }
  inline static size_t getSize (void *) {
    return RequestSize;
  }
};

}

#endif
