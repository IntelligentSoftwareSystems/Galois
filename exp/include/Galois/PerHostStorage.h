#ifndef GALOIS_PERHOSTSTORAGE
#define GALOIS_PERHOSTSTORAGE

#include "Galois/Runtime/Serialize.h"

namespace Galois {
using namespace Galois::Runtime::Distributed;

template<typename T>
class PerHostStorage : public T {

public:
  typedef int tt_has_serialize;
  typedef int tt_is_persistent;

  PerHostStorage() { }
  PerHostStorage(DeSerializeBuffer& s) { }

  void serialize(SerializeBuffer& s) const { }
  void deserialize(DeSerializeBuffer& s) { }
};

namespace hidden {
using namespace Galois::Runtime::Distributed;

template<typename T>
struct Deallocate {
  gptr<T> p;
  T* rp;
  
  Deallocate(gptr<T> p): p(p), rp(0) { }
  Deallocate() { }

  void operator()(unsigned x, unsigned tid) {
    if (Galois::Runtime::LL::getTID() == 0 && rp) {
      delete rp;
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, p); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, p); rp = &*p; }
};

} // end namespace

/**
 * Deallocate per host copies, but master still has to deallocate original (if necessary).
 */
template<typename T>
void deallocatePerHost(gptr<T> p) {
  Galois::on_each(hidden::Deallocate<T>(p));
}

} // end namespace

#endif
