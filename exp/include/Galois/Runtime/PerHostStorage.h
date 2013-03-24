#ifndef GALOIS_RUNTIME_PERHOSTSTORAGE
#define GALOIS_RUNTIME_PERHOSTSTORAGE

#include "Galois/Runtime/Serialize.h"

namespace Galois {
namespace Runtime {

using namespace Galois::Runtime::Distributed;

namespace hidden {
using namespace Galois::Runtime::Distributed;

template<typename T>
struct Deallocate {
  gptr<T> p;
  T* rp;
  
  Deallocate(gptr<T> p): p(p), rp(0) { }
  Deallocate() { }

  void operator()(unsigned tid, unsigned) {
    if (tid == 0 && rp) {
      delete rp;
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, p); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, p); rp = &*p; }
};

template<typename T>
struct AllocateEntry {
  gptr<T> p;
  T* rp;
  
  AllocateEntry(gptr<T> p): p(p), rp(0) { }
  AllocateEntry() { }

  void operator()(unsigned tid, unsigned) {
    if (tid == 0) {
      rp = &*p; // Fault in persistent object
    }
  }

  typedef int tt_has_serialize;
  void serialize(SerializeBuffer& s) const { gSerialize(s, p); }
  void deserialize(DeSerializeBuffer& s) { gDeserialize(s, p); }
};

} // end namespace

template<typename T>
void allocatePerHost(T* orig) {
  Galois::on_each(hidden::AllocateEntry<T>(gptr<T>(orig)));
}

/**
 * Deallocate per host copies, but master still has to deallocate original (if necessary).
 */
template<typename T>
void deallocatePerHost(gptr<T> p) {
  Galois::on_each(hidden::Deallocate<T>(p));
}

template<typename T>
class PerHostStorage : public T {

public:
  typedef int tt_has_serialize;
  typedef int tt_is_persistent;

  PerHostStorage() { 
    //allocatePerHost(this);
  }
  PerHostStorage(DeSerializeBuffer& s) { }

  void serialize(SerializeBuffer& s) const { }
  void deserialize(DeSerializeBuffer& s) { }
};

} // end namespace
} // end namespace

#endif
