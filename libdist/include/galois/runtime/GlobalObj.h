#include <vector>
#include <cstdint>
#include <cassert>

#ifndef _GALOIS_DIST_GLOBAL_OBJECT_H
#define _GALOIS_DIST_GLOBAL_OBJECT_H

namespace galois {
namespace runtime {

class GlobalObject {
  // FIXME: lock?
  // TODO make a pointer to avoid static initialization?
  static std::vector<uintptr_t> allobjs;
  uint32_t objID;

 protected:
  GlobalObject(const GlobalObject&) = delete;
  GlobalObject(GlobalObject&&) = delete;

  static uintptr_t ptrForObj(unsigned oid);

  template<typename T>
  GlobalObject(const T* ptr) {
    objID = allobjs.size();
    allobjs.push_back(reinterpret_cast<uintptr_t>(ptr));
  }

  uint32_t idForSelf() const {
    return objID;
  }
};

} // end namespace runtime
} // end namespace galois

#endif//_GALOIS_DIST_GLOBAL_OBJECT_H
