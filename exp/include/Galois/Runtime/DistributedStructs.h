#ifndef GALOIS_DISTRIBUTEDSTRUCTS_H
#define GALOIS_DISTRIBUTEDSTRUCTS_H

#include "Galois/Runtime/Directory.h"
#include "Galois/Runtime/MethodFlags.h"

namespace Galois {
namespace Runtime {

template<typename T>
struct dptr {
   int owner;
   T *obj;

   dptr() {
      owner = DIR::getTaskRank();
      obj = new T;
      return;
   }

   T *operator->() {
      T *ret;
      // get the pointer to the copy of the struct
      ret = (T*)DIR::resolve(obj,owner,sizeof(T));
      return ret;
   }
};

} } // end namespace
#endif
