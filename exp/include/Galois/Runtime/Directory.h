#ifndef GALOIS_DIRECTORY_H
#define GALOIS_DIRECTORY_H

#include "Galois/Runtime/NodeRequest.h"
#include <unordered_map>
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/MethodFlags.h"

#define PLACEREQ 10000

namespace GaloisRuntime {

// the default value is false
extern bool distributed_foreach;

static inline void set_distributed_foreach(bool val) {
   distributed_foreach = val;
   return;
}

static inline bool get_distributed_foreach() {
   return distributed_foreach;
}

namespace DIR {

extern NodeRequest nr;

static inline int getTaskRank() {
   return nr.taskRank;
}

static inline int getNoTasks() {
   return nr.numTasks;
}

static inline void comm() {
   nr.Communicate();
   return;
}

   static void *resolve (void *ptr, int owner, size_t size) {
      Lockable *L;
      int count;
      static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
   pthread_mutex_lock(&mutex);
      void *tmp = nr.remoteAccess(ptr,owner);
      if (!tmp) {
         // go get a copy of the object
         nr.PlaceRequest (owner, ptr, size);
         count = 0;
         do {
            if (count++ == PLACEREQ) {
               // reqs may be lost as remote tasks may ask for the same node
               count = 0;
               // may be the only running thread - req a call to comm
               nr.Communicate();
               // sleep so that the caller is not flooded with requests
               usleep(10000);
               nr.PlaceRequest (owner, ptr, size);
            }
            // another thread might have got the same data
            nr.checkRequest(ptr,owner,&tmp,size);
         } while(!tmp);
      }
   pthread_mutex_unlock(&mutex);
      // lock the object if locked for use by the directory (setLockValue)
      L = reinterpret_cast<Lockable*>(tmp);
 //   if (get_distributed_foreach() && (getNoTasks() != 1))
      if (get_distributed_foreach())
        lockAcquire(L,Galois::MethodFlag::ALL);
      else
        acquire(L,Galois::MethodFlag::ALL);
      return tmp;
   }

} // end of DIR namespace

} // end namespace
#endif
