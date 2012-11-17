#ifndef GALOIS_DIRECTORY_H
#define GALOIS_DIRECTORY_H

#include "Galois/Runtime/NodeRequest.h"
#include <unordered_map>
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/MethodFlags.h"

#define PLACEREQ 100

namespace GaloisRuntime {

namespace DIR {

extern NodeRequest nr;

static inline int getTaskRank() {
   return nr.taskRank;
}

static inline int getNoTasks() {
   return nr.numTasks;
}

/* return all the remote nodes before termination */
static inline void return_comm() {
   nr.return_remote();
   for (int i = 0; i < 100; i++) {
      usleep(1);
      nr.Communicate();
   }
   return;
}

static inline void comm() {
   nr.Communicate();
   return;
}

// class dir {

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
               // requests may be lost as remote tasks may request
               // for the same node
               count = 0;
               nr.PlaceRequest (owner, ptr, size);
            }
            usleep(1);
            // another thread might have got the same data
            nr.checkRequest(ptr,owner,&tmp,size);
         } while(!tmp);
      }
   pthread_mutex_unlock(&mutex);
      // lock the object
      L = reinterpret_cast<Lockable*>(tmp);
      acquire(L,Galois::MethodFlag::ALL);
      return tmp;
   }
// };

} // end of DIR namespace

} // end namespace
#endif
