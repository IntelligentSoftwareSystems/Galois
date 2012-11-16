#ifndef GALOIS_DIRECTORY_H
#define GALOIS_DIRECTORY_H

#include "Galois/Runtime/NodeRequest.h"
#include <unordered_map>
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/MethodFlags.h"

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

   extern unordered_map<Pair,void*,HashFunction,SetEqual>      hash_table;

   static void *resolve (void *ptr, int owner, size_t size) {
      Lockable *L;
      static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
      // check if we are the owner n haven't sent the data out
      if ((owner == nr.taskRank) && (!nr.checkSent(owner,ptr))) {
         L = reinterpret_cast<Lockable*>(ptr);
         // lock the object
         acquire(L,Galois::MethodFlag::ALL);
         return ptr;
      }
   pthread_mutex_lock(&mutex);
      Pair p(owner,ptr);
      void *tmp = hash_table[p];
      bool flag = false;
      if (!tmp) {
         char *buf = (char*)malloc(sizeof(char)*size+5);
         // go get a copy of the object
         nr.PlaceRequest (owner, ptr, size);
         do {
            usleep(1);
            // another thread might have got the same data
            flag = nr.checkRequest(ptr,owner,buf,size);
         } while(!flag);
         if (flag)
           tmp = (void*)buf;
         else
           free(buf);
         if (owner == nr.taskRank) {
            memcpy(ptr,buf,size);
            free(buf);
            tmp = ptr;
            L = reinterpret_cast<Lockable*>(ptr);
            // lock the object
            acquire(L,Galois::MethodFlag::ALL);
         }
         else {
            // add to hash_table[owner][ptr]
            hash_table[p] = tmp;
         }
      }
   pthread_mutex_unlock(&mutex);
      return tmp;
   }
// };

} // end of DIR namespace

} // end namespace
#endif
