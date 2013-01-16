/** Galois Distributed Directory -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2012, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Manoj Dhanapal <madhanap@cs.utexas.edu>
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_DIRECTORY_H
#define GALOIS_RUNTIME_DIRECTORY_H

#include "Galois/Runtime/NodeRequest.h"
#include <unordered_map>
#include "Galois/Runtime/ll/TID.h"
#include "Galois/Runtime/MethodFlags.h"
#include <unistd.h>

#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/Network.h"

#define PLACEREQ 10000

#if 0
namespace Galois {
namespace Runtime {

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
}
} // end namespace
#endif

#define INELIGIBLE_COUNT 12

namespace Galois {
namespace Runtime {
namespace Distributed {

class RemoteDirectory {

  struct objstate {
    // Remote - The object has been returned to the owner
    // Local  - Local object eligible for use as soon as received
    //          Inelgible for transfer till INELI2ELI_COUNT reqs or local use
    enum ObjStates { Remote, Local };

    uintptr_t localobj;
    enum ObjStates state;
    int count;
  };

  struct ohash : public unary_function<std::pair<uintptr_t, uint32_t>, size_t> {
    size_t operator()(const std::pair<uintptr_t, uint32_t>& v) const {
      return std::hash<uintptr_t>()(v.first) ^ std::hash<uint32_t>()(v.second);
    }
  };

  std::unordered_map<std::pair<uintptr_t, uint32_t>, objstate, ohash> curobj;
  Galois::Runtime::LL::SimpleLock<true> Lock;

  //returns a valid local pointer to the object if it exists
  //or returns null
  uintptr_t haveObject(uintptr_t ptr, uint32_t owner);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t owner, recvFuncTy pad);

public:

  // Handles incoming requests for remote objects
  // if Ineligible, transfer to Eligible after INELI2ELI_COUNT requests
  // if Eligible return the object back to owner and mark as Remote
  template<typename T>
  static void remoteReqLandingPad(Galois::Runtime::Distributed::RecvBuffer &);

  // Landing Pad for incoming remote objects
  template<typename T>
  static void remoteDataLandingPad(Galois::Runtime::Distributed::RecvBuffer &);

  //resolve a pointer, owner pair
  //precondition: owner != networkHostID
  template<typename T>
  T* resolve(uintptr_t ptr, uint32_t owner);
};

class LocalDirectory: public SimpleRuntimeContext {

  struct objstate {
    // Remote - Object passed to a remote host
    // Local - Local object may be locked
    enum ObjStates { Remote, Local };

    int sent_to;  // valid only for remote objects
    enum ObjStates state;
  };

  std::unordered_map<uintptr_t, objstate> curobj;
  Galois::Runtime::LL::SimpleLock<true> Lock;

  // returns a valid local pointer to the object if not remote
  uintptr_t haveObject(uintptr_t ptr, int *remote);

  // places a remote request for the node
  void fetchRemoteObj(uintptr_t ptr, uint32_t remote, recvFuncTy pad);

  // virtual function in SimpleRuntimeContext
  virtual void sub_acquire(Lockable* L);

public:

  LocalDirectory(): SimpleRuntimeContext(true) {}

  // forward the request if the state is remote
  // send the object if local and not locked, also mark as remote
  template<typename T>
  static void localReqLandingPad(Galois::Runtime::Distributed::RecvBuffer &);

  // send the object if local, not locked and mark obj as remote
  template<typename T>
  static void localDataLandingPad(Galois::Runtime::Distributed::RecvBuffer &);

  // resolve a pointer
  template<typename T>
  T* resolve(uintptr_t ptr);
};


RemoteDirectory& getSystemRemoteDirectory();

LocalDirectory& getSystemLocalDirectory();

} //Distributed
} //Runtime
} //Galois

template<typename T>
T* Galois::Runtime::Distributed::RemoteDirectory::resolve(uintptr_t ptr, uint32_t owner) {
  assert(owner != networkHostID);
  uintptr_t p = haveObject(ptr, owner);
  if (!p) {
    fetchRemoteObj(ptr, owner, &Galois::Runtime::Distributed::LocalDirectory::localReqLandingPad<T>);
    throw 1; //FIXME: integrate with acquire's throw type
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void Galois::Runtime::Distributed::RemoteDirectory::remoteReqLandingPad(RecvBuffer &buf) {
  int owner;
  T *data;
  uintptr_t ptr;
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  // check if the object can be sent
  if ((iter == rd.curobj.end()) || (OBJSTATE.state == RemoteDirectory::objstate::Remote)) {
    // object can't be remote if the owner makes a request
    abort();
  }
  else if (OBJSTATE.state == RemoteDirectory::objstate::Local) {
// modify the check here to use acquire and free instead of Eligible and Ineligible states
    // check number of requests and make eligible if needed
    OBJSTATE.count++;
    if (OBJSTATE.count == INELIGIBLE_COUNT) {
      // the data should be sent
//    OBJSTATE.state = RemoteDirectory::objstate::Eligible;
    }
//  if (OBJSTATE.state == RemoteDirectory::objstate::Eligible) {
      // object should be sent to the remote host
      data = reinterpret_cast<T*>(OBJSTATE.localobj);
      SendBuffer sbuf;
      NetworkInterface& net = getSystemNetworkInterface();
      sbuf.serialize(ptr);
      sbuf.serialize(sizeof(*data));
      sbuf.serialize(*data);
      rd.curobj.erase(make_pair(ptr,owner));
      net.sendMessage(owner,&Galois::Runtime::Distributed::LocalDirectory::localDataLandingPad<T>,sbuf);
      free(data);
//  }
  }
  else {
    cout << "Unexpected state in remoteReqLandingPad" << endl;
  }
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
void Galois::Runtime::Distributed::RemoteDirectory::remoteDataLandingPad(RecvBuffer &buf) {
  uint32_t owner;
  int size;
  T *data;
  uintptr_t ptr;
  RemoteDirectory& rd = getSystemRemoteDirectory();
#define OBJSTATE (*iter).second
  rd.Lock.lock();
  buf.deserialize(ptr);
  buf.deserialize(owner);
  auto iter = rd.curobj.find(make_pair(ptr,owner));
  buf.deserialize(size);
  data = (T*)calloc(1,size);
  buf.deserialize((*data));
  OBJSTATE.state = RemoteDirectory::objstate::Local;
// lock the object with the special value to mark eligible
  OBJSTATE.localobj = (uintptr_t)data;
  OBJSTATE.count = 0;
  rd.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
T* Galois::Runtime::Distributed::LocalDirectory::resolve(uintptr_t ptr) {
  int sent = 0;
  RemoteDirectory& rd = getSystemRemoteDirectory();
  uintptr_t p = haveObject(ptr, &sent);
  if (!p) {
    fetchRemoteObj(ptr, sent, &Galois::Runtime::Distributed::RemoteDirectory::remoteReqLandingPad<T>);
    throw 1; //FIXME: integrate with acquire's throw type
  }
  return reinterpret_cast<T*>(p);
}

template<typename T>
void Galois::Runtime::Distributed::LocalDirectory::localReqLandingPad(RecvBuffer &buf) {
  uint32_t remote_to;
  T *data;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  LocalDirectory& ld = getSystemLocalDirectory();
  ld.Lock.lock();
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  auto iter = ld.curobj.find(ptr);
  buf.deserialize(remote_to);
  // add object to list if it's not already there
  if (iter == ld.curobj.end()) {
    LocalDirectory::objstate list_obj;
    list_obj.sent_to = remote_to;
    list_obj.state = LocalDirectory::objstate::Local;
    ld.curobj[ptr] = list_obj;
  }
  // check if the object can be sent
  if (OBJSTATE.state == LocalDirectory::objstate::Remote) {
    // object is remote so place a return request
    ld.fetchRemoteObj(ptr, OBJSTATE.sent_to, &Galois::Runtime::Distributed::RemoteDirectory::remoteReqLandingPad<T>);
  }
  else if (OBJSTATE.state == LocalDirectory::objstate::Local) {
// check if locked using the new acquire calls!
    // object should be sent to the remote host
    SendBuffer sbuf;
    NetworkInterface& net = getSystemNetworkInterface();
    sbuf.serialize(ptr);
    sbuf.serialize(networkHostID);
    sbuf.serialize(sizeof(*data));
    sbuf.serialize(*data);
    OBJSTATE.state = LocalDirectory::objstate::Remote;
    net.sendMessage(remote_to,&Galois::Runtime::Distributed::RemoteDirectory::remoteDataLandingPad<T>,sbuf);
  }
  else {
    cout << "Unexpected state in localReqLandingPad" << endl;
  }
  ld.Lock.unlock();
#undef OBJSTATE
  return;
}

template<typename T>
void Galois::Runtime::Distributed::LocalDirectory::localDataLandingPad(RecvBuffer &buf) {
  int size;
  T *data;
  uintptr_t ptr;
#define OBJSTATE (*iter).second
  LocalDirectory& ld = getSystemLocalDirectory();
  ld.Lock.lock();
  buf.deserialize(ptr);
  data = reinterpret_cast<T*>(ptr);
  auto iter = ld.curobj.find(ptr);
  buf.deserialize(size);
  buf.deserialize((*data));
  OBJSTATE.state = LocalDirectory::objstate::Local;
  ld.Lock.unlock();
#undef OBJSTATE
  return;
}

#endif
