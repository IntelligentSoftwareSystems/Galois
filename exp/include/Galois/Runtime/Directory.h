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
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_DIRECTORY_H
#define GALOIS_RUNTIME_DIRECTORY_H

#include "Galois/gstl.h"
#include "Galois/Runtime/ll/TID.h"
//#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/Network.h"
#include "Galois/Runtime/Tracer.h"
#include "Galois/Runtime/Support.h"
#include "Galois/Runtime/ll/SimpleLock.h"
#include "Galois/Runtime/ll/gio.h"
#include "Galois/Runtime/FatPointer.h"
#include "Galois/Runtime/Lockable.h"
#include "Galois/Runtime/CacheManager.h"

#include <boost/utility.hpp>
#include <boost/intrusive_ptr.hpp>

#include <mutex>
#include <unordered_map>
#include <set>
#include <functional>
#include <array>

namespace Galois {
namespace Runtime {

enum ResolveFlag {INV=0, RO=1, RW=2};

//Base class for common directory operations
class BaseDirectory {

protected:

  //These wrap type information for various dispatching purposes.  This
  //let's us keep vtables out of user objects
  class typeHelper {
  protected:
    typeHelper(recvFuncTy rWR);

  public:
    virtual void serialize(SendBuffer&, Lockable*) const = 0;
    virtual void deserialize(RecvBuffer&, Lockable*) const = 0;
    virtual void cmCreate(fatPointer, ResolveFlag, RecvBuffer&) const = 0;

    recvFuncTy writeremote;
  };
  
  template<typename T>
  class typeHelperImpl : public typeHelper {
    typeHelperImpl();
  public:
    static typeHelperImpl* get() {
      static typeHelperImpl th;
      return &th;
    }

    virtual void serialize(SendBuffer& buf, Lockable* obj) const;
    virtual void deserialize(RecvBuffer&, Lockable*) const;
    virtual void cmCreate(fatPointer, ResolveFlag, RecvBuffer&) const;
  };
    
  LockManagerBase dirContext;
  LL::SimpleLock dirContext_lock;

  bool dirAcquire(Lockable*);
  void dirRelease(Lockable*);
  bool dirOwns(Lockable*);

};

//handle local objects
class LocalDirectory : public BaseDirectory {
  struct metadata {
    //Lock protecting this structure
    LL::SimpleLock lock;
    //Locations which have the object in RO state
    std::set<uint32_t> locRO;
    //Location which has the object in RW state
    uint32_t locRW;
    //ID of host recalled for
    uint32_t recalled;
    //outstanding requests
    std::set<uint32_t> reqsRO;
    std::set<uint32_t> reqsRW;
    //Last sent transfer
    //whether object is participating in priority protocol
    bool contended;

    //Type aware helper functions
    typeHelper* th;

    //Add a request
    void addReq(uint32_t dest, ResolveFlag flag);

    //Get next requestor
    std::pair<uint32_t, bool> getNextDest() {
      uint32_t nextDest = ~0;
      bool nextIsRW = false;
      if (!reqsRO.empty())
        nextDest = *reqsRO.begin();
      if (!reqsRW.empty()) {
        nextDest = std::min(nextDest, *reqsRW.begin());
        if (*reqsRW.begin() == nextDest)
          nextIsRW = true;
      }
      return std::make_pair(nextDest, nextIsRW);
    }


    //!returns whether object is still needs processing
    bool writeback();

    uint32_t removeNextRW() {
      uint32_t retval = *reqsRW.begin();
      reqsRW.erase(reqsRW.begin());
      assert(retval != ~0);
      assert(retval != NetworkInterface::ID);
      assert(locRW == ~0);
      assert(locRO.empty());
      return retval;
    }

    uint32_t removeNextRO() {
      uint32_t retval = *reqsRO.begin();
      reqsRO.erase(reqsRO.begin());
      locRO.insert(retval);
      assert(retval != ~0);
      assert(retval != NetworkInterface::ID);
      assert(locRW == ~0);
      return retval;
    }

    metadata() :locRW(~0), recalled(~0), th(nullptr) {}

    friend std::ostream& operator<< (std::ostream& os, const metadata& md) {
      std::ostream_iterator<uint32_t> out_it(os, ",");
      os << "locRO:<";
      std::copy(md.locRO.begin(), md.locRO.end(), out_it);
      os << ">,locRW:" << md.locRW << ",recalled:" << md.recalled << ",reqsRO:<";
      std::copy(md.reqsRO.begin(), md.reqsRO.end(), out_it);
      os << ">,reqsRW:<";
      std::copy(md.reqsRW.begin(), md.reqsRW.end(), out_it);
      os << ">,contended:" << md.contended << ",th:" << md.th;
      return os;
    }
  };

  std::unordered_map<Lockable*, metadata> dir;
  LL::SimpleLock dir_lock;
  
  metadata& getMD(Lockable*);

  std::atomic<int> outstandingReqs;

  //!Send object to dest
  void sendObj(metadata&, uint32_t dest, Lockable*, ResolveFlag);

  //!Send object to all outstanding readers
  void sendToReaders(metadata&, Lockable*);

  //!Send invalidate to all outstanding readers
  void invalidateReaders(metadata&, Lockable*);

  void considerObject(metadata& m, Lockable*);

  // bool updateObjState(Lockable*, metadata&);

  // void recvRequestImpl(fatPointer ptr, ResolveFlag flag, uint32_t dest, typeHelper* th);
  // void recvObjectImpl(fatPointer ptr);



  void ackInvalidateImpl(fatPointer, uint32_t);

  void recvObjectImpl(fatPointer, RecvBuffer&);

  void recvRequestImpl(fatPointer, ResolveFlag, uint32_t, typeHelper*);

protected:
  //remote poriton of the API
  friend class RemoteDirectory;

  //Object arived at destination
  //static void ackObject(RecvBuffer&); //fatPointer, ResolveFlag, uint32_t);

  //Recieve a remote request for an object
  template<typename T>
  static void recvRequest(RecvBuffer&);

  //Recieve an object (writeback)
  static void recvObject(RecvBuffer&);

  static void ackInvalidate(RecvBuffer&);

public:
  //Local portion of API

  void fetch(fatPointer ptr, ResolveFlag flag) {
    assert(ptr.isLocal());
    metadata& md = getMD(static_cast<Lockable*>(ptr.getObj()));
    std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
    assert(md.th);
    md.addReq(NetworkInterface::ID, flag);
  }

  //! engage priority protocol for ptr
  void setContended(fatPointer ptr);
  //! unengage priority protocol for ptr
  void clearContended(fatPointer ptr);

  void makeProgress();
  void dump();
};

LocalDirectory& getLocalDirectory();

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

//Generic landing pad for requests
template<typename T>
void LocalDirectory::recvRequest(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  uint32_t dest;
  gDeserialize(buf, ptr, flag, dest);
  getLocalDirectory().recvRequestImpl(ptr, flag, dest, typeHelperImpl<T>::get());
}

////////////////////////////////////////////////////////////////////////////////


class RemoteDirectory : public BaseDirectory {

  //metadata for an object.
  struct metadata {
    enum StateFlag {
      INVALID=0,    //Not present and not requested
      PENDING_RO=1, //Not present and requested RO
      PENDING_RW=2, //Not present and requested RW
      HERE_RO=3,         //present as RO
      HERE_RW=4,         //present as RW
      UPGRADE=5     //present as RO and requested RW
    };
    LL::SimpleLock lock;
    StateFlag state;
    uint32_t recalled;
    bool contended;
    typeHelper* th;

    metadata();
    void invalidate();
    void recvObj(ResolveFlag);
    ResolveFlag fetch(ResolveFlag flag);
  };
  //sigh
  friend std::ostream& operator<<(std::ostream& os, const metadata& md);

  std::unordered_map<fatPointer, metadata> md;
  LL::SimpleLock md_lock;

  // std::deque<std::tuple<fatPointer, Lockable*, typeHelper*>> writeback;

  //get metadata for pointer
  metadata& getMD(fatPointer ptr);

  //do everything needed to evict ptr
  void doInvalidate(metadata& md, fatPointer ptr);

  //try to writeback ptr, may fail
  void tryWriteBack(metadata& md, fatPointer ptr);

  //handle RW->INV request
  void recvRequestImpl(fatPointer, uint32_t);

  //handle object ariving
  void recvObjectImpl(fatPointer, ResolveFlag, typeHelper*, RecvBuffer&);

  //handle RO->INV request
  void recvInvalidateImpl(fatPointer, uint32_t);

protected: 
  //Remote portion of API
  friend class LocalDirectory;
  template<typename T> friend class BaseDirectory::typeHelperImpl;

  //Recieve a request to writeback an RW obj (RW->INV)
  static void recvRequest(RecvBuffer& buf);

  //Recieve an object
  template<typename T>
  static void recvObject(RecvBuffer& buf);

  //Request for RO->INV from owner
  static void recvInvalidate(RecvBuffer& buf);

public:
  //Local portion of API

  //! process any queues
  void makeProgress();

  //! initiate, if necessary, a fetch of a remote object
  template<typename T>
  void fetch(fatPointer ptr, ResolveFlag flag);

  //! engage priority protocol for ptr
  void setContended(fatPointer ptr);
  //! unengage priority protocol for ptr
  void clearContended(fatPointer ptr);

  void dump(fatPointer ptr); //dump one object info
  void dump(); //dump direcotry status
};

RemoteDirectory& getRemoteDirectory();

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template<typename T>
void RemoteDirectory::recvObject(RecvBuffer& buf) {
  fatPointer ptr;
  ResolveFlag flag;
  gDeserialize(buf, ptr, flag);
  getRemoteDirectory().recvObjectImpl(ptr, flag, typeHelperImpl<T>::get(), buf);
}

template<typename T>
void RemoteDirectory::fetch(fatPointer ptr, ResolveFlag flag) {
  metadata& md = getMD(ptr);
  std::lock_guard<LL::SimpleLock> lg(md.lock, std::adopt_lock);
  //trace("RemoteDirectory::fetch for % flag %\n", ptr, flag);
  assert(md.th == typeHelperImpl<T>::get() || !md.th);
  if (!md.th)
    md.th = typeHelperImpl<T>::get();
  ResolveFlag requestFlag = md.fetch(flag);
  if (requestFlag != INV) {
    SendBuffer sbuf;
    gSerialize(sbuf, ptr, requestFlag, NetworkInterface::ID);
    getSystemNetworkInterface().send(ptr.getHost(), &LocalDirectory::recvRequest<T>, sbuf);
  }
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
BaseDirectory::typeHelperImpl<T>::typeHelperImpl()
  : typeHelper(&RemoteDirectory::recvObject<T>)
{}

template<typename T>
void BaseDirectory::typeHelperImpl<T>::serialize(SendBuffer& buf, Lockable* ptr) const {
  gSerialize(buf, *static_cast<T*>(ptr));
}

template<typename T>
void BaseDirectory::typeHelperImpl<T>::deserialize(RecvBuffer& buf, Lockable* ptr) const {
  gDeserialize(buf, *static_cast<T*>(ptr));
}

template<typename T>
void BaseDirectory::typeHelperImpl<T>::cmCreate(fatPointer ptr, ResolveFlag flag, RecvBuffer& buf) const {
  //FIXME: deal with RO
  getCacheManager().create<T>(ptr, buf);
}

////////////////////////////////////////////////////////////////////////////////

struct remote_ex { fatPointer ptr; };

////////////////////////////////////////////////////////////////////////////////

//! Make progress in the network
inline void doNetworkWork() {
  if ((NetworkInterface::Num > 1)) {// && (LL::getTID() == 0)) {
    auto& net = getSystemNetworkInterface();
    net.flush();
    while (net.handleReceives()) { net.flush(); }
    getRemoteDirectory().makeProgress();
    getLocalDirectory().makeProgress();
    net.flush();
    while (net.handleReceives()) { net.flush(); }
  }
}


} // namespace Runtime

} // namespace Galois

#endif
