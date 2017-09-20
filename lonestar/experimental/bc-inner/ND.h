#ifndef _ND_H_
#define _ND_H_

#define USEPTHREADSM 0

#include <cassert>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <algorithm>

#if USEPTHREADSM
#include <pthread.h>
#endif

#include "galois/substrate/SimpleLock.h"
#include "llvm/ADT/SmallVector.h"
#include "control.h"
//#include "datatypes.h"

extern int DEF_DISTANCE;

struct ND {
 
  int id;
#if CONCURRENT
#if USEPTHREADSM
  pthread_mutex_t spinLock;
#else 
  galois::substrate::SimpleLock spinLock;
#endif
#else
  unsigned char spinLock;
#endif 
  //  typedef std::vector<ND*> predTY;
  typedef llvm::SmallVector<ND*, 2> predTY;
  predTY preds;
  int distance;
  int nsuccs;
#if USE_MARKING
  char b;
#endif
  double sigma;
  double delta;
  double bc;

#if USEPTHREADSM
  ND(const int _id) : id(_id), preds(), distance(DEF_DISTANCE), nsuccs(0), sigma(0), delta(0), bc(0)
#if USE_MARKING
                      , b(0) 
#endif
  {
//    std::cerr << "Using pthread-mutex\n";
    pthread_mutex_init(&spinLock, 0);
  }
  ND() : id(DEF_DISTANCE), preds(), distance(DEF_DISTANCE), nsuccs(0), sigma(0), delta(0), bc(0)
#if USE_MARKING
         , b(0)
#endif
         {
//    std::cerr << "Using pthread-mutex\n";
    pthread_mutex_init(&spinLock, 0);
  }
#else 
  ND(const int _id) : id(_id), spinLock(), preds(), distance(DEF_DISTANCE), nsuccs(0), sigma(0), delta(0), bc(0)
#if USE_MARKING
                      , b(0) 
#endif
  {}
  ND() : id(DEF_DISTANCE), spinLock(), preds(), distance(DEF_DISTANCE), nsuccs(0), sigma(0), delta(0), bc(0)
#if USE_MARKING
         , b(0) 
#endif
         {}
#endif 
  
  bool predsContain(const ND* a) const {
    /*int sz = preds.size();
    for (int i=0; i<sz; ++i) {
      if (a == preds[i])
        return true;
    }
    return false;*/
    predTY::const_iterator it = preds.end();
    return (std::find(preds.begin(), it, a) != it); 
  }

  void lock() {
#if CONCURRENT
#if USEPTHREADSM
    pthread_mutex_lock(&spinLock);
#else
    spinLock.lock();
#endif
#endif
  }

  bool try_lock() {
#if CONCURRENT
#if USEPTHREADSM
    pthread_mutex_lock(&spinLock);
    return true;
#else
    return spinLock.try_lock();
#endif
#endif
  }

  void unlock() {
#if CONCURRENT
#if USEPTHREADSM
    pthread_mutex_unlock(&spinLock);
#else
    spinLock.unlock();
#endif
#endif
  }
  
  std::string toString() const {
    std::ostringstream s;
    s << id << " distance: " << distance << " sigma: " << sigma << " bc: " << bc << " nsuccs: " << nsuccs << " npreds: " << preds.size();
    return s.str();
  }

  inline void reset() {
    preds.clear();//resize(0);
    distance = DEF_DISTANCE;
    nsuccs = 0; // Reset flags as follows: inFringe = false, deltaDone = false
    sigma = 0;
    delta = 0;
#if USE_MARKING
    b = 0;
#endif
  }

  void checkClear() const {

    if (!preds.empty() || nsuccs != 0 || sigma != 0 || delta != 0)
      std::cerr << " Problem, node not clear\n";
    assert (preds.empty());
    assert (distance == DEF_DISTANCE);
    assert (nsuccs == 0 && sigma == 0 && delta == 0);
  }  

  void initAsSource() {
    distance = 0;
    sigma = 1;
  }

#if USE_MARKING
  void markOut() {
#if CONCURRENT
    __sync_fetch_and_and(&b, 0);
//    b = 0;
#else
//    b = 0;
#endif
  }

  char isAlreadyIn() {
#if CONCURRENT
    return __sync_fetch_and_or(&b, 1);
/*    char retval = b;
    b = 1;
    return retval;
	*/	
#else
    char retval = b;
    b = 1;
    return retval;
#endif
  }
#endif

};
#endif
