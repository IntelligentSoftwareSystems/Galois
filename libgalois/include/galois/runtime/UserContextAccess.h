#ifndef GALOIS_RUNTIME_USERCONTEXTACCESS_H
#define GALOIS_RUNTIME_USERCONTEXTACCESS_H

#include "galois/UserContext.h"

namespace galois {
namespace runtime {

//! Backdoor to allow runtime methods to access private data in UserContext
template<typename T>
class UserContextAccess : public galois::UserContext<T> {
public:
  typedef galois::UserContext<T> SuperTy;
  typedef typename SuperTy::PushBufferTy PushBufferTy;
  typedef typename SuperTy::FastPushBack FastPushBack;

  void resetAlloc() { SuperTy::__resetAlloc(); }
  PushBufferTy& getPushBuffer() { return SuperTy::__getPushBuffer(); }
  void resetPushBuffer() { SuperTy::__resetPushBuffer(); }
  SuperTy& data() { return *static_cast<SuperTy*>(this); }
  void setLocalState(void *p) { SuperTy::__setLocalState(p); }
  void setFastPushBack(FastPushBack f) { SuperTy::__setFastPushBack(f); }
  void setBreakFlag(bool *b) { SuperTy::didBreak = b; }

  void setFirstPass (void) { SuperTy::__setFirstPass(); }
  void resetFirstPass (void) { SuperTy::__resetFirstPass(); }

// TODO: move to a separate class dedicated for speculative executors
#ifdef GALOIS_USE_EXP
  void rollback() { SuperTy::__rollback (); }

  void commit() { SuperTy::__commit (); }

  void reset() {
    SuperTy::__resetPushBuffer();
    SuperTy::__resetUndoLog();
    SuperTy::__resetCommitLog();
    SuperTy::__resetAlloc();
  }
#endif
};

}
} // end namespace galois

#endif
