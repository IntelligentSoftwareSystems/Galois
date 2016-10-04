/** Manipulate the user context -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2016, The University of Texas at Austin. All rights
 * reserved.
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */

#ifndef GALOIS_RUNTIME_USERCONTEXTACCESS_H
#define GALOIS_RUNTIME_USERCONTEXTACCESS_H

#include "Galois/UserContext.h"

namespace Galois {
namespace Runtime {

//! Backdoor to allow runtime methods to access private data in UserContext
template<typename T>
class UserContextAccess : public Galois::UserContext<T> {
public:
  typedef Galois::UserContext<T> SuperTy;
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
} // end namespace Galois

#endif
