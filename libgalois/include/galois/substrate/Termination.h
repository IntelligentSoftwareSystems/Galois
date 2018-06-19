/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef GALOIS_SUBSTRATE_TERMINATION_H
#define GALOIS_SUBSTRATE_TERMINATION_H

#include "galois/substrate/PerThreadStorage.h"
#include "galois/substrate/CacheLineStorage.h"

#include <atomic>

namespace galois {
namespace substrate {

class TerminationDetection;
/*
 * returns an object.  The object will be reused, but reinitialized to
 * activeThreads
 */
TerminationDetection& getSystemTermination(unsigned activeThreads);

class TerminationDetection {

  friend TerminationDetection& getSystemTermination(unsigned);

protected:
  CacheLineStorage<std::atomic<int>> globalTerm;

  /**
   * for internal use by child classes
   */
  virtual void init(unsigned activeThreads) = 0;

public:
  virtual ~TerminationDetection(void);
  /**
   * Initializes the per-thread state.  All threads must call this
   * before any call localTermination.
   */
  virtual void initializeThread() = 0;

  /**
   * Process termination locally.  May be called as often as needed.  The
   * argument workHappened signals that since last time it was called, some
   * progress was made that should prevent termination. All threads must call
   * initializeThread() before any thread calls this function.  This function
   * should not be on the fast path (this is why it takes a flag, to allow the
   * caller to buffer up work status changes).
   */
  virtual void localTermination(bool workHappened) = 0;

  /**
   * Returns whether global termination is detected.
   */
  bool globalTermination() const { return globalTerm.data; }
};

namespace internal {
// Dijkstra style 2-pass ring termination detection
template <typename _UNUSED = void>
class LocalTerminationDetection : public TerminationDetection {

  struct TokenHolder {
    friend class TerminationDetection;
    std::atomic<long> tokenIsBlack;
    std::atomic<long> hasToken;
    long processIsBlack;
    bool lastWasWhite; // only used by the master
  };

  galois::substrate::PerThreadStorage<TokenHolder> data;

  unsigned activeThreads;

  // send token onwards
  void propToken(bool isBlack) {
    unsigned id     = ThreadPool::getTID();
    TokenHolder& th = *data.getRemote((id + 1) % activeThreads);
    th.tokenIsBlack = isBlack;
    th.hasToken     = true;
  }

  void propGlobalTerm() { globalTerm = true; }

  bool isSysMaster() const { return ThreadPool::getTID() == 0; }

protected:
  virtual void init(unsigned aThreads) { activeThreads = aThreads; }

public:
  LocalTerminationDetection() {}

  virtual void initializeThread() {
    TokenHolder& th   = *data.getLocal();
    th.tokenIsBlack   = false;
    th.processIsBlack = true;
    th.lastWasWhite   = true;
    globalTerm        = false;
    if (isSysMaster())
      th.hasToken = true;
    else
      th.hasToken = false;
  }

  virtual void localTermination(bool workHappened) {
    assert(!(workHappened && globalTerm.get()));
    TokenHolder& th = *data.getLocal();
    th.processIsBlack |= workHappened;
    if (th.hasToken) {
      if (isSysMaster()) {
        bool failed     = th.tokenIsBlack || th.processIsBlack;
        th.tokenIsBlack = th.processIsBlack = false;
        if (th.lastWasWhite && !failed) {
          // This was the second success
          propGlobalTerm();
          return;
        }
        th.lastWasWhite = !failed;
      }
      // Normal thread or recirc by master
      assert(!globalTerm.get() &&
             "no token should be in progress after globalTerm");
      bool taint        = th.processIsBlack || th.tokenIsBlack;
      th.processIsBlack = th.tokenIsBlack = false;
      th.hasToken                         = false;
      propToken(taint);
    }
  }
};

// Dijkstra style 2-pass tree termination detection
template <typename _UNUSED = void>
class TreeTerminationDetection : public TerminationDetection {
  static const int num = 2;

  struct TokenHolder {
    friend class TerminationDetection;
    // incoming from above
    volatile long down_token;
    // incoming from below
    volatile long up_token[num];
    // my state
    long processIsBlack;
    bool hasToken;
    bool lastWasWhite; // only used by the master
    int parent;
    int parent_offset;
    TokenHolder* child[num];
  };

  PerThreadStorage<TokenHolder> data;

  unsigned activeThreads;

  void processToken() {
    TokenHolder& th = *data.getLocal();
    // int myid = LL::getTID();
    // have all up tokens?
    bool haveAll = th.hasToken;
    bool black   = th.processIsBlack;
    for (int i = 0; i < num; ++i) {
      if (th.child[i]) {
        if (th.up_token[i] == -1)
          haveAll = false;
        else
          black |= th.up_token[i];
      }
    }
    // Have the tokens, propagate
    if (haveAll) {
      th.processIsBlack = false;
      th.hasToken       = false;
      if (isSysMaster()) {
        if (th.lastWasWhite && !black) {
          // This was the second success
          propGlobalTerm();
          return;
        }
        th.lastWasWhite = !black;
        th.down_token   = true;
      } else {
        data.getRemote(th.parent)->up_token[th.parent_offset] = black;
      }
    }

    // recieved a down token, propagate
    if (th.down_token) {
      th.down_token = false;
      th.hasToken   = true;
      for (int i = 0; i < num; ++i) {
        th.up_token[i] = -1;
        if (th.child[i])
          th.child[i]->down_token = true;
      }
    }
  }

  void propGlobalTerm() { globalTerm = true; }

  bool isSysMaster() const { return ThreadPool::getTID() == 0; }

protected:
  virtual void init(unsigned aThreads) { activeThreads = aThreads; }

public:
  TreeTerminationDetection() {}

  virtual void initializeThread() {
    TokenHolder& th = *data.getLocal();
    th.down_token   = false;
    for (int i = 0; i < num; ++i)
      th.up_token[i] = false;
    th.processIsBlack = true;
    th.hasToken       = false;
    th.lastWasWhite   = false;
    globalTerm        = false;
    auto tid          = ThreadPool::getTID();
    th.parent         = (tid - 1) / num;
    th.parent_offset  = (tid - 1) % num;
    for (unsigned i = 0; i < num; ++i) {
      unsigned cn = tid * num + i + 1;
      if (cn < activeThreads)
        th.child[i] = data.getRemote(cn);
      else
        th.child[i] = 0;
    }
    if (isSysMaster()) {
      th.down_token = true;
    }
  }

  virtual void localTermination(bool workHappened) {
    assert(!(workHappened && globalTerm.get()));
    TokenHolder& th = *data.getLocal();
    th.processIsBlack |= workHappened;
    processToken();
  }
};

void setTermDetect(TerminationDetection* term);
} // end namespace internal

} // namespace substrate
} // end namespace galois

#endif
