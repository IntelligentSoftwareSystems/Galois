/** Dikstra style termination detection -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.
 *
 * @section Description
 *
 * Implementation of Dikstra dual-ring Termination Detection
 *
 * @author Andrew Lenharth <andrewl@lenharth.org>
 */

#include "Galois/Runtime/ActiveThreads.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

using namespace Galois::Runtime;

namespace {
//Dijkstra style 2-pass ring termination detection
class LocalTerminationDetection : public TerminationDetection {
  struct TokenHolder {
    friend class TerminationDetection;
    volatile long tokenIsBlack;
    volatile long hasToken;
    long processIsBlack;
    bool lastWasWhite; // only used by the master
  };

  PerThreadStorage<TokenHolder> data;
  
  //send token onwards
  void propToken(bool isBlack) {
    unsigned id = LL::getTID();
    TokenHolder& th = *data.getRemote((id + 1) % activeThreads);
    th.tokenIsBlack = isBlack;
    LL::compilerBarrier();
    th.hasToken = true;
  }

  void propGlobalTerm() {
    globalTerm.data = true;
  }

  bool isSysMaster() const {
    return LL::getTID() == 0;
  }

public:
  LocalTerminationDetection() {}

  virtual void initializeThread() {
    TokenHolder& th = *data.getLocal();
    th.hasToken = false;
    th.tokenIsBlack = false;
    th.processIsBlack = true;
    th.lastWasWhite = true;
    globalTerm.data = false;
    if (isSysMaster()) {
      th.hasToken = true;
    }
  }

  virtual void localTermination(bool workHappened) {
    assert(!(workHappened && globalTerm.data));
    TokenHolder& th = *data.getLocal();
    th.processIsBlack |= workHappened;
    if (th.hasToken) {
      if (isSysMaster()) {
	bool failed = th.tokenIsBlack || th.processIsBlack;
	th.tokenIsBlack = th.processIsBlack = false;
	if (th.lastWasWhite && !failed) {
	  //This was the second success
	  propGlobalTerm();
	  return;
	}
	th.lastWasWhite = !failed;
      }
      //Normal thread or recirc by master
      assert (!globalTerm.data && "no token should be in progress after globalTerm");
      bool taint = th.processIsBlack || th.tokenIsBlack;
      th.processIsBlack = th.tokenIsBlack = false;
      th.hasToken = false;
      propToken(taint);
    }
  }
};

static LocalTerminationDetection& getLocalTermination() {
  static LocalTerminationDetection term;
  return term;
}


//Dijkstra style 2-pass tree termination detection
class TreeTerminationDetection : public TerminationDetection {
  static const int num = 2;

  struct TokenHolder {
    friend class TerminationDetection;
    //incoming from above
    volatile long down_token;
    //incoming from below
    volatile long up_token[num];
    //my state
    long processIsBlack;
    bool hasToken;
    bool lastWasWhite; // only used by the master
    int parent;
    int parent_offset;
    TokenHolder* child[num];
  };

  PerThreadStorage<TokenHolder> data;

  void processToken() {
    TokenHolder& th = *data.getLocal();
    //int myid = LL::getTID();
    //have all up tokens?
    bool haveAll = th.hasToken;
    bool black = th.processIsBlack;
    for (int i = 0; i < num; ++i) {
      if (th.child[i]) {
	if( th.up_token[i] == -1 )
	  haveAll = false;
	else
	  black |= th.up_token[i];
      }
    }
    //Have the tokens, propagate
    if (haveAll) {
      th.processIsBlack = false;
      th.hasToken = false;
      if (isSysMaster()) {
	if (th.lastWasWhite && !black) {
	  //This was the second success
	  propGlobalTerm();
	  return;
	}
	th.lastWasWhite = !black;
	th.down_token = true;
      } else {
	data.getRemote(th.parent)->up_token[th.parent_offset] = black;
      }
    }

    //recieved a down token, propagate
    if (th.down_token) {
      th.down_token = false;
      th.hasToken = true;
      for (int i = 0; i < num; ++i) {
	th.up_token[i] = -1;
	if (th.child[i])
	  th.child[i]->down_token = true;
      }
    }
  }

  void propGlobalTerm() {
    globalTerm.data = true;
  }

  bool isSysMaster() const {
    return LL::getTID() == 0;
  }

public:
  TreeTerminationDetection() {}

  virtual void initializeThread() {
    TokenHolder& th = *data.getLocal();
    th.down_token = false;
    for (int i = 0; i < num; ++i) 
      th.up_token[i] = false;
    th.processIsBlack = true;
    th.hasToken = false;
    th.lastWasWhite = false;
    globalTerm.data = false;
    th.parent = (LL::getTID() - 1) / num;
    th.parent_offset = (LL::getTID() - 1) % num;
    for (int i = 0; i < num; ++i) {
      int cn = LL::getTID() * num + i + 1;
      if (cn < (int) activeThreads)
	th.child[i] = data.getRemote(cn);
      else
	th.child[i] = 0;
    }
    if (isSysMaster()) {
      th.down_token = true;
    }
  }

  virtual void localTermination(bool workHappened) {
    assert(!(workHappened && globalTerm.data));
    TokenHolder& th = *data.getLocal();
    th.processIsBlack |= workHappened;
    processToken();
  }
};

__attribute__((unused))
static TreeTerminationDetection& getTreeTermination() {
  static TreeTerminationDetection term;
  return term;
}

} // namespace

Galois::Runtime::TerminationDetection& Galois::Runtime::getSystemTermination() {
  return getLocalTermination();
  //return getTreeTermination();
}

