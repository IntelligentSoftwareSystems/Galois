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

using namespace Galois::Runtime;

TerminationDetection::TerminationDetection()
  :globalTerm(false), lastWasWhite(false)
{
  initializeThread();
}

void TerminationDetection::reset() {
  assert(data.getLocal()->hasToken);
  globalTerm = false;
  lastWasWhite = false;
  data.getLocal()->hasToken = true;
  data.getLocal()->tokenIsBlack = true;
}

void TerminationDetection::propToken(TokenHolder& th, TokenHolder& thn) {
  assert (!globalTerm && "no token should be in progress after globalTerm");
  bool taint = th.processIsBlack || th.tokenIsBlack;
  th.processIsBlack = th.tokenIsBlack = false;
  th.hasToken = false;
  thn.tokenIsBlack = taint;
  __sync_synchronize();
  thn.hasToken = true;
}

void TerminationDetection::localTermination() {
  unsigned myID = LL::getTID();
  TokenHolder& th = *data.getLocal();
  if (th.hasToken) {
    TokenHolder& thn = *data.getRemote((myID + 1) % activeThreads);
    if (myID == 0) {
      //master
      bool failed = th.tokenIsBlack || th.processIsBlack;
      th.tokenIsBlack = th.processIsBlack = false;
      if (lastWasWhite && !failed) {
	//This was the second success
	globalTerm = true;
	return;
      }
      lastWasWhite = !failed;
    }
    //Normal thread or recirc by master
    propToken(th,thn);
  }
  //  L.unlock();
}
