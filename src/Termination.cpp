/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

//Dikstra dual-ring termination algorithm

#include "Galois/Threads.h"
#include "Galois/Runtime/Termination.h"

using namespace GaloisRuntime;

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

void TerminationDetection::localTermination() {
  unsigned myID = LL::getTID();
  TokenHolder& th = *data.getLocal();
  if (th.hasToken) {
    TokenHolder& thn = *data.getRemote((myID + 1) % Galois::getActiveThreads());
    if (myID == 0) {
      //master
      if (th.tokenIsBlack || th.processIsBlack) {
	//failed circulation
	lastWasWhite = false;
	th.processIsBlack = false;
	th.hasToken = false;
	thn.tokenIsBlack = false;
	__sync_synchronize();
	thn.hasToken = true;
      } else {
	if (lastWasWhite) {
	  //This was the second time around
	  globalTerm = true;
	} else {
	  //Start a second round of voting
	  lastWasWhite = true;
	  th.hasToken = false;
	  thn.tokenIsBlack = false;
	  __sync_synchronize();
	  thn.hasToken = true;
	}
      }
    } else {
      //Normal thread
      if (th.processIsBlack) {
	//Black process colors the token
	//color resets to white
	th.processIsBlack = false;
	th.tokenIsBlack = false;
	th.hasToken = false;
	thn.tokenIsBlack = true;
	__sync_synchronize();
	thn.hasToken = true;
      } else {
	//white process pass the token
	thn.tokenIsBlack = th.tokenIsBlack;
	th.hasToken = false;
	__sync_synchronize();
	thn.hasToken = true;
      }
    }
  }
  //  L.unlock();
}
