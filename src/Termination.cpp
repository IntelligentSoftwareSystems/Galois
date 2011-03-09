//Dikstra dual-ring termination algorithm

#include "Galois/Runtime/PerCPU.h"
#include "Galois/Runtime/Termination.h"
#include "Galois/Runtime/SimpleLock.h"

#include <iostream>

using namespace GaloisRuntime;

TerminationDetection::TerminationDetection()
  :globalTerm(false), lastWasWhite(false)
{
  data.get(0).hasToken = true;
  data.get(0).tokenIsBlack = true;
}

void TerminationDetection::localTermination() {
  tokenHolder& th = data.get();
  tokenHolder& thn = data.getNext();
  //  static SimpleLock<int, true> L;
  //  L.lock();
  //  std::cerr << "th " << &th << " thn " << &thn << "\n";
  //  std::cerr << ThreadPool::getMyID() << "\n";
  //  for (int i = 0; i < 4; ++i) {
  //    tokenHolder& T = data.get(i);
  //    std::cerr << "[" << i << " p " << T.processIsBlack << " t " << T.tokenIsBlack << " h " << T.hasToken << "] ";
  //  }
  //  std::cerr << "\n";
  if (th.hasToken) {
    if (ThreadPool::getMyID() == 1) {
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
