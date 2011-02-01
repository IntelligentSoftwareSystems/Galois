// -*- C++ -*-

namespace GaloisRuntime {


//Dikstra dual-ring termination algorithm
class TerminationDetection {

  struct tokenHolder {
    volatile bool tokenIsBlack;
    volatile bool hasToken;
    volatile bool processIsBlack;
    tokenHolder() :tokenIsBlack(false), hasToken(false), processIsBlack(true) {}
  };

  CPUSpaced<tokenHolder> data;
  volatile bool globalTerm;
  bool lastWasWhite;
public:

  TerminationDetection()
    :data(0), globalTerm(false), lastWasWhite(false)
  {
    data.get(0).hasToken = true;
  }

  void workHappened() {
    data.get().processIsBlack = true;
  }

  void localTermination() {
    tokenHolder& th = data.get();
    tokenHolder& thn = data.getNext();
    if (ThreadPool::getMyID() == 1) {
      //master
      if (th.hasToken) {
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
	//Do nothing while waiting for the token
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

  bool globalTermination() {
    return globalTerm;
  }

};

}
