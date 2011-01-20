// -*- C++ -*-

namespace GaloisRuntime {

class TerminationDetection {
  volatile unsigned int masterToken;
  unsigned int allUnchangedToken;

  bool AllDone;

  SimpleLock<int,true> Lock;

  struct pd {
    unsigned int* counter;
    unsigned int counterOld;
    unsigned int token;
    unsigned int changed;
    pd() : counter(0), counterOld(0), token(0), changed(true) {}
    static void merge(pd& lhs, pd& rhs) {}
  };
  CPUSpaced<pd> Acks;

  bool allAckedCurrent() {
    for (int i = 1; i < Acks.getCount(); ++i)
      if (Acks.getRemote(i).token != masterToken)
	return false;
    return true;
  }

  bool allUnChanged() {
    for (int i = 1; i < Acks.getCount(); ++i)
      if (Acks.getRemote(i).changed)
	return false;
    return true;
  }


  //Query the relivent children for doneness
  void masterQuery() {
    //Let only one thread do this at a time, but don't care who it is
    if (Lock.try_lock(1)) {
      if (allAckedCurrent()) {
	//Everyone has acked the current token
	//Cases:
	if (allUnChanged()) {
	  //No one changed durring the last round
	  //This could either be coincidence (and we need to send another token)
	  //or this was the second token
	  if (allUnchangedToken == masterToken) {
	    AllDone = true;
	  } else {
	    //initiate another round
	    ++masterToken;
	    allUnchangedToken = masterToken;
	  }
	} else {
	  //Someone changed, issue a new token
	  ++masterToken;
	}
      }
    }
  }

public:

  TerminationDetection() :masterToken(1), allUnchangedToken(0), AllDone(false), Acks(pd::merge) {}

  bool areWeThereYet() {
    return AllDone;
  }

  void locallyDone() {
    pd& ack = Acks.get();
    unsigned int mToken = masterToken;
    if (ack.token != mToken) {
      //The master has put out a new query
      //Ack it
      //Order matters:
      unsigned int newcounter = *ack.counter;
      ack.changed = ack.counterOld != newcounter;
      ack.counterOld = newcounter;
      __sync_synchronize ();
      ack.token = mToken;
    }

    masterQuery();
  }

  void initialize(unsigned int* count) {
    Acks.get().counter = count;
  }
};

}
