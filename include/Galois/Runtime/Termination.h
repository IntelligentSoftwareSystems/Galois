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

  TerminationDetection();

  inline void workHappened() {
    data.get().processIsBlack = true;
  }

  void localTermination();

  // Returns
  bool globalTermination() {
    return globalTerm;
  }

};

}
