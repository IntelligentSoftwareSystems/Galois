// -*- C++ -*-

namespace GaloisRuntime {

//Dikstra dual-ring termination algorithm
class TerminationDetection {

  struct tokenHolder {
    volatile long tokenIsBlack;
    volatile long hasToken;
    volatile long processIsBlack;
    tokenHolder() :tokenIsBlack(false), hasToken(false), processIsBlack(true) {}
  };

  PerCPU<tokenHolder> data;
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
