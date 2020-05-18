#include "galois/Galois.h"
#include "galois/Bag.h"
#include "galois/UserContext.h"
#include "galois/substrate/PerThreadStorage.h"
#include "Lonestar/BoilerPlate.h"

#include <iostream>
#include <fstream>

class ExampleWrappedWorklist {
private:
  galois::InsertBag<int> bag;
  galois::substrate::PerThreadStorage<galois::UserContext<int>*> ctxPtr;
  bool inParallelPhase;

private:
  void reset() {
    bag.clear();
    for (unsigned i = 0; i < ctxPtr.size(); i++) {
      *(ctxPtr.getRemote(i)) = nullptr;
    }
  }

public:
  ExampleWrappedWorklist() : inParallelPhase(false) { reset(); }

  void enqueue(int item) {
    if (inParallelPhase) {
      (*(ctxPtr.getLocal()))->push(item);
    } else {
      bag.push(item);
    }
  }

  void execute() {
    inParallelPhase = true;

    galois::for_each(
        galois::iterate(bag),
        [&](int item, auto& ctx) {
          if (nullptr == *(ctxPtr.getLocal())) {
            *(ctxPtr.getLocal()) = &ctx;
          }

          std::cout << item << std::endl;

          if (item < 2000) {
            this->enqueue(item + item);
          }
        },
        galois::loopname("execute"), galois::no_conflicts());

    inParallelPhase = false;
    reset();
  }
};

int main(int argc, char* argv[]) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, nullptr, nullptr, nullptr);

  ExampleWrappedWorklist q;
  for (unsigned i = 0; i < galois::getActiveThreads(); i++) {
    q.enqueue(i + 1);
  }
  q.execute();

  return 0;
}
