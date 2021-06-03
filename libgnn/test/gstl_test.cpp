#include "galois/Galois.h"
#include "galois/gstl.h"

int main(int argc, char* argv[]) {
  galois::SharedMemSys G;
  if (argc != 2) {
    printf("Thread arg not specified\n");
    exit(1);
  }
  galois::setActiveThreads(std::stoi(argv[1]));
  printf("Initialized Galois Shared Mem with %u threads\n",
         galois::getActiveThreads());

  // std vector has no leak issues
  using VecType = galois::gstl::Vector<float>;
  // using VecType = std::vector<float>;

  for (size_t i = 0; i < 1000000; i++) {
    if (i % 10000 == 0)
      galois::gPrint("Current is ", i, "\n");
    size_t how_many = 100000;

    std::vector<VecType> carrier;
    carrier.resize(how_many);

    galois::do_all(galois::iterate(size_t{0}, how_many), [&](size_t iter) {
      // allocate some vector then do something with it
      VecType dummy_vec(16);
      for (unsigned j = 0; j < dummy_vec.size(); j++) {
        dummy_vec[j] = j;
      }
      carrier[iter].swap(dummy_vec);
    });

    galois::do_all(galois::iterate(size_t{0}, how_many), [&](size_t iter) {
      VecType to_swap;
      carrier[iter].swap(to_swap);
    });
  }

  return 0;
}
