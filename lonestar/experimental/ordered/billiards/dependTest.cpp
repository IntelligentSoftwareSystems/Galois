#include "dependTest.h"

namespace cll = llvm::cl;
cll::opt<unsigned> vmaxFactor("vmax",
                              cll::desc("upper bound on the velocity as a "
                                        "multiple of maximum initial velocity"),
                              cll::init(5));
