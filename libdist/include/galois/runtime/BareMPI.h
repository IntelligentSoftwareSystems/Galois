#pragma once
#ifdef __GALOIS_BARE_MPI_COMMUNICATION__
#include "mpi.h"
#include "llvm/Support/CommandLine.h"

enum BareMPI { noBareMPI, nonBlockingBareMPI, oneSidedBareMPI };

extern llvm::cl::opt<BareMPI> bare_mpi;
#endif
