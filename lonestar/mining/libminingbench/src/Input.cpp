#include "MiningBench/Start.h"

cll::opt<std::string> inputFile(cll::Positional,
                                cll::desc("<filename: symmetrized graph>"),
                                cll::Required);
