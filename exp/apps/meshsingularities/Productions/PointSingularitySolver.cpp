#include <stdio.h>
#include <iostream>

#include "Production.h"
#include "GaloisWorker.h"
#include "GraphGenerator.hxx"

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "Galois/Graph/LC_Morph_Graph.h"

#include "Lonestar/BoilerPlate.h"

#include "llvm/Support/CommandLine.h"

const char* const name = "Mesh singularities";
const char* const desc = "Compute the solution of differential equation";
const char* const url = NULL;

namespace cll = llvm::cl;
static cll::opt<int> nrOfTiers("nrOfTiers", cll::desc("Number of Tiers"), cll::init(16));

template<typename Algorithm>
void run(int nrOfTiers) {
	Galois::StatTimer U(name);
	Algorithm algorithm;
	U.start();
	algorithm(nrOfTiers);
	U.stop();
}

int main(int argc, char** argv)
{
	Galois::StatManager statManager;
 	LonestarStart(argc, argv, name, desc, url);

	run<ProductionProcess>(nrOfTiers);
	return 0;
}
