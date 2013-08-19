#include <stdio.h>
#include "GraphGenerator.hxx"
#include <iostream>
#include "Galois/Graph/LC_Morph_Graph.h"
#include "Production.h"
#include "GaloisWorker.h"

#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Bag.h"
#include "Galois/Statistic.h"
#include "Galois/UnionFind.h"
#include "Galois/Graph/LCGraph.h"
#include "Galois/ParallelSTL/ParallelSTL.h"
#include "llvm/Support/CommandLine.h"
#include "Lonestar/BoilerPlate.h"

const char* name = "Mesh singularities";
const char* desc = "Compute the solution of differential equation";
const char* url = NULL;

namespace cll = llvm::cl;
static cll::opt<int> nrOfTiers("nrOfTiers", cll::desc("Number of Tiers"), cll::init(16));

template<typename Algorithm>
void run(int nrOfTiers) {
	Algorithm algorithm;
	algorithm(nrOfTiers);
}

int main(int argc, char** argv)
{
	Galois::StatManager statManager;
 	LonestarStart(argc, argv, name, desc, url);

	run<ProductionProcess>(nrOfTiers);
	return 0;
}
