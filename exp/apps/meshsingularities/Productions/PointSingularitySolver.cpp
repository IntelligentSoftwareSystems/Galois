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

template<typename Algorithm>
void run() {
	Algorithm algorithm;

	//Galois::StatTimer T;
	//T.start();
	algorithm();
	//T.stop();
}


int main(int argc, char** argv)
{
	Galois::StatManager statManager;
 	LonestarStart(argc, argv, name, desc, url);

	run<ProductionProcess>();
	return 0;

}
