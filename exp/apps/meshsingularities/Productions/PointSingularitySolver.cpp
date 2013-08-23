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

#include "Functions.h"

const char* const name = "Mesh singularities";
const char* const desc = "Compute the solution of differential equation";
const char* const url = NULL;

namespace cll = llvm::cl;

static cll::opt<int> tiers("tiers", cll::desc("Number of Tiers"), cll::init(16));
static cll::opt<int> dimensions("dimensions", cll::desc("Dimensions"), cll::init(3));

static cll::opt<Functions> function("function", cll::desc("Choose a function:"),
    cll::values(
      clEnumVal(F1, "f1(x, [y, [z]]) = 1.0"),
      clEnumVal(F2, "f2(x, [y, [z]]) = x+[y+[z]]"),
      clEnumVal(F3, "f3(x, [y, [z]]) = x^2+[y^2+[z^2]]"),
      clEnumVal(F4, "f4(x, [y, [z]]) = x^3+[y^3+[z^3]]"),
      clEnumValEnd), cll::init(F1));

static cll::opt<double> coord_x("coord_x", cll::desc("X coord for left-bottom vertex of shape"), cll::init(0.0));
static cll::opt<double> coord_y("coord_y", cll::desc("Y coord for left-bottom vertex of shape"), cll::init(0.0));
static cll::opt<double> coord_z("coord_z", cll::desc("Z coord for left-bottom vertex of shape"), cll::init(0.0));

static cll::opt<double> size("size", cll::desc("Size of the shape"), cll::init(1.0));
static cll::opt<bool> performTests("performTests", cll::desc("Run extended tests of computed solution"), cll::init(true));

#define WITH_MUMPS_ENABLED 0

#ifdef WITH_MUMPS_ENABLED
static cll::opt<bool> mumps("mumps", cll::desc("Pass data to MUMPS"), cll::init(false));
#endif

template<typename Algorithm>
std::vector<double> *run(TaskDescription &td) {
	Galois::StatTimer U(name);
	Algorithm algorithm;
	U.start();
	std::vector<double> *res = algorithm(td);
	U.stop();
	return res;
}

#ifdef WITH_MUMPS_ENABLED
extern int execute_mumps(int argc, char** argv, TaskDescription &taskDescription);
#endif

int main(int argc, char** argv)
{
	Galois::StatManager statManager;
 	LonestarStart(argc, argv, name, desc, url);

 	TaskDescription taskDescription;

 	taskDescription.dimensions = dimensions;
 	taskDescription.function = functionsTable[function].func;
 	taskDescription.nrOfTiers = tiers;
 	taskDescription.size = size;
 	taskDescription.x = coord_x;
 	taskDescription.y = coord_y;
 	taskDescription.z = coord_z;

 	taskDescription.performTests = performTests;

#ifdef WITH_MUMPS_ENABLED
 	if (mumps)
 		execute_mumps(argc, argv, taskDescription);
 	else
#endif
 	{
	std::vector<double> *result = run<ProductionProcess>(taskDescription);

 	delete result;
 	}
	return 0;
}
