#include <stdio.h>
#include "GraphGenerator.hxx"
#include <iostream>
#include "Galois/Graph/LC_Morph_Graph.h"
#include "Production.h"
#include "GaloisWorker.h"
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

	run<ProductionProcess>();
	return 0;

}
