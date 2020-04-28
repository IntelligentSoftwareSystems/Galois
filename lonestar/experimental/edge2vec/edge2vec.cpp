#include "galois/Galois.h"
#include "galois/gstl.h"
#include "galois/Reduction.h"
#include "galois/Timer.h"
#include "galois/Timer.h"
#include "galois/graphs/LCGraph.h"
#include "galois/graphs/TypeTraits.h"
#include "llvm/Support/CommandLine.h"
#include "galois/AtomicHelpers.h"

#include <iostream>
#include <fstream>
#include <deque>
#include <type_traits>

#include <random>
#include <math.h>
#include <algorithm>


#include "Lonestar/BoilerPlate.h"
#include "edge2vec.h"
#include "galois/DynamicBitset.h"

namespace cll = llvm::cl;

static const char* name = "edge2vec";


//function HeteroRandomWalk
void heteroRandomWalk(Graph& graph, std::vector<std::vector<double> >& M, 
galois::InsertBag<std::vector<uint32_t>>& bag, uint32_t walk_length,
double p, double q){

	galois::do_all(galois::iterate(graph),
	[&](GNode n){
		std::vector<uint32_t> walk;
		std::vector<uint32_t> T;

		walk.push_back(n);
		while(walk.size() < walk_length){
		
			if(walk.size() == 1){
				
				uint32_t curr = walk[0];

				//random value between 0 and 1
				double prob = distribution(generator);	

				//sample a neighbor of curr
				uint32_t m, type;
				uint32_t total_wt=0;
				for(auto e: graph.edges(n)){
				
				//	uint32_t v = graph.getEdgeDst(e);
					uint32_t wt = graph.getEdgeData(e).weight;
			
					total_wt += wt;
				}			

				prob = prob*total_wt;
				total_wt = 0;
				for(auto e: graph.edges(n)){
				
					uint32_t wt = graph.getEdgeData(e).weight;
					total_wt +=  wt;

					if(total_wt >= prob){
						m = graph.getEdgeDst(e);
						type = graph.getEdgeData(e).type;
						break;
					}
				}


				walk.push_back(m);
				T.push_back(type);
				
			}
			else{
				uint32_t curr = walk[walk.size()-1];
				uint32_t prev = walk[walk.size()-2];
			
				uint32_t p1 = T.back(); //last element of T

				double total_ew = 0.0f;

				std::vector<double> vec_ew;
				std::vector<uint32_t> neighbors;
				std::vector<uint32_t> types; 

				for(auto k: graph.edges(curr)){
				
					uint32_t p2 = graph.getEdgeData(k).type;
		
					double alpha;
					uint32_t next = graph.getEdgeDst(k);

					if(next == prev)
						alpha = 1.0f/p;
					else if(graph.edges(next).find(prev) != graph.edges(next).end() || graph.edges(prev).find(next) != graph.edges(prev).end() )
						alpha = 1.0f;
					else
						alpha = 1.0f/q;
		
					double ew = M[p1][p2]*((double)graph.getEdgeData(k).weight)*alpha;
	
					total_ew += ew;
					neighbors.push_back(next);
					vec_ew.push_back(ew);
					types.push_back(p2);
				}

				//randomly sample a neighobr of curr	
				//random value between 0 and 1
				double prob = distribution(generator);
				prob = prob*total_ew;
				total_ew = 0.0f;

				//sample a neighbor of curr
			  uint32_t m, type;
			
				int32_t idx = 0;
				for(auto k: neighbors){
					total_ew += vec_ew[idx];
					if(total_ew >= prob){
						m = k;
						type = types[idx];
						break;
					}
					idx++;
				}

				walk.push_back(m);
				T.push_back(type);		
			}//end if-else loop
		}//end while
	});
}

int main(int argc, char** argv) {
  galois::SharedMemSys G;
  LonestarStart(argc, argv, name, desc, url);

	Graph graph;

	std::ifstream f(filename.c_str());
	//read graph
	

	return 0;
}
