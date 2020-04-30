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

static cll::opt<std::string> filename(cll::Positional,
                                      cll::desc("<input file>"), cll::Required);
static cll::opt<uint32_t> N("N",
  cll::desc("Number of iterations"),
  cll::init(10));

static cll::opt<uint32_t> walk_length("Walk Length",
  cll::desc("Length of random walk"),
  cll::init(50));

static cll::opt<double> p("Walk Length",
  cll::desc("Length of random walk"),
  cll::init(50));

void computeVectors(std::vector<std::vector<uint32_t>>& v, galois::InsertBag<std::vector<uint32_t>>& walks, uint32_t num_edge_types){

	galois::InsertBag<std::vector<uint32_t>> bag;
	galois::do_all(galois::iterate(walks),
	[&](std::vector<uint32_t>& walk){
	
		std::vector<uint32_t> vec(num_edge_types+1, 0);
		
		for(auto type:walk)
			vec[type]++;

		bag.push(vec);
	});	

	for(auto vec:bag)
		v.push_back(vec); 
}

double sigmoid(double pears) {
    return 1 / (1 + exp(-pears)); //exact sig
    //return (pears / (1 + abs(pears))); //fast sigmoid
    
}

double pearsonCorr(uint32_t i, uint32_t j, std::vector<std::vector<uint32_t>>& v){
    int sum_x = 0, sum_y = 0, sum_xy = 0, squareSum_x = 0, squareSum_y = 0;
    std::vector<uint32_t> x = v[i];
    std::vector<uint32_t> y = v[j];
    for (uint32_t m = 0; m < x.size(); m++) 
    { 
        sum_x = sum_x + x.at(m); 
        sum_y = sum_y + y.at(m); 
        sum_xy = sum_xy + x.at(m) * y.at(m); 
        squareSum_x = squareSum_x + x.at(m) * x.at(m); 
        squareSum_x = squareSum_y + y.at(m) * y.at(m); 
    } 
  
    double corr = (double)(x.size() * sum_xy - sum_x * sum_y)  
                  / sqrt((n * squareSum_x - sum_x * sum_x)  
                      * (n * squareSum_y - sum_y * sum_y)); 
    return corr;
	
}

void computeM(std::vector<std::vector<uint32_t>>& v, std::vector<std::vector<double> >& M,uint32_t num_edge_types){

	galois::do_all(galois::iterate((uint32_t) 1, num_edge_types+1),
	[&](uint32_t i){

		for(uint32_t j=1;j<=num_edge_types;j++){
			
			double pearson_corr = pearsonCorr(i,j, v);
			double sigmoid = sigmoid(pearson_corr);

			M[i][j] = sigmoid;
		}
	});
}

//function generateTransitionMatrix
//M should have all entries set to 1
void generateTransitionMatrix(Graph& graph, std::vector<std::vector<double> >& M, uint32_t N,
uint32_t walk_length,
double p, double q, uint32_t num_edge_types){

	while(N>0){
		N--;
		
		//E step; generate walks
		galois::InsertBag<std::vector<uint32_t>> walks;
		heteroRandomWalk(graph, M, walks, walk_length, p, q);
		
		//M step
		uint32_t size = std::distance(walks.begin(), walks.end());
		std::vector<std::vector<uint32_t>> v;
		computeVectors(v, walks, num_edge_types);

		computeM(v, M);		
	}
}

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

  uint32_t nodes;
	uint32_t edges;

	std::string line;
  std::getline(f, line);
  std::stringstream ss(line);

	ss >> edges >> nodes;

	std::vector<std::vector<uint32_t> > edges_id(nodes);
  std::vector<std::vector<EdgeTy> > edges_data(nodes);
  std::vector<uint32_t> prefix_edges(nodes);

	uint64_t max_type = 0;

	while(std::getline(f, line)){
	
		std::stringstream ss(line);
		uint32_t src, dst, type;
		ss >> src >> dst >> type;

		edges_id[src-1].push_back(dst-1);
		EdgeTy edgeTy;

		edgeTy.weight = 1;
		edgeTy.type = type;

		if(type > max_type)
			max_type = type;

		edges_data[src-1].push_back(edgeTy);
	}

	f.close();

	galois::do_all(galois::iterate(uint32_t{0}, nodes),
    [&](uint32_t c){
      prefix_edges[c] = edges_id[c].size();
  });
	
	for (uint32_t c = 1; c < nodes; ++c) {
    prefix_edges[c] += prefix_edges[c - 1];
  }	

	graph.constructFrom(nodes, edges, prefix_edges, edges_id, edges_data);

	//transition matrix
	std::vector<std::vector<double>> M(max_type+1);

	//initialize transition matrix
	for(uint32_t i=0;i<=max_type;i++){
		for(uint32_t j=0;j<=max_type;j++)
			M[i].push_back(1.0f);
	}	
	
	generateTransitionMatrix(graph, M, N, walk_length,
double p, double q, uint32_t num_edge_types)		
	return 0;
}
