#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>

#include "galois/graphs/FileGraph.h"
#include "galois/graphs/OfflineGraph.h"

static const char* const name = "add random edge weights";
static const char* const desc = "Utility to add random edge weights for edge-list/dimacs graphs on disk";
static const char* const url = 0;

namespace cll = llvm::cl;
static cll::opt<std::string> inputFile(cll::Positional, cll::desc("<input file>"), cll::Required);
static cll::opt<unsigned int> minWt("min", cll::desc("Minimum edge weight"), cll::init(1));
static cll::opt<unsigned int> maxWt("max", cll::desc("Maximum edge weight"), cll::init(100));
static cll::opt<std::string> outputFile(cll::Positional, cll::desc("Name of the output file graph."), cll::Required);


int main(int argc, char** argv) {
   LonestarStart(argc, argv, name, desc, url);
   galois::Timer T_total;
   typedef uint64_t NodeIDType;
   {
      std::ifstream infile (inputFile);
      std::ofstream outfile (outputFile);
      NodeIDType src, dst;
      std::mt19937 gen;
      std::uniform_int_distribution<unsigned int> dist(minWt, maxWt);
      NodeIDType numEdges=0, numNodes=0;
      T_total.start();
      while(infile){
         if(numEdges%10000000==0)fprintf(stdout, "\r%15u",numEdges);
         infile>>src>>dst;
         numEdges++;
         NodeIDType cmax =std::max(dst, src);
         numNodes=std::max(numNodes, cmax);
         unsigned int wt= dist(gen);
         outfile<<src<<" "<<dst <<" " << wt<<std::endl;
      }
      T_total.stop();
      std::cout<<"\nDone\nNodes= "<<numNodes<<"\nEdge = " << numEdges<<"\n";
      std::cout<<"Time= "<<T_total.get()<<"\n";
      infile.close();
      outfile.close();

   }
   std::cout << "Completed edge-weight-add.\n";
   return 0;
}

