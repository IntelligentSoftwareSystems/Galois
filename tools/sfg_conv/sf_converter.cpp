/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @author Dimitrios Prountzos <dprountz@cs.utexas.edu>
 */
#include "Galois/Graphs/LCGraph.h"
#include "Galois/Graphs/FileGraph.h"
#include "Galois/Galois.h"
#include "Galois/IO/gr.h"

#include "Galois/Graphs/Serialize.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stack>
#include <queue>
#include <vector>


static const char* name = "RMAT format binary generator";
static const char* description =
  "Reads in an RMAT text file and produces a binary version of the file.\n";
static const char* url = 0;
static const char* help = "<input file> <output file>";

void readRMATFileAndSerialize(const char *infilename, const char *outfilename) {
  Galois::Graph::FirstGraph<int,int,true> inputGr;
  typedef Galois::Graph::FirstGraph<int,int,true>::GraphNode GNode1;

  std::ifstream infile;
  infile.open(infilename, std::ifstream::in); // opens the vector file
  if (!infile) { // file couldn't be opened
    std::cerr << "file " << infilename << " could not be opened" << std::endl;
    abort();
  }

  std::string linebuf;
  int nnodes = 0, nedges = 0, edgeNum = 0;
  bool firstLineAfterComments = true;
  std::vector<GNode1> graphNodes;

  while (!std::getline(infile, linebuf).eof()) {
    //cerr << linebuf << endl;
    if (linebuf[0] == '%')
      continue; // Skip the first lines that start with comments
    if (firstLineAfterComments) {
      std::stringstream(linebuf) >> nnodes >> nedges;
      //cerr << nnodes << " " << nedges << endl;
      for (int i=0; i<nnodes; ++i) {
        GNode1 n = inputGr.createNode(i);
        graphNodes.push_back(n);
        inputGr.addNode(n, Galois::Graph::NONE);
      }
      firstLineAfterComments = false;
      continue;
    }
    int nid, nedg; // current node and number of adjacent edges
    std::stringstream slb(linebuf);
    slb >> nid >> nedg;
    for (int j=0; j<nedg; ++j) {
      edgeNum++;
      int nbrId, edgW; // current neighbor and edge weight
      slb >> nbrId >> edgW;
      inputGr.addEdge(graphNodes[nid], graphNodes[nbrId], Galois::Graph::NONE);
      //std::cerr << "Added edge " << graphNodes[nid].getData(Galois::Graph::NONE) << " -> " << graphNodes[nbrId].getData(Galois::Graph::NONE) << std::endl;
    }
  }
  std::cout << "Finished reading graph " << nnodes 
    << " " << nedges << " "  << edgeNum << "\n";
  infile.close();

  Galois::Graph::LCGraph<int, int>  *lcg = new Galois::Graph::LCGraph<int, int>();
  lcg->createGraph(&inputGr);

  outputGraph(outfilename, inputGr);
}

int main(int argc, const char** argv) {
  std::vector<const char*> args = parse_command_line(argc, argv, help);

  if (args.size() != 2 ) {
    std::cout << "not enough arguments, use -help for usage information\n";
    return 1;
  }
  printBanner(std::cout, name, description, url);

  readRMATFileAndSerialize(args[0], args[1]);

  return 0;
}
