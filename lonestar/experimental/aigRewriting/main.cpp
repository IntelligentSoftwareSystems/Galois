/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#include "parsers/AigParser.h"
#include "writers/AigWriter.h"
#include "subjectgraph/aig/Aig.h"
#include "algorithms/CutManager.h"
#include "algorithms/NPNManager.h"
#include "algorithms/RewriteManager.h"
#include "algorithms/PreCompGraphManager.h"
#include "algorithms/ReconvDrivenCut.h"
#include "galois/Galois.h"
#include <chrono>
#include <iostream>
#include <sstream>

using namespace std::chrono;

void aigRewriting(aig::Aig& aig, std::string& fileName, int nThreads,
                  int verbose);
void kcut(aig::Aig& aig, std::string& fileName, int nThreads, int verbose);
void rdCut(aig::Aig& aig, std::string& fileName, int nThreads, int verbose);
std::string getFileName(std::string path);

int main(int argc, char* argv[]) {

  galois::SharedMemSys
      G; // shared-memory system object initializes global variables for galois

  if (argc < 3) {
    std::cout << "Mandatory arguments: <nThreads> <AigInputFile>" << std::endl;
    std::cout << "Optional arguments: -v (verrbose)" << std::endl;
    exit(1);
  }

  int nThreads = atoi(argv[1]);

  std::string path(argv[2]);
  std::string fileName = getFileName(path);

  aig::Aig aig;
  AigParser aigParser(path, aig);
  aigParser.parseAig();
  // aigParser.parseAag();

  int verbose = 0;
  if (argc > 3) {
    std::string verbosity(argv[3]);
    if (verbosity.compare("-v") == 0) {
      verbose = 1;
    }
  }

  if (verbose == 1) {
    std::cout << "############## AIG REWRITING ##############" << std::endl;
    std::cout << "Design Name: " << fileName << std::endl;
    std::cout << "|Nodes|: " << aig.getGraph().size() << std::endl;
    std::cout << "|I|: " << aigParser.getI() << std::endl;
    std::cout << "|L|: " << aigParser.getL() << std::endl;
    std::cout << "|O|: " << aigParser.getO() << std::endl;
    std::cout << "|A|: " << aigParser.getA() << std::endl;
    std::cout << "|E|: " << aigParser.getE() << " (outgoing edges)"
              << std::endl;
  }

  // std::vector< int > levelHistogram = aigParser.getLevelHistogram();
  // int i = 0;
  // for( int value : levelHistogram ) {
  //	std::cout << i++ << ": " << value << std::endl;
  //}

  aigRewriting(aig, fileName, nThreads, verbose);

  // kcut( aig, fileName, nThreads, verbose );

  // rdCut( aig, fileName, nThreads, verbose );

  return 0;
}

void aigRewriting(aig::Aig& aig, std::string& fileName, int nThreads,
                  int verbose) {

  int numThreads = galois::setActiveThreads(nThreads);

  int K = 4, C = 500;
  int triesNGraphs = 500;
  bool compTruth   = true;
  bool useZeros    = false;
  bool updateLevel = false;

  if (verbose == 1) {
    std::cout << "############# Configurations ############## " << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "C: " << C << std::endl;
    std::cout << "TriesNGraphs: " << triesNGraphs << std::endl;
    std::cout << "CompTruth: " << (compTruth ? "yes" : "no") << std::endl;
    std::cout << "UseZeroCost: " << (useZeros ? "yes" : "no") << std::endl;
    std::cout << "UpdateLevel: " << (updateLevel ? "yes" : "no") << std::endl;
    std::cout << "nThreads: " << numThreads << std::endl;
  }

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  // CutMan
  algorithm::CutManager cutMan(aig, K, C, numThreads, compTruth);

  // NPNMan
  algorithm::NPNManager npnMan;

  // StrMan
  algorithm::PreCompGraphManager pcgMan(npnMan);
  pcgMan.loadPreCompGraphFromArray();
  pcgMan.processDecompositionGraphs();

  // RWMan
  algorithm::RewriteManager rwtMan(aig, cutMan, npnMan, pcgMan, triesNGraphs,
                                   useZeros, updateLevel);
  algorithm::runRewriteOperator(rwtMan);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  long double rewriteTime = duration_cast<microseconds>(t2 - t1).count();

  if (verbose == 0) {
    std::cout << fileName << ";" << C << ";" << triesNGraphs << ";" << useZeros
              << ";" << aig.getNumAnds() << ";" << aig.getDepth() << ";"
              << numThreads << ";" << rewriteTime << std::endl;
  }

  if (verbose == 1) {
    std::cout << "################ Results ################## " << std::endl;
    std::cout << "Size: " << aig.getNumAnds() << std::endl;
    std::cout << "Depth: " << aig.getDepth() << std::endl;
    std::cout << "Runtime (us): " << rewriteTime << std::endl;
  }

  // WRITE AIG //
  std::cout << "Writing final AIG file..." << std::endl;
  AigWriter aigWriter(fileName + "_rewritten.aig");
  aigWriter.writeAig(aig);

  // WRITE DOT //
  // aig.writeDot( fileName + "_rewritten.dot", aig.toDot() );
}

void kcut(aig::Aig& aig, std::string& fileName, int nThreads, int verbose) {

  int numThreads = galois::setActiveThreads(nThreads);

  int K = 6, C = 20;
  bool compTruth = false;

  if (verbose == 1) {
    std::cout << "############# Configurations ############## " << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "C: " << C << std::endl;
    std::cout << "CompTruth: " << (compTruth ? "yes" : "no") << std::endl;
    std::cout << "nThreads: " << numThreads << std::endl;
  }

  long double kcutTime                 = 0;
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  algorithm::CutManager cutMan(aig, K, C, numThreads, compTruth);
  algorithm::runKCutOperator(cutMan);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  kcutTime = duration_cast<microseconds>(t2 - t1).count();

  if (verbose == 0) {
    std::cout << fileName << ";" << K << ";" << C << ";" << compTruth << ";"
              << aig.getNumAnds() << ";" << aig.getDepth() << ";" << numThreads
              << ";" << kcutTime << std::endl;
  }

  if (verbose >= 1) {
    std::cout << "################ Results ################## " << std::endl;
    // cutMan.printAllCuts();
    // cutMan.printRuntimes();
    cutMan.printCutStatistics();
    std::cout << "Size: " << aig.getNumAnds() << std::endl;
    std::cout << "Depth: " << aig.getDepth() << std::endl;
    std::cout << "Runtime (us): " << kcutTime << std::endl;
  }
}

void rdCut(aig::Aig& aig, std::string& fileName, int nThreads, int verbose) {

  int numThreads = galois::setActiveThreads(nThreads);

  int K = 4;

  if (verbose == 1) {
    std::cout << "############# Configurations ############## " << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "nThreads: " << numThreads << std::endl;
  }

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  algorithm::ReconvDrivenCut rdcMan(aig);
  rdcMan.run(K);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  long double rdCutTime = duration_cast<microseconds>(t2 - t1).count();

  if (verbose == 0) {
    std::cout << fileName << ";" << K << ";" << aig.getNumAnds() << ";"
              << aig.getDepth() << ";" << numThreads << ";" << rdCutTime
              << std::endl;
  }

  if (verbose == 1) {
    std::cout << "################ Results ################## " << std::endl;
    std::cout << "Size: " << aig.getNumAnds() << std::endl;
    std::cout << "Depth: " << aig.getDepth() << std::endl;
    std::cout << "Runtime (us): " << rdCutTime << std::endl;
  }
}

std::string getFileName(std::string path) {

  std::size_t slash    = path.find_last_of("/") + 1;
  std::size_t dot      = path.find_last_of(".");
  std::string fileName = path.substr(slash, (dot - slash));
  return fileName;
}
