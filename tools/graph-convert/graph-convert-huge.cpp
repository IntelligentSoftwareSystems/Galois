/** Graph converter -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @author Donald Nguyen <ddn@cs.utexas.edu>
 */

#include "galois/LargeArray.h"
#include "galois/Graphs/FileGraph.h"
#include "galois/runtime/OfflineGraph.h"

#include "llvm/Support/CommandLine.h"

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/mpl/if.hpp>
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <ios>
#include <limits>
#include <stdint.h>
#include <vector>
#include <random>
#include <chrono>
#include <regex>
#include <fcntl.h>
#include <cstdlib>

namespace cll = llvm::cl;


std::ios_base::failure::failure(char const*, std::error_code const&) {}

static cll::opt<std::string> inputFilename(cll::Positional, 
    cll::desc("<input file>"), cll::Required);
static cll::opt<std::string> outputFilename(cll::Positional,
    cll::desc("<output file>"), cll::Required);
static cll::opt<bool> useSmallData("32bitData", cll::desc("Use 32 bit data"), cll::init(false));
static cll::opt<bool> edgesSorted("edgesSorted", cll::desc("Edges are sorted by the sourceIDs."), cll::init(false));
static cll::opt<unsigned long long> numNodes("numNodes", cll::desc("Total number of nodes given."), cll::init(0));

union dataTy {
  int64_t ival;
  double dval;
  float fval;
  int32_t i32val;
};

void perEdge(std::istream& is, 
             std::function<void(uint64_t, uint64_t, dataTy)> fn, 
             std::function<void(uint64_t, uint64_t)> fnPreSize) {
  std::string line;

  uint64_t bytes = 0;
  uint64_t counter = 0;
  uint64_t totalBytes = 0;

  const std::regex problemLine("^p[[:space:]]+[[:alpha:]]+[[:space:]]+([[:digit:]]+)[[:space:]]+([[:digit:]]+)");
  const std::regex noData(   "^a?[[:space:]]*([[:digit:]]+)[[:space:]]+([[:digit:]]+)[[:space:]]*");
  //const std::regex noData_nospace(   "^a?[[:space:]]*([[:digit:]]+)[[:space:]]+([[:digit:]]+)");
  const std::regex intData(  "^a?[[:space:]]*([[:digit:]]+)[[:space:]]+([[:digit:]]+)[[:space:]]+(-?[[:digit:]]+)");
  const std::regex floatData("^a?[[:space:]]*([[:digit:]]+)[[:space:]]+([[:digit:]]+)[[:space:]]+(-?[[:digit:]]+\\.[[:digit:]]+)");

  auto timer = std::chrono::system_clock::now();
  auto timerStart = timer;

  std::smatch matches;
  bool zeroBased = false; // set to 1 if file is one-indexed
  bool seenEdge = false;

  while(std::getline(is, line)) {
    auto t = line.size() + 1;
    bytes += t;
    totalBytes += t;
    ++counter;

    if (counter == 1024*128) {
      counter = 0;
      auto timer2 = std::chrono::system_clock::now();
      std::cout << "Scan: " << (double)bytes / std::chrono::duration_cast<std::chrono::microseconds>(timer2 - timer).count() << " MB/s\n";
      timer = timer2;
      bytes = 0;
    }

    dataTy data;
    bool match = false;
    if (std::regex_match(line, matches, floatData)) {
      if (useSmallData)
        data.fval = std::stof(matches[3].str());
      else
        data.dval = std::stod(matches[3].str());
      match = true;
    } else if (std::regex_match(line, matches, intData)) {
      if (useSmallData)
        data.i32val = std::stoul(matches[3].str());
      else
        data.ival = std::stoll( matches[3].str());
      match = true;
    } else if (std::regex_match(line, matches, noData)){ // || std::regex_match(line, matches, noData_nospace)) {
      data.ival = 0;
      match = true;
    } else if (std::regex_match(line, matches, problemLine)) {
      if (seenEdge) {
        std::cerr << "Error: seeing a dimacs problem line after seeing edges\n";
        abort();
      }
      zeroBased = true; // dimacs files are 1-indexed
      fnPreSize(std::stoull(matches[1].str()), std::stoull(matches[2].str()));
    }
    if (match) {
      seenEdge = true;
      uint64_t src = std::stoull(matches[1].str());
      uint64_t dst = std::stoull(matches[2].str());
      if (zeroBased) {
        if (src == 0 || dst == 0) {
          std::cerr << "Error: node id 0 in a dimacs graph\n";
          abort();
        }
        src -= 1;
        dst -= 1;
      }
      fn(src, dst, data);
    }
  }
  auto timer2 = std::chrono::system_clock::now();
  std::cout << "File Scan: " << (double)totalBytes / std::chrono::duration_cast<std::chrono::microseconds>(timer2 - timerStart).count() << " MB/s\n";
}

void go(std::istream& input) {
  try {
    std::deque<uint64_t> edgeCount;
    perEdge(input,
            [&edgeCount] (uint64_t src, uint64_t, dataTy) { if (edgeCount.size() <= src) edgeCount.resize(src + 1); ++edgeCount[src];},
            [&edgeCount] (uint64_t nodes, uint64_t edges) { edgeCount.resize(nodes); }
            );
    input.clear();
    input.seekg(0, std::ios_base::beg);
    galois::graphs::OfflineGraphWriter outFile(outputFilename, useSmallData);
    outFile.setCounts(edgeCount);
    perEdge(input,
            [&outFile, &edgeCount] (uint64_t src, uint64_t dst, dataTy data) {
              auto off = --edgeCount[src];
              if (useSmallData)
                outFile.setEdge(src, off, dst, data.i32val);
              else
                outFile.setEdge(src, off, dst, data.ival);
            },
            [] (uint64_t, uint64_t) {}
            );
  } catch( const char* c) {
    std::cerr << "Failed with: " << c << "\n";
    abort();
  }
}

void go_edgesSorted(std::istream& input, uint64_t numNodes) {
  try {
    std::deque<uint64_t> edgeCount(numNodes, 0);
    input.clear();
    input.seekg(0, std::ios_base::beg);
    galois::graphs::OfflineGraphWriter outFile(outputFilename, useSmallData, numNodes);
    outFile.setCounts(edgeCount);
    outFile.seekEdgesDstStart();
    uint64_t curr_src = 0;
    uint64_t curr_src_edgeCount = 0;
    perEdge(input,
            [&outFile, &edgeCount, &curr_src, &curr_src_edgeCount] (uint64_t src, uint64_t dst, dataTy data) {
              if(src == curr_src){
                ++curr_src_edgeCount;
              }
              else{
                //std::cout << "CHANGES : " << src << " : " << curr_src << " COUNT : " << curr_src_edgeCount << "\n";
                if(src < curr_src){
                  std::cerr << " ERROR : File is not sorted\n";
                  abort();
                }
                edgeCount[curr_src] = curr_src_edgeCount;
                curr_src = src;
                curr_src_edgeCount = 1;
              }
                outFile.setEdgeSorted(dst);
            },
            [] (uint64_t, uint64_t) {}
            );
    //To take care of the last src node ID.
    edgeCount[curr_src] = curr_src_edgeCount;
    outFile.setCounts(edgeCount);
  } catch( const char* c) {
    std::cerr << "Failed with: " << c << "\n";
    abort();
  }
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);
  //  std::ios_base::sync_with_stdio(false);
  std::cout << "Data will be " << (useSmallData ? 4 : 8) << " Bytes\n";

  std::ifstream infile(inputFilename, std::ios_base::in);
  if (!infile) {
    std::cout << "Failed to open " << inputFilename << "\n";
    return 1;
  }

  // // if (isCompressed(inputType)) {
  // //   boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
  // //   inbuf.push(boost::iostreams::gzip_decompressor());
  // //   inbuf.push(infile);
  // //   //Convert streambuf to istream
  // //   std::istream instream(&inbuf);
  // //   go(instream);
  // // } else {
  if(numNodes > 0 && edgesSorted){
    go_edgesSorted(infile, numNodes);
  }
  else{
    go(infile);
  }
  //  }

  return 0;
}
