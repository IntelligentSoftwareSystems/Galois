//Write graph structure out to file in V1 structure format -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.
*/

#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <map>
#include <fstream>

namespace Galois {
namespace Graph {

template<typename Graph>
bool outputGraph(const char* file, Graph& G) {
  //ASSUME LE machine
  mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
  int fd = open(file, O_WRONLY | O_CREAT |O_TRUNC, mode);
  
  //version
  uint64_t tmp64 = 1;
  write(fd, &tmp64, sizeof(uint64_t));

  tmp64 = sizeof(typename Graph::EdgeDataTy);
  write(fd, &tmp64, sizeof(uint64_t));

  //num nodes
  tmp64 = G.size();
  write(fd, &tmp64, sizeof(uint64_t));

  //num edges and outidx computation
  tmp64 = 0;
  uint64_t offset = 0;
  std::vector<uint64_t> outIdx;
  uint64_t count = 0;
  std::map<typename Graph::GraphNode, uint32_t> NodeIDs;
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    NodeIDs[*ii] = count;
    ++count;
    tmp64 += G.neighborsSize(*ii);
    offset += G.neighborsSize(*ii);
    outIdx.push_back(offset);
  }
  write(fd, &tmp64, sizeof(uint64_t));

  //outIdx
  write(fd, &outIdx[0], sizeof(uint64_t) * outIdx.size());

  //outs
  count = 0;
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
	   ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      uint32_t tmp32 = NodeIDs[*ni];
      write(fd, &tmp32, sizeof(uint32_t));
      ++count;
    }
  }
  if (count % 2) {
    uint32_t tmp32 = 0;
    write(fd, &tmp32, sizeof(uint32_t));
  }

  //edgeData
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
	   ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      write(fd, &G.getEdgeData(*ii, *ni), sizeof(typename Graph::EdgeDataTy));
    }
  }

  close(fd);

  return true;
}

//Parsable 
template<typename Graph>
bool outputTextEdgeData(const char* ofile, Graph& G) {
  std::ofstream file(ofile);
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
	   ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      file << G.getEdgeData(*ii, *ni) << '\n';
    }
  }
  return true;
}


}
}
