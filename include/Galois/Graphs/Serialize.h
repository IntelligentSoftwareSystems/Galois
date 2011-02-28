//Write graph structure out to file in V1 structure format -*- C++ -*-
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
  uint64_t tmp = 1;
  write(fd, &tmp, sizeof(uint64_t));

  //num nodes
  tmp = G.size();
  write(fd, &tmp, sizeof(uint64_t));

  //num edges and outidx computation
  tmp = 0;
  uint64_t offset = 0;
  std::vector<uint64_t> outIdx;
  uint64_t count = 0;
  std::map<typename Graph::GraphNode, uint64_t> NodeIDs;
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    NodeIDs[*ii] = count;
    ++count;
    tmp += G.neighborsSize(*ii);
    offset += G.neighborsSize(*ii);
    outIdx.push_back(offset);
  }
  write(fd, &tmp, sizeof(uint64_t));

  //outIdx
  write(fd, &outIdx[0], sizeof(uint64_t) * outIdx.size());

  //outs
  for (typename Graph::active_iterator ii = G.active_begin(),
	 ee = G.active_end(); ii != ee; ++ii) {
    for (typename Graph::neighbor_iterator ni = G.neighbor_begin(*ii),
	   ne = G.neighbor_end(*ii); ni != ne; ++ni) {
      tmp = NodeIDs[*ni];
      write(fd, &tmp, sizeof(uint64_t));
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
