#ifndef INP_WRITER_H
#define INP_WRITER_H

#include "../model/Coordinates.h"
#include "../model/Graph.h"

#include <cstddef>
#include <string>
#include <vector>

struct InpNodeInfo {
  Coordinates coods;
  size_t mat;
  size_t id;
};

struct InpConecInfo {
  std::vector<size_t> conec;
  size_t mat;
  std::string type;
};

void inpWriter(const std::string filename, Graph& graph);

void processGraph(Graph& graph, std::vector<InpNodeInfo>& nodeVector,
                  std::vector<InpConecInfo>& conecVector);

#endif // INP_WRITER_H
