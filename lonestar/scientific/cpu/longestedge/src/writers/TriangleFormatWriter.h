#ifndef TRI_WRITER_H
#define TRI_WRITER_H

#include "../model/Coordinates.h"
#include "../model/Graph.h"

#include <cstddef>
#include <string>
#include <vector>

struct TriNodeInfo {
  Coordinates coods;
  size_t mat;
  size_t id;
};

struct TriConecInfo {
  std::vector<size_t> conec;
  size_t mat;
};

struct TriSegmInfo {
  std::vector<size_t> points;
  bool border;
};

void triangleFormatWriter(const std::string& filename, Graph& graph);

void trProcessGraph(Graph& graph, std::vector<TriNodeInfo>& nodeVector,
                    std::vector<TriSegmInfo>& segmVector,
                    std::vector<TriConecInfo>& conecVector);

void changeOrientationIfRequired(std::vector<unsigned long> element,
                                 std::vector<TriNodeInfo> nodeVector);

#endif // TRI_WRITER_H
