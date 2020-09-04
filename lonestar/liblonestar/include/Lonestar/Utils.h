

#pragma once
#include <random>
#include <vector>
#include <algorithm>

//! Used to pick random non-zero degree starting points for search algorithms
//! This code has been copied from GAP benchmark suite
//! (https://github.com/sbeamer/gapbs/blob/master/src/benchmark.h)
template <typename Graph>
class SourcePicker {
  static const uint32_t kRandSeed;
  std::mt19937 rng;
  std::uniform_int_distribution<typename Graph::GraphNode> udist;
  const Graph& graph;

public:
  explicit SourcePicker(const Graph& g)
      : rng(kRandSeed), udist(0, g.size() - 1), graph(g) {}

  auto PickNext() {
    typename Graph::GraphNode source;
    do {
      source = udist(rng);
    } while (graph.getDegree(source) == 0);
    return source;
  }
};
template <typename Graph>
const uint32_t SourcePicker<Graph>::kRandSeed = 27491095;

//! Used to determine if a graph has power-law degree distribution or not
//! by sampling some of the vertices in the graph randomly
//! This code has been copied from GAP benchmark suite
//! (https://github.com/sbeamer/gapbs/blob/master/src/tc.cc WorthRelabelling())
template <typename Graph>
bool isApproximateDegreeDistributionPowerLaw(const Graph& graph) {
  uint32_t averageDegree = graph.sizeEdges() / graph.size();
  if (averageDegree < 10)
    return false;
  SourcePicker<Graph> sp(graph);
  uint32_t num_samples = 1000;
  if (num_samples > graph.size())
    num_samples = graph.size();
  uint32_t sample_total = 0;
  std::vector<uint32_t> samples(num_samples);
  for (uint32_t trial = 0; trial < num_samples; trial++) {
    typename Graph::GraphNode node = sp.PickNext();
    samples[trial]                 = graph.getDegree(node);
    sample_total += samples[trial];
  }
  std::sort(samples.begin(), samples.end());
  double sample_average = static_cast<double>(sample_total) / num_samples;
  double sample_median  = samples[num_samples / 2];
  return sample_average / 1.25 > sample_median;
}
