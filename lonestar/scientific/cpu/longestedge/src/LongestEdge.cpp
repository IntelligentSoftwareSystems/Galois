#include "conditions/TerrainConditionChecker.h"
#include "libmgrs/utm.h"
#include "model/Coordinates.h"
#include "model/Graph.h"
#include "model/Map.h"
#include "model/ProductionState.h"
#include "productions/Production.h"
#include "productions/Production1.h"
#include "productions/Production2.h"
#include "productions/Production3.h"
#include "productions/Production4.h"
#include "productions/Production5.h"
#include "productions/Production6.h"
#include "readers/InpReader.h"
#include "readers/SrtmReader.h"
#include "writers/InpWriter.h"
#include "writers/TriangleFormatWriter.h"
#include "utils/ConnectivityManager.h"
#include "utils/GraphGenerator.h"
#include "utils/Utils.h"
#include "readers/AsciiReader.h"

#include <Lonestar/BoilerPlate.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

namespace cll = llvm::cl;

static const char* name = "Longest edge mesh generator";
static const char* desc = "Implementation of Rivara's Longest Edge algorithm "
                          "based on hyper-graph grammars.";
static const char* url = "longest_edge";

// Command line arguments
static cll::opt<std::string> dataDir("data", cll::Positional,
                                     cll::desc("Directory with data files"));
static cll::opt<std::string> output("o", cll::Positional,
                                    cll::desc("Basename for output file"));
static cll::opt<int>
    tolerance("l", cll::Positional,
              cll::desc("Tolerance for for refinement in meters"),
              cll::init(5));
static cll::opt<bool>
    version2D("version2D",
              cll::desc("Calculate distances using only XY coordinates"));
static cll::opt<int> steps("s", cll::Positional, cll::desc("Number of steps"));
static cll::opt<double> N("N", cll::desc("Latitude of north border"));
static cll::opt<double> S("S", cll::desc("Latitude of south border"));
static cll::opt<double> E("E", cll::desc("Longitude of east border"));
static cll::opt<double> W("W", cll::desc("Longitude of west border"));
static cll::opt<bool> ascii("a", cll::desc("Read data from ascii file"));
static cll::opt<std::string>
    asciiFile("asciiFile", cll::Positional,
              cll::desc("File with data in ASCII format"));
static cll::opt<bool> square("square", cll::desc("Bind domain to square"));
static cll::opt<std::string>
    inputMeshFile("imesh", cll::desc("Filename of the inp mesh. It should be "
                                     "inside dataDir and use UTM coordinates"));
static cll::opt<long> zone("zone", cll::desc("UTM zone of the inputMeshFile"));
static cll::opt<char> hemisphere("hemisphere",
                                 cll::desc("Hemisphere of the inputMeshFile"));
static cll::opt<bool> altOutput(
    "altOutput",
    cll::desc("Write to .ele,.node,.poly files instead of AVS UCD (.inp)"));
static cll::opt<bool> display("display",
                              cll::desc("Use external visualizator."));

void afterStep(int i, Graph& graph);

bool basicCondition(const Graph& graph, GNode& node);

int main(int argc, char** argv) {
  galois::SharedMemSys G;

  LonestarStart(argc, argv, name, desc, url, nullptr);
  Graph graph{};

  galois::reportPageAlloc("MeminfoPre1");
  // Tighter upper bound for pre-alloc, useful for machines with limited memory,
  // e.g., Intel MIC. May not be enough for deterministic execution
  constexpr size_t NODE_SIZE = sizeof(**graph.begin());

  // preallocating memory
  galois::preAlloc(5 * galois::getActiveThreads() +
                   NODE_SIZE * 32 * graph.size() /
                       galois::runtime::pagePoolSize());

  galois::reportPageAlloc("MeminfoPre2");

  galois::gInfo("Initial configuration set.");

  Map* map;

  // creates the initial mesh using the borders and the new map
  if (inputMeshFile.empty()) {
    if (ascii) {
      AsciiReader reader;
      map = reader.read(asciiFile);
      GraphGenerator::generateSampleGraphWithData(
          graph, *map, 0, map->getLength() - 1, map->getWidth() - 1, 0,
          version2D);
    } else {
      SrtmReader reader;
      // terrain setup:  load terrain heights into the map object
      map = reader.read(W, N, E, S, dataDir.c_str());
      galois::gInfo("Terrain data read.");
      GraphGenerator::generateSampleGraphWithDataWithConversionToUtm(
          graph, *map, W, N, E, S, version2D, square);
    }
    galois::gInfo("Initial graph generated");
  } else {

    inpRead(dataDir + "/" + inputMeshFile, graph, N, S, E, W, version2D);
    galois::gInfo("INP mesh read.");

    // Let's convert the four corners to geodesic coordinates
    double x1, x2, x3, x4, y1, y2, y3, y4;
    Convert_UTM_To_Geodetic(zone, hemisphere, E, N, &y1, &x1);
    Convert_UTM_To_Geodetic(zone, hemisphere, E, S, &y2, &x2);
    Convert_UTM_To_Geodetic(zone, hemisphere, W, N, &y3, &x3);
    Convert_UTM_To_Geodetic(zone, hemisphere, W, S, &y4, &x4);

    std::tie(W, E) = std::minmax(
        {Utils::r2d(x1), Utils::r2d(x2), Utils::r2d(x3), Utils::r2d(x4)});
    std::tie(S, N) = std::minmax(
        {Utils::r2d(y1), Utils::r2d(y2), Utils::r2d(y3), Utils::r2d(y4)});

    // Create the map
    SrtmReader reader;

    // terrain setup:  load terrain heights into the map object
    map = reader.read(W, N, E, S, dataDir.c_str());
    galois::gInfo("Terrain data read.");

    map->setZone(zone);
    map->setHemisphere(hemisphere);

    // Update the coordinates of all graph nodes (mesh nodes, and the interior
    // nodes)
    for (auto node : graph) {
      const auto coords = node->getData().getCoords();

      node->getData().setCoords(
          Coordinates{coords.getX(), coords.getY(), *map});
    }
  }

  // initialize wrapper over graph object (ConnManager)
  ConnectivityManager connManager{graph};
  //    DummyConditionChecker checker = DummyConditionChecker();
  TerrainConditionChecker checker =
      TerrainConditionChecker(tolerance, connManager, *map);
  Production1 production1{connManager};
  Production2 production2{connManager};
  Production3 production3{connManager};
  Production4 production4{connManager};
  Production5 production5{connManager};
  Production6 production6{connManager};
  vector<Production*> productions = {&production1, &production2, &production3,
                                     &production4, &production5, &production6};
  galois::gInfo("Loop is being started...");
  //    afterStep(0, graph);
  for (int j = 0; j < steps; j++) {
    galois::for_each(galois::iterate(graph.begin(), graph.end()),
                     [&](GNode node, auto&) {
                       if (basicCondition(graph, node)) {

                         // terrain checker to see if refinement needed
                         // based on terrain
                         checker.execute(node);
                       }
                     });
    galois::gInfo("Condition chceking in step ", j, " finished.");
    galois::StatTimer step(("step" + std::to_string(j)).c_str());
    step.start();

    auto prodExecuted = true;

    while (prodExecuted) {
      prodExecuted = false;

      galois::for_each(
          galois::iterate(graph.begin(), graph.end()),
          [&](GNode node, auto& ctx) {
            // only need to check hyperedges
            if (!basicCondition(graph, node)) {
              return;
            }

            // TODO does this have to be initialized for every one?
            // may be able to optimize
            ProductionState pState(connManager, node, version2D,
                                   [&map](double x, double y) -> double {
                                     return map->get_height(x, y);
                                   });

            // loop through productions and apply the first applicable
            // one
            for (Production* production : productions) {
              if (production->execute(pState, ctx)) {
                afterStep(j, graph);
                prodExecuted = true;
                return;
              }
            }
          },
          galois::loopname(("step" + std::to_string(j)).c_str()));
    }

    step.stop();
    galois::gInfo("Step ", j, " finished.");
  }
  galois::gInfo("All steps finished.");

  // final result writing
  if (!output.empty()) {
    if (altOutput) {
      triangleFormatWriter(output, graph);
    } else {
      inpWriter(output + ".inp", graph);
    }
    galois::gInfo("Graph written to file ", output);
  }

  if (display) {
    if (system((std::string("./display.sh ") + output).c_str()))
      std::abort();
  }

  delete map;
  return 0;
}

//! Checks if node exists + is hyperedge
bool basicCondition(const Graph& graph, GNode& node) {
  return graph.containsNode(node, galois::MethodFlag::WRITE) &&
         node->getData().isHyperEdge();
}

//! Writes intermediate data to file
void afterStep(int GALOIS_UNUSED(step), Graph& GALOIS_UNUSED(graph)) {}
