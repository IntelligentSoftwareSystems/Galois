#include "conditions/TerrainConditionChecker.h"
#include "libmgrs/utm.h"
#include "model/Coordinates.h"
#include "model/Graph.h"
#include "model/Map.h"
#include "model/NodeData.h"
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
#include "utils/Config.h"
#include "writers/InpWriter.h"
#include "utils/ConnectivityManager.h"
#include "utils/GraphGenerator.h"
#include "utils/MyGraphFormatWriter.h"
#include "utils/Utils.h"

#include <galois/graphs/Graph.h>
#include <Lonestar/BoilerPlate.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

static const char* name = "Mesh generator";
static const char* desc = "...";
static const char* url  = "mesh_generator";

void afterStep(int i, Graph& graph);

bool basicCondition(const Graph& graph, GNode& node);

int main(int argc, char** argv) {
  Config config = Config{argc, argv};

  galois::SharedMemSys G;
  // uses own threading functions
  if (config.cores > 0) {
    galois::setActiveThreads(config.cores);
  }

  LonestarStart(argc, argv, name, desc, url);
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

  //    AsciiReader reader;
  //    Map *map = reader.read("data/test2.asc");
  galois::gInfo("Initial configuration set.");


  Map* map;

  // creates the initial mesh using the borders and the new map
  if (config.inputMeshFile.size() == 0) {

      SrtmReader reader;

      // terrain setup:  load terrain heights into the map object
      map = reader.read(config.W, config.N, config.E, config.S,
                             config.dataDir.c_str());
      galois::gInfo("Terrain data read.");
      //    GraphGenerator::generateSampleGraph(graph);
      //    GraphGenerator::generateSampleGraphWithData(graph, *map, 0,
      //    map->getLength() - 1, map->getWidth() - 1, 0, config.version2D);


    GraphGenerator::generateSampleGraphWithDataWithConversionToUtm(
      graph, *map, config.W, config.N, config.E, config.S, config.version2D);

  galois::gInfo("Initial graph generated");
  } else {

    inpRead(config.dataDir+"/"+config.inputMeshFile, graph, config);
    galois::gInfo("INP mesh read.");

    // Let's convert the four corners to geodesic coordinates
    double x1, x2, x3, x4, y1, y2, y3, y4;
    Convert_UTM_To_Geodetic(config.zone, config.hemisphere, config.E, config.N, &y1, &x1);
    Convert_UTM_To_Geodetic(config.zone, config.hemisphere, config.E, config.S, &y2, &x2);
    Convert_UTM_To_Geodetic(config.zone, config.hemisphere, config.W, config.N, &y3, &x3);
    Convert_UTM_To_Geodetic(config.zone, config.hemisphere, config.W, config.S, &y4, &x4);

    std::tie(config.W, config.E) = std::minmax({Utils::r2d(x1), Utils::r2d(x2), Utils::r2d(x3), Utils::r2d(x4)});
    std::tie(config.S, config.N) = std::minmax({Utils::r2d(y1), Utils::r2d(y2), Utils::r2d(y3), Utils::r2d(y4)});

    // Create the map
    SrtmReader reader;

    // terrain setup:  load terrain heights into the map object
    map = reader.read(config.W, config.N, config.E, config.S,
                           config.dataDir.c_str());
    galois::gInfo("Terrain data read.");


    map->setZone(config.zone);
    map->setHemisphere(config.hemisphere);

    // Update the coordinates of all graph nodes (mesh nodes, and the interior nodes)
    for (auto node : graph) {
            const auto coords = node->getData().getCoords();

            node->getData().setCoords(Coordinates{coords.getX(), coords.getY(), *map});
    }

  }

  // initialize wrapper over graph object (ConnManager)
  ConnectivityManager connManager{graph};
  //    DummyConditionChecker checker = DummyConditionChecker();
  TerrainConditionChecker checker =
      TerrainConditionChecker(config.tolerance, connManager, *map);
  Production1 production1{
      connManager}; // TODO: consider boost pointer containers, as they are
                    // believed to be better optimized
  Production2 production2{connManager};
  Production3 production3{connManager};
  Production4 production4{connManager};
  Production5 production5{connManager};
  Production6 production6{connManager};
  vector<Production*> productions = {&production1, &production2, &production3,
                                     &production4, &production5, &production6};
  galois::gInfo("Loop is being started...");
  //    afterStep(0, graph);
  for (int j = 0; j < config.steps; j++) {
    galois::for_each(galois::iterate(graph.begin(), graph.end()),
                     [&](GNode node, auto& ctx) {
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

            // TODO pretty sure this can be commented out since
            // it will capture connManager outside by reference
            ConnectivityManager connManager{graph};

            // TODO does this have to be initialized for every one?
            // may be able to optimize
            ProductionState pState(connManager, node, config.version2D,
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
          // Write the inp file
          inpWriter("output_" + std::to_string(j) + ".inp", graph);
    }
    
    step.stop();
    galois::gInfo("Step ", j, " finished.");
  }
  galois::gInfo("All steps finished.");

  // final result writing
  // MyGraphFormatWriter::writeToFile(graph, config.output);
  // galois::gInfo("Graph written to file ", config.output);


  inpWriter("output_test.inp", graph);

  if (config.display) {
    system((std::string("./display.sh ") + config.output).c_str());
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
void afterStep(int i, Graph& graph) {
  //  auto path = std::string("out/step") + std::to_string((i - 1)) + ".mgf";
  //  MyGraphFormatWriter::writeToFile(graph, path);
  //    system((std::string("./display.sh ") + path).c_str());
  //    std::cout << std::endl;
}
