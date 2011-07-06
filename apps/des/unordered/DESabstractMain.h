/*
 * DESabstractMain.h
 *
 *  Created on: Jun 24, 2011
 *      Author: amber
 */

#ifndef DES_ABSTRACT_MAIN_H_
#define DES_ABSTRACT_MAIN_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include <cstdio>

#include "Galois/Statistic.h"
#include "Galois/Graphs/Graph.h"
#include "Galois/Galois.h"

#include "Lonestar/Banner.h"
#include "Lonestar/CommandLine.h"

#include "SimObject.h"
#include "SimInit.h"

static const char* name = "Discrete Event Simulation";
static const char* description = "Uses Chandy-Misra's algorithm, which is unordered, to perform logic circuit simulations";
static const char* url = "http://iss.ices.utexas.edu/lonestar/des.html";
static const char* help = "<progname> netlistFile";

/**
 * The Class DESabstractMain holds common functionality for {@link DESunorderedSerial} and {@link DESunordered}.
 */
class DESabstractMain {
public:
  typedef SimObject::Graph Graph;
  typedef SimObject::GNode GNode;

protected:

  /** The graph. */
  Graph graph;

  virtual  bool  isSerial() const = 0;

  /**
   * Run loop.
   *
   * @throws Exception the exception
   */
  virtual void runLoop(const SimInit<Graph, GNode>& simInit) = 0;

  /**
   * Gets the version.
   *
   * @return the version
   */
  std::string getVersion() const {
    return (isSerial() ? "Serial" : "Galois");
  }

public:
  /**
   * Run.
   *
   * @param args the args
   */
  void run(int argc, const char* argv[]) {
    std::vector<const char*> args = parse_command_line (argc, argv, help);


    if (args.size () < 1) {
      std::cerr << help << std::endl;
      assert (false);
    }

    printBanner(std::cout, name, description, url);

    const char* netlistFile = args[0];
    SimInit<Graph, GNode> simInit(graph, netlistFile);


    printf("circuit graph: %u nodes, %zd edges\n", graph.size(), simInit.getNumEdges());
    printf("Number of initial events = %zd\n", simInit.getInitEvents().size());

    Galois::StatTimer t;

    t.start ();

      runLoop(simInit);

    t.stop ();

    if (!skipVerify) {
      verify(simInit);
    }

  }

  /**
   * Verify the output.
   */
  void verify(const SimInit<Graph, GNode>& simInit) {
    const std::vector<SimObject*>& outputObjs = simInit.getOutputObjs();
    const std::map<std::string, LogicVal>& outValues = simInit.getOutValues();

    int exitStatus = 0;

    for (std::vector<SimObject*>::const_iterator i = outputObjs.begin (), ei = outputObjs.end (); i != ei; ++i) {
      SimObject* so = *i;

      Output<Graph, GNode>* outObj = dynamic_cast< Output<Graph, GNode>* > (so);

      assert (outObj != NULL);

      const LogicVal& simulated = outObj->getOutputVal();
      const LogicVal& expected = (outValues.find (outObj->getInputName ()))->second;

      if (simulated != expected) {
        exitStatus = 1;
        std::cerr << "Wrong output value for " << outObj->getOutputName() 
          << ", simulated : " << simulated << " expected : " << expected << std::endl;
      }
    }

    if (exitStatus != 0) {
      std::cerr << "-----------------------------------------------------------" << std::endl;

      for (std::vector<SimObject*>::const_iterator i = outputObjs.begin (), ei = outputObjs.end (); i != ei; ++i) {
        SimObject* so = *i;

        Output<Graph, GNode>* outObj = dynamic_cast< Output<Graph, GNode>* > (so);
        assert (outObj != NULL);

        std::cerr << outObj->toString () << std::endl;
      }

      assert (false);
    } else {
      std::cout << ">>> OK: Simulation verified as correct" << std::endl;
    }
  }

};

#endif // DES_ABSTRACT_MAIN_H_ 
