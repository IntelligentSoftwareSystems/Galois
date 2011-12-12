/** DESabstractMain holds common functionality for main classes -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * Created on: Jun 24, 2011
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
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

#include "llvm/Support/CommandLine.h"

#include "Lonestar/BoilerPlate.h"

#include "SimObject.h"
#include "SimInit.h"

namespace cll = llvm::cl;

static const char* name = "Discrete Event Simulation";
static const char* desc = "Uses Chandy-Misra's algorithm, which is unordered, to perform logic circuit simulations";
static const char* url = "discrete_event_simulation";

static cll::opt<std::string> netlistFile(cll::Positional, cll::desc("<input file>"), cll::Required);

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

  /**
   * return true in serial versions
   * and false in Galois versions
   */
  virtual  bool  isSerial() const = 0;

  /**
   * Run loop.
   *
   * @throws Exception the exception
   */
  virtual void runLoop(const SimInit& simInit) = 0;

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
   * Run the simulation
   * @param argc
   * @param argv
   */
  void run(int argc, char* argv[]) {

    LonestarStart(argc, argv, std::cout, name, desc, url);

    printf ("Processing %zd events per iteration\n", AbstractSimObject::NEVENTS_PER_ITER);

    SimInit simInit(graph, netlistFile.c_str());


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
   * Verify the output by comparing the final values of the outputs of the circuit
   * from simulation against the values precomputed in the netlist file
   */
  void verify(const SimInit& simInit) {
    const std::vector<SimObject*>& outputObjs = simInit.getOutputObjs();
    const std::map<std::string, LogicVal>& outValues = simInit.getOutValues();

    int exitStatus = 0;

    for (std::vector<SimObject*>::const_iterator i = outputObjs.begin (), ei = outputObjs.end (); i != ei; ++i) {
      SimObject* so = *i;

      Output* outObj = dynamic_cast< Output* > (so);
      assert (outObj != NULL);

      BasicPort& outp = outObj->getImpl ();

      const LogicVal& simulated = outp.getOutputVal();
      const LogicVal& expected = (outValues.find (outp.getInputName ()))->second;

      if (simulated != expected) {
        exitStatus = 1;
        std::cerr << "Wrong output value for " << outp.getInputName () 
          << ", simulated : " << simulated << " expected : " << expected << std::endl;
      }
    }

    if (exitStatus != 0) {
      std::cerr << "-----------------------------------------------------------" << std::endl;

      for (std::vector<SimObject*>::const_iterator i = outputObjs.begin (), ei = outputObjs.end (); i != ei; ++i) {
        SimObject* so = *i;

        Output* outObj = dynamic_cast< Output* > (so);
        assert (outObj != NULL);

        std::cerr << outObj->toString () << std::endl;
      }

      abort ();
    } else {
      std::cout << ">>> OK: Simulation verified as correct" << std::endl;
    }
  }

};

#endif // DES_ABSTRACT_MAIN_H_ 
