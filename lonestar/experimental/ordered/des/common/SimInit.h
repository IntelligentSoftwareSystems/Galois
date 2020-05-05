/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
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
 */

#ifndef DES_SIMINIT_H
#define DES_SIMINIT_H

#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>

#include <cassert>

#include "comDefs.h"
#include "BaseSimObject.h"
#include "Event.h"
#include "SimInit.h"
#include "SimGate.h"
#include "Input.h"
#include "Output.h"

#include "logicDefs.h"
#include "LogicUpdate.h"
#include "LogicFunctions.h"
#include "LogicGate.h"
#include "OneInputGate.h"
#include "TwoInputGate.h"
#include "BasicPort.h"
#include "NetlistParser.h"

namespace des {

template <NullEventOpt NULL_EVENT_OPT, typename SimGate_tp, typename Input_tp,
          typename Output_tp>
class SimInit {

public:
  typedef NetlistParser::StimulusMapType StimulusMapType;
  typedef SimGate_tp SimGate_ty;
  typedef Input_tp Input_ty;
  typedef Output_tp Output_ty;
  typedef Event<LogicUpdate> Event_ty;
  typedef BaseSimObject<Event_ty> BaseSimObj_ty;

protected:
  /** The netlist parser. */
  NetlistParser parser;

  /** The input simulation objs. */
  std::vector<BaseSimObj_ty*> inputObjs;

  /** The output simulation objs. */
  std::vector<BaseSimObj_ty*> outputObjs;

  /** The gates i.e. other than input and output ports. */
  std::vector<BaseSimObj_ty*> gateObjs;

  /** The initial events. */
  std::vector<Event_ty> initEvents;

  /** The num edges. */
  size_t numEdges;

  /** The num nodes. */
  size_t numNodes;

  /** Counter for BaseSimObj_ty's */
  size_t simObjCntr;

protected:
  /**
   * Creates the input simulation objs.
   */
  void createInputObjs() {
    const std::vector<BasicPort*>& inputPorts = parser.getInputPorts();
    for (std::vector<BasicPort*>::const_iterator i = inputPorts.begin(),
                                                 e = inputPorts.end();
         i != e; ++i) {
      inputObjs.push_back(new Input_ty((simObjCntr++), **i));
    }
  }

  /**
   * Creates the output simulation objs.
   */
  void createOutputObjs() {
    const std::vector<BasicPort*>& outputPorts = parser.getOutputPorts();
    for (std::vector<BasicPort*>::const_iterator i = outputPorts.begin(),
                                                 e = outputPorts.end();
         i != e; ++i) {
      outputObjs.push_back(new Output_ty((simObjCntr++), **i));
    }
  }

  /**
   * Creates the gate objs.
   */
  void createGateObjs() {
    for (std::vector<LogicGate*>::const_iterator i  = parser.getGates().begin(),
                                                 ei = parser.getGates().end();
         i != ei; ++i) {
      gateObjs.push_back(new SimGate_ty((simObjCntr++), **i));
    }
  }

  /**
   * Creates the initial events.
   */
  void createInitEvents(bool createNullEvents = true) {

    const StimulusMapType& inputStimulusMap = parser.getInputStimulusMap();

    for (std::vector<BaseSimObj_ty*>::const_iterator i  = inputObjs.begin(),
                                                     ei = inputObjs.end();
         i != ei; ++i) {

      Input_ty* currInput = dynamic_cast<Input_ty*>(*i);

      assert((currInput != NULL));

      const BasicPort& impl = currInput->getImpl();

      StimulusMapType::const_iterator it =
          inputStimulusMap.find(impl.getOutputName());
      assert((it != inputStimulusMap.end()));

      const std::vector<std::pair<SimTime, LogicVal>>& tvList = it->second;

      for (std::vector<std::pair<SimTime, LogicVal>>::const_iterator
               j  = tvList.begin(),
               ej = tvList.end();
           j != ej; ++j) {

        const std::pair<SimTime, LogicVal>& p = *j;
        LogicUpdate lu(impl.getInputName(), p.second);

        Event_ty e = currInput->makeEvent(currInput, lu,
                                          Event_ty::REGULAR_EVENT, p.first);

        initEvents.push_back(e);
      }

      if (NULL_EVENT_OPT == NEEDS_NULL_EVENTS) {
        // final NULL_EVENT scheduled at INFINITY_SIM_TIME to signal that no
        // more non-null events will be received on an input

        LogicUpdate lu(impl.getInputName(), LOGIC_ZERO);
        Event_ty fe = currInput->makeEvent(currInput, lu, Event_ty::NULL_EVENT,
                                           INFINITY_SIM_TIME);
        initEvents.push_back(fe);
      }
    }
  }
  /**
   * helper function, which creates graph nodes corresponding to any simulation
   * object and add the node to the graph. No connections made yet
   */
  template <typename G>
  static void createGraphNodes(const std::vector<BaseSimObj_ty*>& simObjs,
                               G& graph, size_t& numNodes) {
    typedef typename G::GraphNode GNode;

    for (typename std::vector<BaseSimObj_ty*>::const_iterator
             i  = simObjs.begin(),
             ei = simObjs.end();
         i != ei; ++i) {
      BaseSimObj_ty* so = *i;
      GNode n           = graph.createNode(so);
      graph.addNode(n, galois::MethodFlag::UNPROTECTED);
      ++numNodes;
    }
  }
  /**
   * Creates the connections i.e. edges in the graph
   * An edge is created whenever a gate's output is connected to
   * another gate's input.
   */
  template <typename G>
  void createConnections(G& graph) {
    typedef typename G::GraphNode GNode;

    // read in all nodes first, since iterator may not support concurrent
    // modification
    std::vector<GNode> allNodes;

    for (typename G::iterator i = graph.begin(), ei = graph.end(); i != ei;
         ++i) {
      allNodes.push_back(*i);
    }

    for (typename std::vector<GNode>::iterator i  = allNodes.begin(),
                                               ei = allNodes.end();
         i != ei; ++i) {
      GNode& src = *i;

      SimGate_tp* sg = dynamic_cast<SimGate_tp*>(
          graph.getData(src, galois::MethodFlag::UNPROTECTED));
      assert(sg != NULL);
      const LogicGate& srcGate = sg->getImpl();

      const std::string& srcOutName = srcGate.getOutputName();

      for (typename std::vector<GNode>::iterator j  = allNodes.begin(),
                                                 ej = allNodes.end();
           j != ej; ++j) {
        GNode& dst     = *j;
        SimGate_tp* dg = dynamic_cast<SimGate_tp*>(
            graph.getData(dst, galois::MethodFlag::UNPROTECTED));
        assert(dg != NULL);
        const LogicGate& dstGate = dg->getImpl();

        if (dstGate.hasInputName(srcOutName)) {
          assert(&srcGate != &dstGate); // disallowing self loops
          if (graph.findEdge(src, dst) == graph.edge_end(src)) {
            ++numEdges;
            graph.addEdge(src, dst, galois::MethodFlag::UNPROTECTED);
          }
        }

      } // end inner for
    }   // end outer for
  }

  /**
   * destructor helper
   */
  void destroy() {
    destroyVec(inputObjs);
    destroyVec(outputObjs);
    destroyVec(gateObjs);
    initEvents.clear();
  }

public:
  /**
   * Instantiates a new simulation initializer.
   *
   * @param netlistFile the netlist file
   */
  SimInit(const std::string& netlistFile)
      : parser(netlistFile), simObjCntr(0) {}

  virtual ~SimInit() { destroy(); }

  /**
   * Initialize.
   *
   * Processing steps
   * create the input and output objects and add to netlistArrays
   * create the gate objects
   * connect the netlists by populating the fanout lists
   * create a list of initial events
   */
  template <typename G>
  void initialize(G& graph) {
    typedef typename G::GraphNode GNode;

    destroy();

    numNodes = 0;
    numEdges = 0;

    // create input and output objects
    createInputObjs();

    createOutputObjs();

    createGateObjs();

    createInitEvents();

    // create nodes for inputObjs
    createGraphNodes(inputObjs, graph, numNodes);

    // create nodes for outputObjs
    createGraphNodes(outputObjs, graph, numNodes);

    // create nodes for all gates
    createGraphNodes(gateObjs, graph, numNodes);

    // create the connections based on net names
    createConnections(graph);
  }

  /**
   * Verify the output by comparing the final values of the outputs of the
   * circuit from simulation against the values precomputed in the netlist file
   */
  void verify() const {

    // const std::vector<BaseSimObj_ty*>& outputObjs = getOutputObjs();
    const std::map<std::string, LogicVal>& outValues = getOutValues();

    int exitStatus = 0;

    for (typename std::vector<BaseSimObj_ty*>::const_iterator
             i  = outputObjs.begin(),
             ei = outputObjs.end();
         i != ei; ++i) {
      BaseSimObj_ty* so = *i;

      Output_ty* outObj = dynamic_cast<Output_ty*>(so);
      assert(outObj != NULL);

      BasicPort& outp = outObj->getImpl();

      const LogicVal& simulated = outp.getOutputVal();
      const LogicVal& expected  = (outValues.find(outp.getInputName()))->second;

      if (simulated != expected) {
        exitStatus = 1;
        std::cerr << "Wrong output value for " << outp.getInputName()
                  << ", expected : " << expected
                  << ", simulated : " << simulated << std::endl;
      }
    }

    if (exitStatus != 0) {
      std::cerr << "-----------------------------------------------------------"
                << std::endl;

      for (typename std::vector<BaseSimObj_ty*>::const_iterator
               i  = outputObjs.begin(),
               ei = outputObjs.end();
           i != ei; ++i) {
        BaseSimObj_ty* so = *i;

        Output_ty* outObj = dynamic_cast<Output_ty*>(so);
        assert(outObj != NULL);

        BasicPort& outp = outObj->getImpl();
        const LogicVal& expected =
            (outValues.find(outp.getInputName()))->second;

        std::cerr << "expected: " << expected << ", " << outObj->str()
                  << std::endl;
      }

      abort();
    } else {
      std::cout << ">>> OK: Simulation verified as correct" << std::endl;
    }
  }

  /**
   * Gets the inits the events.
   *
   * @return the inits the events
   */
  const std::vector<Event_ty>& getInitEvents() const { return initEvents; }

  /**
   * Gets the input names.
   *
   * @return the input names
   */
  const std::vector<std::string>& getInputNames() const {
    return parser.getInputNames();
  }

  /**
   * Gets the input objs.
   *
   * @return the input objs
   */
  const std::vector<BaseSimObj_ty*>& getInputObjs() const { return inputObjs; }

  /**
   * Gets the output names.
   *
   * @return the output names
   */
  const std::vector<std::string>& getOutputNames() const {
    return parser.getOutputNames();
  }

  /**
   * Gets the output objs.
   *
   * @return the output objs
   */
  const std::vector<BaseSimObj_ty*> getOutputObjs() const { return outputObjs; }

  /**
   * Gets the out values.
   *
   * @return the out values
   */
  const std::map<std::string, LogicVal>& getOutValues() const {
    return parser.getOutValues();
  }

  /**
   * Gets the number edges.
   *
   * @return the number edges
   */
  size_t getNumEdges() const { return numEdges; }

  /**
   * Gets the number of nodes
   *
   * @return the number of nodes
   */
  size_t getNumNodes() const { return numNodes; }
};

} // end namespace des

#endif /* DES_SIMINIT_H */
