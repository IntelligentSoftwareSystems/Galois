/** SimInit initializes the circuit graph and creates initial set of events -*- C++ -*-
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
 *  Created on: Jun 23, 2011
 *
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */


#ifndef SIMINIT_H_
#define SIMINIT_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>

#include <cassert>

#include "Galois/Graphs/Graph.h"

#include "comDefs.h"
#include "logicDefs.h"
#include "LogicFunctions.h"
#include "NetlistParser.h"
#include "LogicUpdate.h"


#include "Event.h"
#include "Input.h"
#include "LogicGate.h"
#include "OneInputGate.h"
#include "Output.h"
#include "SimObject.h"
#include "AbstractSimObject.h"
#include "TwoInputGate.h"

class SimInit {

public:
  typedef NetlistParser::StimulusMapType StimulusMapType;
  typedef SimObject::Graph Graph;
  typedef SimObject::GNode GNode;
  typedef SimObject::EventTy EventTy;

private:

  /** The graph. */
  Graph& graph; // should contain all the gates, inputs and outputs


  /** The netlist parser. */
  NetlistParser parser;

  /** The input simulation objs. */
  std::vector<SimObject*> inputObjs;

  /** The input nodes. */
  std::vector<GNode> inputNodes;

  /** The output simulation objs. */
  std::vector<SimObject*> outputObjs;

  /** The initial events. */
  std::vector<EventTy > initEvents;

  /** The gates i.e. other than input and output ports. */
  std::vector<SimObject*> gateObjs;

  /** The num edges. */
  size_t numEdges;

  /** The num nodes. */
  size_t numNodes;

  /** Counter for SimObject's */
  size_t simObjCntr;

  /**
   * A mapping from string name (in the netlist) to functor that implements
   * the corresponding functionality. Helps in initialization
   */
  static const std::map<std::string, OneInputFunc* > oneInputFuncMap () {
    static std::map<std::string, OneInputFunc*>  oneInMap;
    oneInMap.insert(std::make_pair (toLowerCase ("INV"), new INV()));
    return oneInMap;
  }

  /**
   * A mapping from string name (in the netlist) to functor that implements
   * the corresponding functionality. Helps in initialization
   */
  static const std::map<std::string, TwoInputFunc*> twoInputFuncMap () {
    static std::map<std::string, TwoInputFunc*> twoInMap;
    twoInMap.insert(std::make_pair (toLowerCase ("AND2") , new AND2()));
    twoInMap.insert(std::make_pair (toLowerCase ("NAND2") , new NAND2()));
    twoInMap.insert(std::make_pair (toLowerCase ("OR2") , new OR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("NOR2") , new NOR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("XOR2") , new XOR2()));
    twoInMap.insert(std::make_pair (toLowerCase ("XNOR2") , new XNOR2()));
    return twoInMap;
  }


private:

  static std::string addInPrefix(const std::string& name) {
    return "in_" + name;
  }

  static std::string addOutPrefix (const std::string& name) {
    return "out_" + name;
  }

  /**
   * Creates the input simulation objs.
   */
  void createInputObjs() {
    const std::vector<std::string>& inputNames = parser.getInputNames ();
    for (std::vector<std::string>::const_iterator i = inputNames.begin (), e = inputNames.end (); i != e; ++i) {
      const std::string& out = *i;
      std::string in = addInPrefix(out);

      inputObjs.push_back(new Input((simObjCntr++), out, in));
    }
  }

  /**
   * Creates the output simulation objs.
   */
  void createOutputObjs() {
    const std::vector<std::string>& outputNames = parser.getOutputNames ();
    for (std::vector<std::string>::const_iterator i = outputNames.begin (), e = outputNames.end (); i != e; ++i) {
      const std::string& in = *i;
      std::string out = addOutPrefix(in);

      outputObjs.push_back(new Output((simObjCntr++), out, in));
    }
  }


  /**
   * Creates the initial events.
   */
  void createInitEvents() {

    const StimulusMapType& inputStimulusMap = parser.getInputStimulusMap();

    for (std::vector<GNode>::const_iterator i = inputNodes.begin (), ei = inputNodes.end (); i != ei; ++i ) {

      const GNode& n = *i;
      Input* currInput = 
        dynamic_cast<Input* > (graph.getData (n, Galois::NONE));

      assert ((currInput != NULL));

      size_t in = currInput->getInputIndex (currInput->getInputName ());

      StimulusMapType::const_iterator it = inputStimulusMap.find (currInput->getOutputName ());
      assert ((it != inputStimulusMap.end ()));
      const std::vector<std::pair<SimTime, LogicVal> >& tvList = it->second;

      std::vector<EventTy > myEvents;

      for (std::vector< std::pair<SimTime, LogicVal> >::const_iterator j = tvList.begin (), ej = tvList.end ();
          j != ej; ++j) {
        const std::pair<SimTime, LogicVal>& p = *j;
        LogicUpdate lu(currInput->getOutputName(), p.second);

        EventTy e = currInput->makeEvent(currInput, currInput, EventTy::REGULAR_EVENT, lu, p.first);

        // adding the event to currInput's pending events 
        currInput->recvEvent(in, e);

        initEvents.push_back(e);
        myEvents.push_back(e);
      }


      // add a final null event with infinity as the time, to allow termination
      EventTy fe = currInput->makeEvent (currInput, currInput, EventTy::NULL_EVENT, LogicUpdate (), INFINITY_SIM_TIME);

      currInput->recvEvent (in, fe);

      currInput->updateActive (); // mark all inputs active

    }

  }

  /**
   * Creates the gate objs.
   */
  void createGateObjs() {
    const std::vector<GateRec>& gateRecs = parser.getGates();

    for (std::vector<GateRec>::const_iterator i = gateRecs.begin (), ei = gateRecs.end (); i != ei; ++i) {
      const GateRec& grec = *i;

      if (oneInputFuncMap().count (grec.name) > 0) {
        const OneInputFunc* func = (oneInputFuncMap().find (grec.name))->second;

        assert ((grec.outputs.size () == 1));
        const std::string& out = grec.outputs[0];

        assert ((grec.inputs.size() == 1));
        const std::string& in = grec.inputs[0];

        const SimTime& delay = grec.delay;

        OneInputGate* g = new OneInputGate ((simObjCntr++), *func, out, in, delay);

        this->gateObjs.push_back(g);

      } else if (twoInputFuncMap().count (grec.name) > 0) {
        const TwoInputFunc* func = (twoInputFuncMap().find (grec.name))->second;

        assert (grec.outputs.size() == 1);
        const std::string& out = grec.outputs[0];

        assert (grec.inputs.size() == 2);
        const std::string& in1 = grec.inputs[0];
        const std::string& in2 = grec.inputs[1];

        const SimTime& delay = grec.delay;

        TwoInputGate* g = new TwoInputGate ((simObjCntr++), *func, out, in1, in2, delay);

        this->gateObjs.push_back(g);

      } else {
        std::cerr << "Found a gate with unknown name: " << grec.getName() << std::endl;
        abort ();
      }

    }

  }

  /**
   * helper function, which creates graph nodes corresponding to any simulation object
   * and add the node to the graph. No connections made yet
   */
  void createGraphNodes (const std::vector<SimObject*>& simObjs) {
    for (std::vector<SimObject*>::const_iterator i = simObjs.begin (), ei = simObjs.end (); i != ei; ++i) {
      SimObject* so = *i;
      GNode n = graph.createNode(so);
      graph.addNode(n, Galois::NONE);
      ++numNodes;
    }
  }
  /**
   * Creates the connections i.e. edges in the graph
   * An edge is created whenever a gate's output is connected to 
   * another gate's input.
   */
  void createConnections() {

    // read in all nodes first, since iterator may not support concurrent modification
    std::vector<GNode> allNodes;

    for (Graph::active_iterator i = graph.active_begin (), ei = graph.active_end (); i != ei; ++i) {
      allNodes.push_back (*i);
    }

    for (std::vector<GNode>::iterator i = allNodes.begin (), ei = allNodes.end (); i != ei; ++i) {
      GNode& src = *i;
      LogicGate* srcGate = 
        dynamic_cast<LogicGate* > (graph.getData (src, Galois::NONE));

      assert (srcGate != NULL);
      const std::string& srcOutName = srcGate->getOutputName ();

      for (std::vector<GNode>::iterator j = allNodes.begin (), ej = allNodes.end (); j != ej; ++j) {
        GNode& dst = *j;
        LogicGate* dstGate = 
           dynamic_cast<LogicGate* > (graph.getData (dst, Galois::NONE));

        assert (dstGate != NULL);

        if (dstGate->hasInputName (srcOutName)) {
          assert (srcGate != dstGate); // disallowing self loops
          if (!src.hasNeighbor (dst)) {
            ++numEdges;
            graph.addEdge (src, dst, Galois::NONE);
          }

        }

      }
    }


  }

  /**
   * Initialize.
   *
   * Processing steps
   * create the input and output objects and add to netlistArrays
   * create the gate objects
   * connect the netlists by populating the fanout lists
   * create a list of initial events
   */
  void initialize() {
    numNodes = 0;
    numEdges = 0;


    // create input and output objects
    createInputObjs();
    createOutputObjs();

    createGateObjs();

    // add all gates, inputs and outputs to the Graph graph
    for (std::vector<SimObject*>::const_iterator i = inputObjs.begin (), ei = inputObjs.end (); i != ei; ++i) {
      SimObject* so = *i;

      GNode n = graph.createNode(so);
      graph.addNode(n, Galois::NONE);
      ++numNodes;
      inputNodes.push_back(n);
    }

    createInitEvents();

    // create nodes for outputObjs
    createGraphNodes (outputObjs);

    // create nodes for all gates
    createGraphNodes (gateObjs);

    // create the connections based on net names
    createConnections();
  }

  /**
   * freeing pointers in a vector
   * before the vector itself is destroyed
   */
  template <typename T>
  static void destroyVec (std::vector<T*>& vec) {
    for (typename std::vector<T*>::iterator i = vec.begin (), ei = vec.end (); i != ei; ++i) {
      delete *i;
      *i = NULL;
    }
  }

  /**
   * destructor helper
   */
  void destroy () {
    destroyVec (inputObjs);
    destroyVec (outputObjs);
    destroyVec (gateObjs);
  }

public:
  /**
   * Instantiates a new simulation initializer.
   *
   * @param graph the graph
   * @param netlistFile the netlist file
   */
  SimInit(Graph& graph, const char* netlistFile) 
    : graph(graph), parser(netlistFile), simObjCntr(0) {
    initialize();
  }

  ~SimInit () {
    destroy ();
  }

  /**
   * Gets the graph.
   *
   * @return the graph
   */
  const Graph& getGraph() const {
    return graph;
  }

  /**
   * Gets the inits the events.
   *
   * @return the inits the events
   */
  const std::vector<EventTy >& getInitEvents() const {
    return initEvents;
  }

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
  const std::vector<SimObject*>& getInputObjs() const {
    return inputObjs;
  }

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
  const std::vector<SimObject*> getOutputObjs() const {
    return outputObjs;
  }

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
  size_t getNumEdges() const {
    return numEdges;
  }

  /**
   * Gets the number of nodes
   *
   * @return the number of nodes
   */
  size_t getNumNodes() const {
    return numNodes;
  }

  /**
   * Gets the input nodes.
   *
   * @return the input nodes
   */
  const std::vector<GNode>& getInputNodes() const {
    return inputNodes;
  }
};


#endif /* SIMINIT_H_ */
