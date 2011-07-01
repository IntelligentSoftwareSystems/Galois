/*
 * SimInit.h
 *
 *  Created on: Jun 23, 2011
 *      Author: amber
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
#include "AbsSimObject.h"
#include "TwoInputGate.h"

template <typename GraphTy, typename GNodeTy>
class SimInit {
public:
  typedef NetlistParser::StimulusMapType StimulusMapType;
  typedef AbsSimObject<GraphTy, GNodeTy, Event<GNodeTy, LogicUpdate> > AbsSimObj;

private:

  /** The graph. */
  GraphTy& graph; // should contain all the gates, inputs and outputs


  /** The parser. */
  NetlistParser parser;

  /** The input objs. */
  std::vector<SimObject*> inputObjs;

  /** The input nodes. */
  std::vector<GNodeTy> inputNodes;

  /** The output objs. */
  std::vector<SimObject*> outputObjs;

  /** The initial events. */
  std::vector<Event<GNodeTy, LogicUpdate> > initEvents;

  /** The gates. */
  std::vector<SimObject*> gateObjs;

  /** The num edges. */
  size_t numEdges;

  /** The num nodes. */
  size_t numNodes;


  static const std::map<std::string, OneInputFunc* > oneInputFuncMap () {
    static std::map<std::string, OneInputFunc*>  oneInMap;
    oneInMap.insert(std::make_pair (toLowerCase ("INV"), new INV()));
    return oneInMap;
  }

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
   * Creates the input objs.
   */
  void createInputObjs() {
    const std::vector<std::string>& inputNames = parser.getInputNames ();
    for (std::vector<std::string>::const_iterator i = inputNames.begin (), e = inputNames.end (); i != e; ++i) {
      const std::string& out = *i;
      std::string in = addInPrefix(out);

      inputObjs.push_back(new Input<GraphTy, GNodeTy>(out, in));
    }
  }

  /**
   * Creates the output objs.
   */
  void createOutputObjs() {
    const std::vector<std::string>& outputNames = parser.getOutputNames ();
    for (std::vector<std::string>::const_iterator i = outputNames.begin (), e = outputNames.end (); i != e; ++i) {
      const std::string& in = *i;
      std::string out = addOutPrefix(in);

      outputObjs.push_back(new Output<GraphTy, GNodeTy>(out, in));
    }
  }


  /**
   * Creates the initial events.
   */
  void createInitEvents() {

    const StimulusMapType& inputStimulusMap = parser.getInputStimulusMap();

    for (typename std::vector<GNodeTy>::const_iterator i = inputNodes.begin (), ei = inputNodes.end (); i != ei; ++i ) {

      const GNodeTy& n = *i;
      Input<GraphTy, GNodeTy>* currInput = 
        dynamic_cast<Input<GraphTy, GNodeTy>* > (graph.getData (n, Galois::Graph::NONE));

      assert ((currInput != NULL));

      StimulusMapType::const_iterator it = inputStimulusMap.find (currInput->getOutputName ());
      assert ((it != inputStimulusMap.end ()));
      const std::vector<std::pair<SimTime, LogicVal> >& tvList = it->second;

      std::vector<Event<GNodeTy, LogicUpdate> > myEvents;

      for (std::vector< std::pair<SimTime, LogicVal> >::const_iterator j = tvList.begin (), ej = tvList.end ();
          j != ej; ++j) {
        const std::pair<SimTime, LogicVal>& p = *j;
        LogicUpdate lu(currInput->getOutputName(), p.second);

        Event<GNodeTy, LogicUpdate> e = Event<GNodeTy, LogicUpdate>::makeEvent(n, n, Event<GNodeTy, LogicUpdate>::REGULAR_EVENT, lu,
            p.first);

        initEvents.push_back(e);
        myEvents.push_back(e);
      }

      // adding the initial events to the respective input
      size_t in = currInput->getInputIndex(currInput->getInputName());
      for (typename std::vector< Event<GNodeTy, LogicUpdate> >::const_iterator j = myEvents.begin ()
          , ej = myEvents.end (); j != ej; ++j) {

        const Event<GNodeTy, LogicUpdate>& event = *j;
        currInput->recvEvent(in, event);

      }

      // add a final null event with infinity as the time, to allow termination
      currInput->recvEvent (in,
          Event<GNodeTy, LogicUpdate>::makeEvent (n, n, Event<GNodeTy, LogicUpdate>::NULL_EVENT,
              LogicUpdate (), INFINITY_SIM_TIME));

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

        OneInputGate<GraphTy, GNodeTy>* g = new OneInputGate<GraphTy, GNodeTy> (*func, out, in, delay);

        this->gateObjs.push_back(g);

      } else if (twoInputFuncMap().count (grec.name) > 0) {
        const TwoInputFunc* func = (twoInputFuncMap().find (grec.name))->second;

        assert (grec.outputs.size() == 1);
        const std::string& out = grec.outputs[0];

        assert (grec.inputs.size() == 2);
        const std::string& in1 = grec.inputs[0];
        const std::string& in2 = grec.inputs[1];

        const SimTime& delay = grec.delay;

        TwoInputGate<GraphTy, GNodeTy>* g = new TwoInputGate<GraphTy, GNodeTy> (*func, out, in1, in2, delay);

        this->gateObjs.push_back(g);

      } else {
        std::cerr << "Found a gate with unknown name: " << grec.getName() << std::endl;
        assert (false);
      }

    }

  }

  void createGraphNodes (const std::vector<SimObject*>& simObjs) {
    for (typename std::vector<SimObject*>::const_iterator i = simObjs.begin (), ei = simObjs.end (); i != ei; ++i) {
      SimObject* so = *i;
      GNodeTy n = graph.createNode(so);
      graph.addNode(n, Galois::Graph::NONE);
      ++numNodes;
    }
  }
  /**
   * Creates the connections.
   */
  // assumes that all the nodes have been added to the graph
  void createConnections() {

    // read in all nodes first, since iterator may not support concurrent modification
    std::vector<GNodeTy> allNodes;

    for (typename GraphTy::active_iterator i = graph.active_begin (), ei = graph.active_end (); i != ei; ++i) {
      allNodes.push_back (*i);
    }

    for (typename std::vector<GNodeTy>::iterator i = allNodes.begin (), ei = allNodes.end (); i != ei; ++i) {
      GNodeTy& src = *i;
      LogicGate<GraphTy, GNodeTy>* srcGate = 
        dynamic_cast<LogicGate<GraphTy, GNodeTy>* > (graph.getData (src, Galois::Graph::NONE));

      assert (srcGate != NULL);
      const std::string& srcOutName = srcGate->getOutputName ();

      for (typename std::vector<GNodeTy>::iterator j = allNodes.begin (), ej = allNodes.end (); j != ej; ++j) {
        GNodeTy& dst = *j;
        LogicGate<GraphTy, GNodeTy>* dstGate = 
           dynamic_cast<LogicGate<GraphTy, GNodeTy>* > (graph.getData (dst, Galois::Graph::NONE));

        assert (dstGate != NULL);

        if (dstGate->hasInputName (srcOutName)) {
          assert (srcGate != dstGate); // disallowing self loops
          if (!src.hasNeighbor (dst)) {
            ++numEdges;
            graph.addEdge (src, dst, Galois::Graph::NONE);
          }

        }

      }
    }


  }

  /*
   * Processing steps
   * create the input and output objects and add to netlistArrays
   * create the gate objects
   * connect the netlists by populating the fanout lists
   * create a list of initial events
   *
   */

  /**
   * Initialize.
   */
  void initialize() {
    numNodes = 0;
    numEdges = 0;


    // reset the id counters for SimObject and Event class so that the ids are numbered 0 .. numItems
    AbsSimObj::resetIdCounter ();
    BaseEvent<GNodeTy, LogicUpdate>::resetIdCounter ();
    

    // create input and output objects
    createInputObjs();
    createOutputObjs();

    createGateObjs();

    // add all gates, inputs and outputs to the Graph graph
    for (typename std::vector<SimObject*>::const_iterator i = inputObjs.begin (), ei = inputObjs.end (); i != ei; ++i) {
      SimObject* so = *i;

      GNodeTy n = graph.createNode(so);
      graph.addNode(n, Galois::Graph::NONE);
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

  template <typename T>
  void destroyVec (std::vector<T*>& vec) {
    for (typename std::vector<T*>::iterator i = vec.begin (), ei = vec.end (); i != ei; ++i) {
      delete *i;
      *i = NULL;
    }
  }

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
  SimInit(GraphTy& graph, const char* netlistFile) 
    : graph(graph), parser(netlistFile) {
    initialize();
  }

  ~SimInit () {
    destroy ();
  }

  // TODO: write a destructor to free the memory etc.

  /**
   * Gets the graph.
   *
   * @return the graph
   */
  const GraphTy& getGraph() const {
    return graph;
  }

  /**
   * Gets the inits the events.
   *
   * @return the inits the events
   */
  const std::vector<Event<GNodeTy, LogicUpdate> >& getInitEvents() const {
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
  const std::vector<GNodeTy>& getInputNodes() const {
    return inputNodes;
  }
};


#endif /* SIMINIT_H_ */
