/** SimObject: the abstract interface to be implemented by any simulation object -*- C++ -*-
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */


#ifndef SIMOBJECT_H_
#define SIMOBJECT_H_

#include <vector>
#include <queue>

#include <cassert>

#include "Galois/Graphs/Graph.h"

#include "comDefs.h"
#include "BaseEvent.h"
#include "LogicUpdate.h"

#include "Event.h"

//TODO: modeling one output for now. Need to extend for multiple outputs
/**
 * @section Description
 *
 * The Class SimObject represents an abstract simulation object (processing station). A simulation application
 * would inherit from this class.
 */


class SimObject {
public:
  typedef Galois::Graph::FirstGraph <SimObject*, void, true>  Graph;
  typedef Graph::GraphNode GNode;
  typedef Event<GNode, LogicUpdate> EventType;


public:
  virtual ~SimObject () {}
  /**
   * a way to construct different subtypes
   * @return a new instance of subtype
   */
  virtual SimObject* clone() const = 0;



  /**
   * Simulate.
   *
   * @param graph: the graph composed of simulation objects/stations and communication links 
   * @param myNode the node in the graph that has this SimObject as its node data
   * @return number of events that were processed during the call
   */
  virtual size_t simulate(Graph& graph, GNode& myNode) = 0; 



  /**
   * Checks if is active.
   *
   * @return true, if is active
   */
  virtual bool isActive() const = 0;

  /**
   * active is set to true if there exists an event on each input pq
   * or if an input pq is empty and an event with INFINITY has been received
   * telling that no more events on this input will be received.
   */
  virtual void updateActive() = 0; 

  /**
   * string representation for printing
   */
  virtual const std::string toString() const = 0;

  /**
   * Gets the id.
   *
   * @return the id
   */
  virtual size_t getId() const = 0;

  /**
   * Recv event.
   *
   * @param inputIndex is the index of the input on which the event e is received
   * @param e the event to be received
   */
  virtual void recvEvent(size_t inputIndex, const EventType& e) = 0;

  /**
   * Exec event.  
   *
   * The user code should override this method inorder to
   * define the semantics of executing and event on
   *
   * @param graph: the graph whose nodes contain simulation objects and edges model the
   * communication links
   *
   * @param myNode the node in the graph that has this SimObject as its node data
   * @param e the input event
   */
  virtual void execEvent(Graph& graph, GNode& myNode, const EventType& e) = 0;

}; // end class

#endif /* SIMOBJECT_H_ */
