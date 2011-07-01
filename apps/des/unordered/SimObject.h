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
   * @param myNode the node in the graph that has this SimObject as its node data
   * @return number of events ready to be executed.
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

  /* (non-Javadoc)
   * @see java.lang.Object#toString()
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
   * @param in the in
   * @param e the e
   */
  virtual void recvEvent(size_t inputIndex, const EventType& e) = 0;

  // The user code should override this method inorder to
  // define the semantics of executing and event on
  /**
   * Exec event.
   *
   * @param myNode the node in the graph that has this SimObject as its node data
   * @param e the input event
   */
  virtual void execEvent(Graph& graph, GNode& myNode, const EventType& e) = 0;

}; // end class

#endif /* SIMOBJECT_H_ */
