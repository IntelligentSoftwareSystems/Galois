#ifndef ORDERED_SIM_OBJECT_H
#define ORDERED_SIM_OBJECT_H

#include <sstream>
#include <iostream>
#include <vector>

#include "comDefs.h"
#include "BaseSimObject.h"
#include "Event.h"

namespace des_ord {

template <typename Event_tp>
class SimObject: public des::BaseSimObject<Event_tp> {

  typedef des::BaseSimObject<Event_tp> Base;
  typedef Event_tp Event_ty;

  template <typename AddNewFunc> 
  struct SendAddList: public Base::SendWrapper {
    AddNewFunc& addNewFunc;

    SendAddList (AddNewFunc& _container): Base::SendWrapper (), addNewFunc (_container) {}

    virtual void send (Base* dstObj, const Event_ty& event) {
      addNewFunc (event);
    }
  };


public:

  SimObject (size_t id, unsigned numOutputs, unsigned numInputs)
    : Base (id) 
  {}

  template <typename G, typename AddNewFunc>
  void execEvent (
      const Event_ty& event, 
      G& graph, 
      typename G::GraphNode& mynode, 
      AddNewFunc& newEvents) {

    assert (event.getRecvObj () == this);

    typename Base::template OutDegIterator<G> beg = this->make_begin (graph, mynode);
    typename Base::template OutDegIterator<G> end = this->make_end (graph, mynode);

    SendAddList<AddNewFunc> addListWrap (newEvents);
    this->execEventIntern (event, addListWrap, beg, end);
  }

  virtual size_t getStateSize () const = 0;
  
  virtual void copyState (void* const buf, const size_t bufSize) const = 0; 

  virtual void restoreState (void* const buf, const size_t bufSize) = 0;
};

} // end namespace des_ord

#endif // ORDERED_SIM_OBJECT_H
