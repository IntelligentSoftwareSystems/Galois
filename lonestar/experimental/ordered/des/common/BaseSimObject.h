/**
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of XYZ License (a
 * copy is located in LICENSE.txt at the top-level directory).
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

#ifndef DES_BASE_SIM_OBJECT_H
#define DES_BASE_SIM_OBJECT_H

#include "galois/graphs/Graph.h"

#include <iostream>
#include <sstream>

namespace des {

template <typename Event_tp>
class BaseSimObject {
protected:
  size_t id;
  size_t eventIDcntr;

public:
  typedef Event_tp Event_ty;

  BaseSimObject(size_t id) : id(id), eventIDcntr(0) {}

  virtual ~BaseSimObject() {}

  size_t getID() const { return id; }

  virtual std::string str() const {
    std::ostringstream ss;
    ss << "SimObject-" << id;
    return ss.str();
  }

  virtual BaseSimObject* clone() const = 0;

  Event_ty makeEvent(BaseSimObject* recvObj,
                     const typename Event_ty::Action_ty& action,
                     const typename Event_ty::Type& type,
                     const des::SimTime& sendTime,
                     des::SimTime delay = des::MIN_DELAY) {

    if (delay < des::MIN_DELAY) {
      delay = des::MIN_DELAY;
    }

    des::SimTime recvTime = sendTime + delay;

    assert(recvTime > sendTime);
    assert(recvTime < 2 * des::INFINITY_SIM_TIME);

    return Event_ty((eventIDcntr++), this, recvObj, action, type, sendTime,
                    recvTime);
  }

  Event_ty makeZeroEvent() {
    return Event_ty((eventIDcntr++), this, this, typename Event_ty::Action_ty(),
                    Event_ty::NULL_EVENT, 0, 0);
  }

protected:
  struct SendWrapper {
    virtual void send(BaseSimObject* dstObj, const Event_ty& event) = 0;
    virtual ~SendWrapper() {}
  };

  struct BaseOutDegIter
      : public std::iterator<std::forward_iterator_tag, BaseSimObject*> {
    typedef std::iterator<std::forward_iterator_tag, BaseSimObject*> Super_ty;

    BaseOutDegIter(){};

    virtual typename Super_ty::reference operator*() = 0;

    virtual typename Super_ty::pointer operator->() = 0;

    virtual BaseOutDegIter& operator++() = 0;

    // since BaseOutDegIter is virtual, can't return copy of BaseOutDegIter here
    virtual void operator++(int) = 0;

    virtual bool is_equal(const BaseOutDegIter& that) const = 0;

    friend bool operator==(const BaseOutDegIter& left,
                           const BaseOutDegIter& right) {
      return left.is_equal(right);
    }

    friend bool operator!=(const BaseOutDegIter& left,
                           const BaseOutDegIter& right) {
      return !left.is_equal(right);
    }
  };

  template <typename G>
  struct OutDegIterator : public BaseOutDegIter {
    typedef BaseOutDegIter Base;
    typedef typename G::edge_iterator GI;

    G& graph;
    GI edgeIter;

    OutDegIterator(G& _graph, GI _edgeIter)
        : BaseOutDegIter(), graph(_graph), edgeIter(_edgeIter) {}

    virtual typename Base::reference operator*() {
      return graph.getData(graph.getEdgeDst(edgeIter),
                           galois::MethodFlag::UNPROTECTED);
    }

    virtual typename Base::pointer operator->() { return &(operator*()); }

    virtual OutDegIterator<G>& operator++() {
      ++edgeIter;
      return *this;
    }

    virtual void operator++(int) { operator++(); }

    virtual bool is_equal(const BaseOutDegIter& t) const {

      assert(dynamic_cast<const OutDegIterator<G>*>(&t) != NULL);
      const OutDegIterator<G>& that = static_cast<const OutDegIterator<G>&>(t);

      assert(&that != NULL);
      assert(&(this->graph) == &(that.graph));

      return (this->edgeIter == that.edgeIter);
    }
  };

  template <typename G>
  OutDegIterator<G> make_begin(G& graph, typename G::GraphNode& node) const {
    assert(graph.getData(node, galois::MethodFlag::UNPROTECTED) == this);
    return OutDegIterator<G>(
        graph, graph.edge_begin(node, galois::MethodFlag::UNPROTECTED));
  }

  template <typename G>
  OutDegIterator<G> make_end(G& graph, typename G::GraphNode& node) const {
    assert(graph.getData(node, galois::MethodFlag::UNPROTECTED) == this);
    return OutDegIterator<G>(
        graph, graph.edge_end(node, galois::MethodFlag::UNPROTECTED));
  }

  virtual void execEventIntern(const Event_ty& event, SendWrapper& sendWrap,
                               BaseOutDegIter& b, BaseOutDegIter& e) = 0;

  virtual size_t getInputIndex(const Event_ty& event) const = 0;
};

} // namespace des

#endif // DES_BASE_SIM_OBJECT_H
