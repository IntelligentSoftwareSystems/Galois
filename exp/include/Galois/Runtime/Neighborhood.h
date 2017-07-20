/** TODO -*- C++ -*-
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
 * @section Description
 *
 * TODO 
 *
 * @author <ahassaan@ices.utexas.edu>
 */


#ifndef GALOIS_RUNTIME_NEIGHBORHOOD_H
#define GALOIS_RUNTIME_NEIGHBORHOOD_H

#include <vector>
#include <algorithm>

#include <cassert>

#include "Galois/MethodFlags.h"
#include "Galois/Atomic.h"

#include "Galois/Runtime/Context.h"
#include "Galois/Runtime/ll/PaddedLock.h"

namespace Galois::Runtime {

template <typename NItem>
struct NhoodIDcmp {
  bool operator () (const NItem* left, const NItem* right) const {
    return left->getID () < right->getID ();
  }
};

template <typename T, typename NItem>
struct NhoodListContext: public SimpleRuntimeContext {
  typedef std::vector<NItem*> NhoodList;
  
  T active;
  NhoodList nhood;

  // NhoodListContext (): active (), nhood () {}

  explicit NhoodListContext (const T& _active): active (_active), nhood () {}

  template <bool sort_tp>
  void addItem (NItem* item) {
    nhood.push_back (item);

    if (sort_tp) {
    std::inplace_merge (nhood.begin (), nhood.end ()-1, nhood.end (), NhoodIDcmp<NItem> ()); 
    }
    
  }

  template <typename Cmp>
  bool isSrc (Cmp& cmp) const {
    // TODO: remove later
    assert (nhood.size () == 3); // for AVI
    bool ret = true;
    for (typename NhoodList::const_iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {
      if (!(*n)->isHighestPriority (this, cmp)) {
        ret = false;
        break;
      }
    }

    return ret;
  }

  bool isLeader (const NItem* ni) const {
    assert (!nhood.empty ());
    assert (std::find (nhood.begin (), nhood.end (), ni) != nhood.end ());
    return ni->getID () == (*nhood.begin ())->getID ();
  }

  void removeFromNhood () {
    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      (*n)->remove (this);
    }

    nhood.clear ();
    
  }

  void printNhood () {
    std::cout << this->active->toString () << " neighborhood: " << std::endl;
    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      (*n)->print (std::cout);

    }
    std::cout << std::endl;

  }
}; 

// Neighborhood items store pointers to contexts
// instead of pointers to active elements. 


template <typename T>
struct NhoodItemShareList: public Lockable {
  typedef NhoodItemShareList<T> MyType;
  typedef NhoodListContext<T, MyType> Ctxt;


  typedef Galois::Runtime::LL::SimpleLock<true> Lock_ty;

  typedef std::vector<Ctxt*> ShareList;


  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE Lock_ty sharersLock;
  size_t id;
  ShareList sharers; 
  
public:
  NhoodItemShareList(size_t _id): 
    Lockable (), id (_id), sharers (), sharersLock () {}

  size_t getID () const { return id; }

  template <typename C>
  void visit (C& cmp, Galois::MethodFlag flag=Galois::NONE) {
    Galois::Runtime::SimpleRuntimeContext* c = Galois::Runtime::getThreadContext ();
    assert (c != NULL);
    Ctxt* ctxt = static_cast<Ctxt*> (c);
    assert (ctxt != NULL);
    acquire (ctxt, cmp);
    ctxt->template addItem<true> (this);
  }


  template <typename C>
  void acquire (Ctxt* ctxt, const C& cmp) {
    sharersLock.lock ();

    assert (std::find (sharers.begin (), sharers.end (), ctxt) == sharers.end ());
    sharers.push_back (ctxt);

    sharersLock.unlock ();
  }

  template <typename C>
  bool isHighestPriority (const Ctxt* ctxt, C& cmp) const {
    bool ret = true;

    sharersLock.lock ();

    for (typename ShareList::const_iterator i = sharers.begin ()
        , endi = sharers.end (); i != endi; ++i) {

      if (cmp ((*i)->active, ctxt->active)) { // (*i)->active < ctxt->active
        ret = false;
        break;
      }
    }

    sharersLock.unlock ();

    return ret;
  }

  template <typename C>
  Ctxt* getHighestPriority (C& cmp) const {

    Ctxt* ctxt = NULL;
    sharersLock.lock ();

    if (!sharers.empty ()) {

      ctxt = *(sharers.begin ());

      for (typename ShareList::const_iterator i = sharers.begin ()
          , endi = sharers.end (); i != endi; ++i) {

        if (cmp ((*i)->active, ctxt->active)) { // (*i)->active < ctxt->active
          ctxt = *i;
        }
      }
    }

    sharersLock.unlock ();

    return ctxt;
  }


  void remove (Ctxt* ctxt) {
    sharersLock.lock ();

    typename ShareList::iterator new_end = std::remove (sharers.begin (), sharers.end (), ctxt);
    assert (new_end != sharers.end ());
    sharers.erase (new_end, sharers.end ());

    sharersLock.unlock ();
  }

  void print (std::ostream& out=std::cout) {
    sharersLock.lock ();

    std::cout << getID () << ": [ ";
    for (typename ShareList::iterator c = sharers.begin ()
        , endc = sharers.end (); c != endc; ++c) {
      std::cout << (*c)->active->toString () << ", ";
    }
    std::cout << "]" << std::endl;

    sharersLock.unlock ();
  }

};

template <typename T>
class NhoodItemPriorityLock: public Lockable {

  typedef NhoodItemPriorityLock<T> MyType;
  typedef NhoodListContext<T, MyType> Ctxt;
  typedef Galois::GAtomic<Ctxt*> AtomicPtr;

  GALOIS_ATTRIBUTE_ALIGN_CACHE_LINE AtomicPtr highest;
  size_t id;

public:

  NhoodItemPriorityLock (size_t _id): Lockable (), id (_id), highest (NULL) {}

  size_t getID () const { return id; }

  template <typename C>
  void visit (C& cmp, Galois::MethodFlag flat=Galois::NONE) {
    Galois::Runtime::SimpleRuntimeContext* c = Galois::Runtime::getThreadContext ();
    assert (c != NULL);
    Ctxt* ctxt = static_cast<Ctxt*> (c);
    assert (ctxt != NULL);

    acquire (ctxt, cmp);

    ctxt->template addItem<false> (this);
  }

  template <typename C>
  bool acquire (Ctxt* ctxt, const C& cmp) {
    assert (ctxt != NULL);

    bool succ = false;

    if (highest == NULL) {
      succ = highest.cas (NULL, ctxt);
    }

    assert (highest != NULL);

    for (Ctxt* curr = highest; cmp (ctxt->active, curr->active); curr = highest) {
      succ = highest.cas (curr, ctxt);
    }

    return succ;
  }

  template <typename C>
  bool isHighestPriority (const Ctxt* ctxt, const C& cmp) {
    assert (ctxt != NULL);
    // assert (highest != NULL);
    return (highest == ctxt);
  }

  void remove (Ctxt* ctxt) {
    assert (ctxt != NULL);
    if (highest == ctxt) {
      highest = NULL;
    }
  }

};

} // end namespace Galois::Runtime


#endif // GALOIS_RUNTIME_NEIGHBORHOOD_H

