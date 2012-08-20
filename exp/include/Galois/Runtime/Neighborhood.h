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

namespace GaloisRuntime {

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

  void addItem (NItem* item) {
    nhood.push_back (item);
    std::inplace_merge (nhood.begin (), nhood.end ()-1, nhood.end (), NhoodIDcmp<NItem> ()); 
    
  }

  template <typename Cmp>
  bool isSrc (Cmp& cmp) const {
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

    //TODO: should clear nhood list as well after this operation
    
  }

  void printNhood () {
    std::cout << this->active->toString () << " neighborhood: " << std::endl;
    for (typename NhoodList::iterator n = nhood.begin ()
        , endn = nhood.end (); n != endn; ++n) {

      std::cout << (*n)->getID () << ": [ ";
      for (typename NItem::ShareList::iterator c = (*n)->sharers.begin ()
          , endc = (*n)->sharers.end (); c != endc; ++c) {
        std::cout << (*c)->active->toString () << ", ";
      }
      std::cout << "]" << std::endl;

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


  // typedef GaloisRuntime::LL::PaddedLock<true> Lock_ty;
  typedef GaloisRuntime::LL::SimpleLock<true> Lock_ty;

  typedef std::vector<Ctxt*> ShareList;


  size_t id;
  ShareList sharers; 
  Lock_ty sharersLock;
  
public:
  NhoodItemShareList(size_t _id): 
    Lockable (), id (_id), sharers (), sharersLock () {}

  size_t getID () const { return id; }

  template <typename C>
  void visit (C& cmp, Galois::MethodFlag flag=Galois::NONE) {
    GaloisRuntime::SimpleRuntimeContext * c = GaloisRuntime::getThreadContext ();
    assert (c != NULL);
    Ctxt* ctxt = static_cast<Ctxt*> (c);
    assert (ctxt != NULL);
    acquire (ctxt, cmp);
    ctxt->addItem (this);
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

};

// template <typename T>
// class NhoodItemPriorityLock: public Lockable {
// 
  // typedef Galois::GAtomicPadded<T*> AtomicPtr;
  // AtomicPtr highest;
// 
  // NhoodItemPriorityLock (): Lockable (), highest (NULL) {}
// 
  // template <typename C>
  // bool acquire (T* active, const C& cmp) {
    // assert (active != NULL);
// 
    // bool succ = false;
// 
    // if (highest == NULL) {
      // succ = highest.cas (NULL, active);
    // }
// 
    // assert (highest != NULL);
// 
    // for (AtomicPtr curr = highest; cmp (active, curr); curr = highest) {
      // succ = highest.cas (curr, active);
    // }
// 
    // return succ;
  // }
// 
  // template <typename C>
  // bool isHighestPriority (T* active, const C& cmp) {
    // return (highest == active);
  // }
// 
// 
// };

} // end namespace GaloisRuntime


#endif // GALOIS_RUNTIME_NEIGHBORHOOD_H

