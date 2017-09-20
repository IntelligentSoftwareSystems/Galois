/** ?? -*- C++ -*-
 * @file
 * @section License
 *
 * This file is part of Galois.  Galoisis a framework to exploit
 * amorphous data-parallelism in irregular programs.
 *
 * Galois is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, version 2.1 of the
 * License.
 *
 * Galois is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Galois.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 * @section Copyright
 *
 * Copyright (C) 2015, The University of Texas at Austin. All rights
 * reserved.
 *
 */

#ifndef GALOIS_RUNTIME_ORDERED_LOCKABLE_H
#define GALOIS_RUNTIME_ORDERED_LOCKABLE_H

#include "Galois/AltBag.h"
#include "Galois/OrderedTraits.h"
#include "Galois/Runtime/Context.h"
#include "Galois/Substrate/ThreadRWlock.h"
#include "Galois/Runtime/UserContextAccess.h"

#include <unordered_map>

namespace galois {
namespace runtime {

using dbg = galois::Substrate::debug<1>;

template <typename T>
class OrderedContextBase: public SimpleRuntimeContext {
  using Base = SimpleRuntimeContext;

protected:

  T active;

public:

  explicit OrderedContextBase (const T& x): 
    Base (true), // call overriden subAcquire
    active (x)
  {}

  const T& getActive (void) const { return active; }

  // XXX: disable this. It will only work for modifications that don't change the priority
  T& getActive () { return active; }

  operator const T& (void) const { return getActive (); }

  operator T (void) const { return getActive (); }

  // XXX: disable this. It will only work for modifications that don't change the priority
  operator T& (void) const { return getActive (); }

};

// TODO: change comparator to three valued int instead of bool
template <typename Ctxt, typename Cmp>
struct ContextComparator {
  const Cmp& cmp;

  explicit ContextComparator (const Cmp& cmp): cmp (cmp) {}

  inline bool operator () (const Ctxt* left, const Ctxt* right) const {
    assert (left != NULL);
    assert (right != NULL);
    return cmp (left->getActive (), right->getActive ());
  }
};


template <typename T, typename Cmp>
class TwoPhaseContext: public OrderedContextBase<T> {

public:

  using Base = OrderedContextBase<T>;
  // using NhoodList =  galois::gdeque<Lockable*, 4>;
  using CtxtCmp = ContextComparator<TwoPhaseContext, Cmp>;

  CtxtCmp ctxtCmp;
  bool source = true;


  using value_type = T;

  explicit TwoPhaseContext (const T& x, const Cmp& cmp)
    : 
      Base (x),  // pass true so that Base::acquire invokes virtual subAcquire
      ctxtCmp (cmp),
      source (true) 
  {}

  bool isSrc (void) const {
    return source;
  }

  void disableSrc (void) {
    source = false;
  }

  void reset () { 
    source = true;
  }

  void enableSrc (void) {
    source = true;
  }

  virtual void subAcquire (Lockable* l, galois::MethodFlag) {


    if (Base::tryLock (l)) {
      Base::addToNhood (l);
    }

    TwoPhaseContext* other = nullptr;

    do {
      other = static_cast<TwoPhaseContext*> (Base::getOwner (l));

      if (other == this) {
        return;
      }

      if (other) {
        bool conflict = ctxtCmp (other, this); // *other < *this
        if (conflict) {
          // A lock that I want but can't get
          this->source = false;
          return; 
        }
      }
    } while (!this->stealByCAS(l, other));

    // Disable loser
    if (other) {
      other->source = false; // Only need atomic write
    }

    return;


    // bool succ = false;
    // if (Base::tryAcquire (l) == Base::NEW_OWNER) {
      // Base::addToNhood (l);
      // succ = true;
    // }
// 
    // assert (Base::getOwner (l) != NULL);
// 
    // if (!succ) {
      // while (true) {
        // TwoPhaseContext* that = static_cast<TwoPhaseContext*> (Base::getOwner (l));
// 
        // assert (that != NULL);
        // assert (this != that);
// 
        // if (PtrComparator::compare (this, that)) { // this < that
          // if (Base::stealByCAS (that, this)) {
            // that->source = false;
            // break;
          // }
// 
        // } else { // this >= that
          // this->source = false; 
          // break;
        // }
      // }
    // } // end outer if
  } // end subAcquire

  virtual bool owns (Lockable* l, MethodFlag m) const {
    return (static_cast<TwoPhaseContext*> (Base::getOwner(l)) == this);
  }


};

template <typename NItem, typename Ctxt, typename CtxtCmp>
struct OrdLocFactoryBase {

  CtxtCmp ctxtcmp;

  explicit OrdLocFactoryBase (const CtxtCmp& ctxtcmp): ctxtcmp (ctxtcmp) {}

  void construct (NItem* ni, Lockable* l) const {
    assert (ni != nullptr);
    assert (l != nullptr);

    new (ni) NItem (l, ctxtcmp);
  }
};

template <typename NItem, typename Ctxt, typename CtxtCmp>
struct OrdLocBase: public LockManagerBase {

  using Base = LockManagerBase;

  using Factory = OrdLocFactoryBase<NItem, Ctxt, CtxtCmp>;

  Lockable* lockable;

  explicit OrdLocBase (Lockable* l): 
    Base (), lockable (l) {}

  bool tryMappingTo (Lockable* l) {
    return Base::CASowner (l, NULL);
  }

  void clearMapping () {
    // release requires having owned the lock
    bool r = Base::tryLock (lockable);
    assert (r);
    Base::release (lockable);
  }

  // just for debugging
  const Lockable* getMapping () const {
    return lockable;
  }

  static NItem* getOwner (Lockable* l) {
    LockManagerBase* o = LockManagerBase::getOwner (l);
    // assert (dynamic_cast<DAGnhoodItem*> (o) != nullptr);
    return static_cast<NItem*> (o);
  }
};

/**
 * NItem inherits from OrdLocBase publicly
 *
 * NItem contains nested type Factory
 *
 * Factory implements interface:
 *
 * NItem* create (Lockable* l);
 *
 * void destroy (NItem* ni);
 *
*/

template<typename NItem>
class PtrBasedNhoodMgr: boost::noncopyable {
public:
  typedef typename NItem::Factory NItemFactory;

  typedef FixedSizeAllocator<NItem> NItemAlloc;
  typedef galois::PerThreadBag<NItem*> NItemWL;

protected:
  NItemAlloc niAlloc;
  NItemFactory& factory;
  NItemWL allNItems;

  NItem* create (Lockable* l) {
    NItem* ni = niAlloc.allocate (1);
    assert (ni != nullptr);
    factory.construct (ni, l);
    return ni;
  }

  void destroy (NItem* ni) {
    // delete ni; ni = NULL;
    niAlloc.destroy (ni);
    niAlloc.deallocate (ni, 1);
    ni = NULL;
  }

  
public:
  PtrBasedNhoodMgr(NItemFactory& f): factory (f) {}

  NItem& getNhoodItem (Lockable* l) {

    if (NItem::getOwner (l) == NULL) {
      // NItem* ni = new NItem (l, cmp);
      NItem* ni = create (l);

      if (ni->tryMappingTo (l)) {
        allNItems.get ().push_back (ni);

      } else {
        destroy (ni);
      }

      assert (NItem::getOwner (l) != NULL);
    }

    NItem* ret = NItem::getOwner (l);
    assert (ret != NULL);
    return *ret;
  }

  LocalRange<NItemWL> getAllRange (void) {
    return makeLocalRange (allNItems);
  }

  NItemWL& getContainer() {
    return allNItems;
  }

  ~PtrBasedNhoodMgr() {
    resetAllNItems();
  }

protected:
  void resetAllNItems() {
    do_all_choice(makeLocalRange(allNItems), 
        [this] (NItem* ni) {
          ni->clearMapping();
          destroy(ni);
        },
        std::make_tuple (
          galois::loopname ("resetNItems"), 
          galois::chunk_size<16>()));
  }
};

template <typename NItem>
class MapBasedNhoodMgr: public PtrBasedNhoodMgr<NItem> {
public:
  typedef MapBasedNhoodMgr MyType;

  // typedef std::tr1::unordered_map<Lockable*, NItem> NhoodMap; 
  //
  typedef Pow_2_BlockAllocator<std::pair<Lockable*, NItem*> > MapAlloc;

  typedef std::unordered_map<
      Lockable*,
      NItem*,
      std::hash<Lockable*>,
      std::equal_to<Lockable*>,
      MapAlloc
    > NhoodMap;

  typedef galois::Substrate::ThreadRWlock Lock_ty;
  typedef PtrBasedNhoodMgr<NItem> Base;

protected:
  NhoodMap nhoodMap;
  Lock_ty map_mutex;

public:

  MapBasedNhoodMgr (const typename Base::NItemFactory& f): 
    Base (f),
    nhoodMap (8, std::hash<Lockable*> (), std::equal_to<Lockable*> (), MapAlloc ())

  {}

  NItem& getNhoodItem (Lockable* l) {

    map_mutex.readLock ();
      typename NhoodMap::iterator i = nhoodMap.find (l);

      if (i == nhoodMap.end ()) {
        // create the missing entry

        map_mutex.readUnlock ();

        map_mutex.writeLock ();
          // check again to avoid over-writing existing entry
          if (nhoodMap.find (l) == nhoodMap.end ()) {
            NItem* ni = Base::factory.create (l);
            Base::allNItems.get ().push_back (ni);
            nhoodMap.insert (std::make_pair (l, ni));

          }
        map_mutex.writeUnlock ();

        // read again now
        map_mutex.readLock ();
        i = nhoodMap.find (l);
      }

    map_mutex.readUnlock ();
    assert (i != nhoodMap.end ());
    assert (i->second != nullptr);

    return *(i->second);
    
  }


  ~MapBasedNhoodMgr () {
    Base::resetAllNItems ();
  }

};



namespace HIDDEN {
  
  struct DummyExecFunc {
    static const unsigned CHUNK_SIZE = 1;
    template <typename T, typename C>
    void operator () (const T&, C&) const {
      std::printf ("Warning: DummyExecFunc shouldn't be executed\n");
    }
  };
}


template <typename T, typename Cmp, typename NhFunc, typename ExFunc, typename OpFunc, typename ArgsTuple, typename Ctxt> 
class OrderedExecutorBase {
protected:

  static const bool NEEDS_CUSTOM_LOCKING = exists_by_supertype<needs_custom_locking_tag, ArgsTuple>::value;
  static const bool HAS_EXEC_FUNC = exists_by_supertype<has_exec_function_tag, ArgsTuple>::value 
    || !std::is_same<ExFunc, HIDDEN::DummyExecFunc>::value;

  static const bool ENABLE_PARAMETER = get_type_by_supertype<enable_parameter_tag, ArgsTuple>::type::value;
  static const bool NEEDS_PUSH = !exists_by_supertype<does_not_need_push_tag, ArgsTuple>::value;

  using CtxtCmp = typename Ctxt::CtxtCmp;
  using CtxtAlloc = FixedSizeAllocator<Ctxt>;
  using CtxtWL = PerThreadBag<Ctxt*>;

  using UserCtxt = UserContextAccess<T>;
  using PerThreadUserCtxt = Substrate::PerThreadStorage<UserCtxt>;


  Cmp cmp;
  NhFunc nhFunc;
  ExFunc exFunc;
  OpFunc opFunc;
  const char* loopname;

  CtxtCmp ctxtCmp;

  CtxtAlloc ctxtAlloc;
  PerThreadUserCtxt userHandles;

  OrderedExecutorBase (const Cmp& cmp, const NhFunc& nhFunc, const ExFunc& exFunc, const OpFunc& opFunc, const ArgsTuple& argsTuple)
    : 
      cmp (cmp), 
      nhFunc (nhFunc), 
      exFunc (exFunc),
      opFunc (opFunc), 
      loopname (get_by_supertype<loopname_tag> (argsTuple).value),
      ctxtCmp (cmp)
  {
    if (!loopname) { loopname = "Ordered"; }
  }

public:
  const Cmp& getItemCmp () const { return cmp; }

  const CtxtCmp& getCtxtCmp () const { return ctxtCmp; }
};



template <typename F, typename Ctxt, typename UserCtxt, typename... Args>
void runCatching (F& func, Ctxt* c, UserCtxt& uhand, Args&&... args) {
  galois::runtime::setThreadContext (c);

  int result = 0;

#ifdef GALOIS_USE_LONGJMP
  if ((result = setjmp(hackjmp)) == 0) {
#else
    try {
#endif
      func (c->getActive (), uhand, std::forward<Args> (args)...);

#ifdef GALOIS_USE_LONGJMP
    } else {
      // TODO
    }
#else 
  } catch (ConflictFlag f) {
    result = f;
  }
#endif

  switch (result) {
    case 0:
      break;
    case CONFLICT: 
      c->disableSrc ();
      break;
    default:
      GALOIS_DIE ("can't handle conflict flag type");
      break;
  }


  galois::runtime::setThreadContext (NULL);
}



} // end namespace runtime
} // end namespace galois



#endif // GALOIS_RUNTIME_ORDERED_LOCKABLE_H
