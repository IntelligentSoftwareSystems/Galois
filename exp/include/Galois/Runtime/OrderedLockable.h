#ifndef GALOIS_RUNTIME_ORDERED_LOCKABLE_H
#define GALOIS_RUNTIME_ORDERED_LOCKABLE_H

#include "Galois/AltBag.h"
#include "Galois/Runtime/ll/ThreadRWlock.h"

#include <unordered_map>

namespace Galois {
namespace Runtime {

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

  typedef MM::FixedSizeAllocator<NItem> NItemAlloc;
  typedef Galois::PerThreadBag<NItem*> NItemWL;

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
  struct Reset {
    PtrBasedNhoodMgr* self; 
    void operator()(NItem* ni) const {
      ni->clearMapping();
      self->destroy(ni);
    }
  };

  void resetAllNItems() {
    Reset fn {this};
    do_all_impl(makeLocalRange(allNItems), fn);
  }
};

template <typename NItem>
class MapBasedNhoodMgr: public PtrBasedNhoodMgr<NItem> {
public:
  typedef MapBasedNhoodMgr MyType;

  // typedef std::tr1::unordered_map<Lockable*, NItem> NhoodMap; 
  //
  typedef MM::Pow_2_BlockAllocator<std::pair<Lockable*, NItem*> > MapAlloc;

  typedef std::unordered_map<
      Lockable*,
      NItem*,
      std::hash<Lockable*>,
      std::equal_to<Lockable*>,
      MapAlloc
    > NhoodMap;

  typedef Galois::Runtime::LL::ThreadRWlock Lock_ty;
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

} // end namespace Runtime
} // end namespace Galois



#endif // GALOIS_RUNTIME_ORDERED_LOCKABLE_H
