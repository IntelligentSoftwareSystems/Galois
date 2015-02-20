#ifndef GALOIS_RUNTIME_CUSTOM_LOCKABLE_H
#define GALOIS_RUNTIME_CUSTOM_LOCKABLE_H


namespace Galois {
namespace Runtime {

/**
 * NItem inherits from LockManagerBase
 *
 * NItem implements the interface:
 *
 * bool tryMappingTo (Lockable* l);
 *
 * void clearMapping (void);
 *
 * const Lockable* getMapping (void) const;
 *
 * static NItem* getOwner (Lockable* l);
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
  NItemFactory factory;
  NItemWL allNItems;
  
public:
  PtrBasedNhoodMgr(const NItemFactory& f): factory (f) {}

  NItem& getNhoodItem (Lockable* l) {

    if (NItem::getOwner (l) == NULL) {
      // NItem* ni = new NItem (l, cmp);
      NItem* ni = factory.create (l);

      if (ni->tryMappingTo (l)) {
        allNItems.get ().push_back (ni);

      } else {
        factory.destroy (ni);
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
      self->factory.destroy(ni);
    }
  };

  void resetAllNItems() {
    Reset fn = { this };
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



#endif // GALOIS_RUNTIME_CUSTOM_LOCKABLE_H
