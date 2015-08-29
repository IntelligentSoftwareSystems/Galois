#ifndef BILLIARDS_PARALLEL_SECTORED_H
#define BILLIARDS_PARALLEL_SECTORED_H

// TODO:
// may need finer work-items consisting of a pair of events (contexts) to be tested. 


struct DepTestUtils {

  static const unsigned COARSE_CHUNK_SIZE = 1;

  template <typename CR, typename Cmp>
  static void testOnRange (const CR& crange, const Cmp& cmp, const char* const loopname) {

    Galois::do_all_choice (crange,
        [&] (C_ptr ctxt) {

          bool indep = true;

          for (auto i = crange.global_begin (), end_i = crange.global_end (); i != end_i; ++i) {

            if ((ctxt != *i) 
              && !cmp (ctxt, *i) // if ctxt >= i
              && OrderDepTest::dependsOn (ctxt->getElem (), (*i)->getElem ()) { 

              indep = false;
              ctxt->disableSrc ();
              break;
            }
          }
        },
        loopname,
        Galois::chunk_size<COARSE_CHUNK_SIZE> ());

    return indep;
  }

  template <typename CI, typename Cmp, typename B>
  static void selfTestRange (const CI& beg, const CI& end, const Cmp& cmp, B& safeBag) {

    for (auto i = beg; i != end; ++i) {

      bool indep = true;
      for (auto j = beg; j != end; ++j) {
        if (i != j 
          && !cmp (*i, *j) // *i >= *j
          && OrderDepTest::dependsOn (*i, *j)) { // *i depends on *j

          indep = false;
          (*i)->disableSrc ();
          break;
        }
      }


      if (indep) {
        assert ((*i)->isSrc ());
        safeBag.push (*i);
      }

    }
  }


};

struct FlatTest {
  template <typename CR, typename Cmp>
  void operator (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;

    testOnRange (crange, cmp, "flat-indep-test");

  }
};


struct ThreadLocalTest {

  template <typename CR, typename Cmp>
  void operator (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = Galois::PerThreadBag<C_ptr, 64>;

    Bag_ty localSafeEvents;

    // TODO: try do-all with fine grained work items. 
    Galois::Runtime::on_each_impl (
        [&] (const unsigned tid, const unsigned numT) {

          selfTestRange (crange.begin_local (), crange.end_local (), cmp, localSafeEvents);

        }
        , "thread-local-safety-test");


    testOnRange (Galois::Runtime::makeLocalRange (localSafeEvents), cmp, "thread-local-round-2");

  }
};

struct SectorLocalTest {



  template <typename CR, typename Cmp>
  void operator (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = Galois::PerThreadBag<C_ptr, 64>;

    const size_t numSectors = table.getNumSectors ();

    std::vector<Bag_ty> sectorBags (numSectors);

    Galois::do_all_choice (crange,
        [] (C_ptr ctxt) {
          
          const Event& e = ctxt->getElem ();
          assert (e.getSector () != nullptr && "event without sector info");
          size_t secID = e.getSector ()->getID ();
          assert (secID < sectorBags.size ());
          sectorBags [secID].push (ctxt);
        },
        "bin-by-sector",
        Galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    auto secRange = Galois::Runtime::makeStandardRange (boost::counting_iterator<size_t> (0),
        boost::counting_iterator<size_t> (numSectors));

    Bag_ty perSectorSafeEvents;

    Galois::do_all_choice (secRange,
        [&] (const size_t secID) {
          DepTestUtils::selfTestRange (
            sectorBags[secID].global_begin (), 
            sectorBags[secID].global_end (), 
            cmp, 
            perSectorSafeEvents);
        },
        "per-sector-test",
        Galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());


    testOnRange (Galois::Runtime::makeLocalRange (localSafeEvents), cmp, "inter-sector-test");


  }


};

struct SectorLocalThreadLocalTest {

  template <typename CR, typename Cmp>
  void operator (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = Galois::PerThreadBag<C_ptr, 64>;

    const size_t numSectors = table.getNumSectors ();

    std::vector<Bag_ty> sectorBags (numSectors);

    Galois::do_all_choice (crange,
        [] (C_ptr ctxt) {
          
          const Event& e = ctxt->getElem ();
          assert (e.getSector () != nullptr && "event without sector info");
          size_t secID = e.getSector ()->getID ();
          assert (secID < sectorBags.size ());
          sectorBags [secID].push (ctxt);
        },
        "bin-by-sector",
        Galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    auto secRange = Galois::Runtime::makeStandardRange (boost::counting_iterator<size_t> (0),
        boost::counting_iterator<size_t> (numSectors));

    
    std::vector<Bag_ty> perThrdSectorLocalEvents;

    const size_t numT = Galois::getActiveThreads ();
    using ThrdSecPair = std::pair<size_t, size_t>;
    std::vector<ThrdSecPair> thrdSecPairs;

    for (size_t i = 0; i < numSectors; ++i) {
      for (size_t j = 0; j < numT; ++j) {
        thrdSecPairs.push_back (std::make_pair (i, j));
      }
    }


    Galois::do_all_choice (Galois::Runtime::makeStandardRange (thrdSecPairs.begin (), thrdSecPairs.end ()),
        [&] (const ThrdSecPair& p) {
          const size_t secID = p.first;
          const size_t tid = p.second;

          assert (secID < numSectors);
          assert (tid < Galois::getActiveThreads ());


          DepTestUtils::selfTestRange (
            sectorBags[secID].get (tid).begin (),
            sectorBags[secID].get (tid).end (),
            cmp,
            perThrdSectorLocalEvents[secID]);
        },
        "thread-local-per-sector-test",
        Galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    Bag_ty perSectorSafeEvents;

    Galois::do_all_choice (secRange,
        [&] (const size_t secID) {
          DepTestUtils::selfTestRange (
            perThrdSectorLocalEvents[secID].global_begin (), 
            perThrdSectorLocalEvents[secID].global_end (), 
            cmp, 
            perSectorSafeEvents);
        },
        "per-sector-test",
        Galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    testOnRange (Galois::Runtime::makeLocalRange (localSafeEvents), cmp, "inter-sector-test");

  }

};



#endif //  BILLIARDS_PARALLEL_SECTORED_H
