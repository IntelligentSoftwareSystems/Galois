#ifndef BILLIARDS_PARALLEL_SECTORED_H
#define BILLIARDS_PARALLEL_SECTORED_H

#include "galois/AltBag.h"

// TODO:
// may need finer work-items consisting of a pair of events (contexts) to be tested. 


struct DepTestUtils {

  static const unsigned COARSE_CHUNK_SIZE = 1;

  template <typename CR, typename Cmp>
  static void testOnRange (const CR& crange, const Cmp& cmp, const char* const loopname) {

    using C_ptr = typename CR::value_type;
    galois::do_all_choice (crange,
        [&] (C_ptr ctxt) {

          bool indep = true;

          for (auto i = crange.begin (), end_i = crange.end (); i != end_i; ++i) {

            if ((ctxt != *i) 
              && !cmp (ctxt, *i) // if ctxt >= i
              && OrderDepTest::dependsOn (ctxt->getElem (), (*i)->getElem ())) { 

              indep = false;
              ctxt->disableSrc ();
              break;
            }
          }
        },
        loopname,
        galois::chunk_size<COARSE_CHUNK_SIZE> ());

  }

  template <typename CI, typename Cmp, typename B>
  static void selfTestRange (const CI& beg, const CI& end, const Cmp& cmp, B& safeBag) {

    for (auto i = beg; i != end; ++i) {

      bool indep = true;
      for (auto j = beg; j != end; ++j) {
        if (i != j 
          && !cmp (*i, *j) // *i >= *j
          && OrderDepTest::dependsOn ((*i)->getElem (), (*j)->getElem ())) { // *i depends on *j

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

template <typename Tbl_t>
struct FlatTest {

  Tbl_t& table;

  template <typename CR, typename Cmp>
  void operator () (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;

    DepTestUtils::testOnRange (crange, cmp, "flat-indep-test");

  }
};


template <typename Tbl_t>
struct ThreadLocalTest {

  Tbl_t& table;

  template <typename CR, typename Cmp>
  void operator () (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = galois::PerThreadBag<C_ptr, 64>;

    Bag_ty localSafeEvents;

    // TODO: try do-all with fine grained work items. 
    galois::runtime::on_each_impl (
        [&] (const unsigned tid, const unsigned numT) {

          DepTestUtils::selfTestRange (crange.begin_local (), crange.end_local (), cmp, localSafeEvents);

        }
        , "thread-local-safety-test");


    DepTestUtils::testOnRange (galois::runtime::makeLocalRange (localSafeEvents), cmp, "thread-local-round-2");

  }
};

template <typename Tbl_t>
struct SectorLocalTest {

  Tbl_t& table;

  template <typename CR, typename Cmp>
  void operator () (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = galois::PerThreadBag<C_ptr, 64>;

    const size_t numSectors = table.getNumSectors ();

    std::vector<Bag_ty> sectorBags (numSectors);

    galois::do_all_choice (crange,
        [&] (C_ptr ctxt) {
          
          const Event& e = ctxt->getElem ();
          assert (e.enclosingSector () != nullptr && "event without sector info");
          size_t secID = e.enclosingSector ()->getID ();
          assert (secID < sectorBags.size ());
          sectorBags [secID].push (ctxt);
        },
        "bin-by-sector",
        galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    auto secRange = galois::runtime::makeStandardRange (boost::counting_iterator<size_t> (0),
        boost::counting_iterator<size_t> (numSectors));

    Bag_ty perSectorSafeEvents;

    galois::do_all_choice (secRange,
        [&] (const size_t secID) {
          DepTestUtils::selfTestRange (
            sectorBags[secID].begin (), 
            sectorBags[secID].end (), 
            cmp, 
            perSectorSafeEvents);
        },
        "per-sector-test",
        galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());


    DepTestUtils::testOnRange (galois::runtime::makeLocalRange (perSectorSafeEvents), cmp, "inter-sector-test");


  }


};

template <typename Tbl_t>
struct SectorLocalThreadLocalTest {

  Tbl_t& table;

  template <typename CR, typename Cmp>
  void operator () (const CR& crange, const Cmp& cmp) {
    using C_ptr = typename CR::value_type;
    using Bag_ty = galois::PerThreadBag<C_ptr, 64>;

    const size_t numSectors = table.getNumSectors ();

    std::vector<Bag_ty> sectorBags (numSectors);

    galois::do_all_choice (crange,
        [] (C_ptr ctxt) {
          
          const Event& e = ctxt->getElem ();
          assert (e.enclosingSector () != nullptr && "event without sector info");
          size_t secID = e.enclosingSector ()->getID ();
          assert (secID < sectorBags.size ());
          sectorBags [secID].push (ctxt);
        },
        "bin-by-sector",
        galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    auto secRange = galois::runtime::makeStandardRange (boost::counting_iterator<size_t> (0),
        boost::counting_iterator<size_t> (numSectors));

    
    std::vector<Bag_ty> perThrdSectorLocalEvents;

    const size_t numT = galois::getActiveThreads ();
    using ThrdSecPair = std::pair<size_t, size_t>;
    std::vector<ThrdSecPair> thrdSecPairs;

    for (size_t i = 0; i < numSectors; ++i) {
      for (size_t j = 0; j < numT; ++j) {
        thrdSecPairs.push_back (std::make_pair (i, j));
      }
    }


    galois::do_all_choice (galois::runtime::makeStandardRange (thrdSecPairs.begin (), thrdSecPairs.end ()),
        [&] (const ThrdSecPair& p) {
          const size_t secID = p.first;
          const size_t tid = p.second;

          assert (secID < numSectors);
          assert (tid < galois::getActiveThreads ());


          DepTestUtils::selfTestRange (
            sectorBags[secID].get (tid).begin (),
            sectorBags[secID].get (tid).end (),
            cmp,
            perThrdSectorLocalEvents[secID]);
        },
        "thread-local-per-sector-test",
        galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    Bag_ty perSectorSafeEvents;

    galois::do_all_choice (secRange,
        [&] (const size_t secID) {
          DepTestUtils::selfTestRange (
            perThrdSectorLocalEvents[secID].begin (), 
            perThrdSectorLocalEvents[secID].end (), 
            cmp, 
            perSectorSafeEvents);
        },
        "per-sector-test",
        galois::chunk_size<DepTestUtils::COARSE_CHUNK_SIZE> ());

    DepTestUtils::testOnRange (galois::runtime::makeLocalRange (perSectorSafeEvents), cmp, "inter-sector-test");

  }

};



#endif //  BILLIARDS_PARALLEL_SECTORED_H
