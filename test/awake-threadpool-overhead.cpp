
#include "Galois/Galois.h"
#include "Galois/Accumulator.h"
#include "Galois/Threads.h"
#include "Galois/Statistic.h"

#include "Galois/Runtime/DoAll.h"
#include "Galois/Runtime/Range.h"
#include "Galois/Runtime/ThreadPool.h"

#include "Lonestar/BoilerPlate.h"
#include "llvm/Support/CommandLine.h"

#include <boost/iterator/counting_iterator.hpp>

#include <cmath>

#include <iostream>

typedef Galois::GAccumulator<double> AccumDouble;

typedef std::pair<size_t, size_t> IndexRange;

static const bool STEAL = true;


namespace cll = llvm::cl;

static cll::opt<unsigned> VEC_SIZE ("vecsz", cll::desc ("length of vectors"), 
    cll::init (1000 * 1000));

static cll::opt<unsigned> NUM_ROUNDS ("rounds", cll::desc ("number of parallel rounds"), 
    cll::init (100));

enum ExecType {
  DOALL_WAKEUP=0,
  DOALL_EXPLICIT,
};

static const char* EXEC_NAMES[] = { "DOALL_WAKEUP", "DOALL_EXPLICIT" };

static cll::opt<ExecType> execType (
    cll::desc ("Executor type"),
    cll::values (
      clEnumVal (DOALL_WAKEUP, "Wake up thread pool"),
      clEnumVal (DOALL_EXPLICIT, "Explicit parallel loop"),
      clEnumValEnd),
    cll::init (DOALL_WAKEUP));


void printConfig (void) {
  std::cout << "executor: " << EXEC_NAMES[execType] << ", NUM_ROUNDS: " << NUM_ROUNDS 
    << ", VEC_SIZE: " << VEC_SIZE << std::endl;
}

double dotProdStep( const double* vecA, const double* vecB, size_t index ){
  assert( index < VEC_SIZE );

  return vecA[ index ] * vecB[ index ];

}


template <typename F, typename I>
void runDoAllWakeup (const F& functor, const I& beg, const I& end) {

  Galois::Runtime::getSystemThreadPool ().burnPower (Galois::getActiveThreads ());

  for (unsigned i = 0; i < NUM_ROUNDS; ++i) {

    Galois::Runtime::do_all_impl (
        Galois::Runtime::makeStandardRange( beg, end ),
        functor, "main_loop", STEAL);
  }

  Galois::Runtime::getSystemThreadPool ().beKind ();
  

}

template <typename F, typename I>
void runDoAllExplicit (const F& functor, const I& beg, const I& end) {

  Galois::Runtime::Barrier& barrier = Galois::Runtime::getSystemBarrier ();

  auto range = Galois::Runtime::makeStandardRange (beg, end);

  Galois::Runtime::DoAllWork<F, decltype(range)> exec (functor, range);

  volatile unsigned i = 0; 
  
  auto loop = [&] (void) {
    while (true) {
  
      if (Galois::Runtime::LL::getTID () == 0) {
        ++i;
      }

      exec.reinit (range);

      barrier ();

      if (i > NUM_ROUNDS) { break; }

      exec ();

      barrier ();

    }
  };

  Galois::Runtime::getSystemThreadPool ().run (Galois::getActiveThreads (), loop);
}


int main( int argc, char* argv[] ){
  Galois::StatManager sm;
  LonestarStart(argc, argv, "none", "none", "none");

  printConfig ();

  double* vecA = new double[ VEC_SIZE ];
  double* vecB = new double[ VEC_SIZE ];

  Galois::do_all (boost::counting_iterator<size_t> (0), boost::counting_iterator<size_t> (VEC_SIZE),
      [vecA, vecB] (const size_t index) {
        vecA[index] = acos (-1.0);
        vecB[index] = acos (-1.0);
      }, Galois::loopname ("init_loop"));

  AccumDouble result;

  auto functor = [&result, &vecA, &vecB] (size_t index) {
    result += dotProdStep (vecA, vecB, index);
  };

  boost::counting_iterator<size_t> beg( 0 );
  boost::counting_iterator<size_t> end( VEC_SIZE );

  Galois::StatTimer t;

  t.start ();
  switch (execType) {
    case DOALL_WAKEUP:
      runDoAllWakeup (functor, beg, end);
      break;

    case DOALL_EXPLICIT:
      runDoAllExplicit (functor, beg, end);
      break;

    default:
      std::abort ();
  };
  t.stop ();

  std::cout << "Dot Product Answer = " << result.reduce () << std::endl;

  delete[] vecA; vecA = nullptr;
  delete[] vecB; vecB = nullptr;

  return 0;
}
