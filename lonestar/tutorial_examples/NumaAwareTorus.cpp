// This example shows
// 1. how to bulid a conflict-aware data structure w/ Locakable
// 2. how to implement conflict detection in your data structure's APIs
// 3. how to define iterators for STL compliance
// 4. how to leverage LargeArray to do NUMA-aware allocation
// 5. how to turn-off conflict-detection when you do not want it
#include "galois/Galois.h"
#include "galois/LargeArray.h"

#include <boost/iterator/counting_iterator.hpp>

#include <iostream>

template<typename T>
class Torus2D {
  //! [Internal type with Lockable]
  //************************************************************************
  // internal type to combine user data with Lockable object
  //************************************************************************
  struct NodeData: 
      public galois::runtime::Lockable
  {
  public:
    using reference = T&;

  public:
    T v;

  public:
    reference getData() { return v; }
  };
  //! [Internal type with Lockable]

  //! [Array of internal type]
  size_t numRows, numCols;

  // use galois::LargeArray for NUMA-aware allocation
  // will allocate numRows*numCols elements in constructors
  galois::LargeArray<NodeData> data;
  //! [Array of internal type]

  //! [Types for STL]
  //************************************************************************
  // subtypes visible to user
  //************************************************************************
public:
  // opaque type for node
  using TorusNode = size_t;

  // iterator for an STL container
  using iterator = boost::counting_iterator<TorusNode>;
  //! [Types for STL]

public:
  //************************************************************************
  // constructor for the torus
  //************************************************************************
  Torus2D(size_t r, size_t c)
      : numRows(r), numCols(c)
  {
    // allocate torus nodes in an interleaved way among NUMA domains
    data.allocateInterleaved(r*c);

    // call constructor for each torus node
    for (size_t n = 0; n < r*c; ++n) {
      data.constructAt(n);
    }
  }

  //! [APIs for sizes]
  //************************************************************************
  // functions for size of the torus
  //************************************************************************
  size_t height() { return numRows; }
  size_t width() { return numCols; }
  size_t size() { return width()*height(); }
  //! [APIs for sizes]

  //! [Iterators]
  //************************************************************************
  // functions to traverse nodes
  //************************************************************************
  iterator begin() { return iterator(0); }
  iterator end() { return iterator(size()); }
  //! [Iterators]

  //! [Acquire node ownership]
  //************************************************************************
  // functions to acquire node ownership
  //************************************************************************
  void acquireNode(TorusNode n, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    // sanity check
    assert(n < size());

    // use this call to detect conflicts and handling aborts
    galois::runtime::acquire(&data[n], mflag);
  }
  //! [Acquire node ownership]

  //! [Get data]
  //************************************************************************
  // function to access node data
  //************************************************************************
  typename NodeData::reference
  getData(TorusNode n, galois::MethodFlag mflag = galois::MethodFlag::WRITE)
  {
    acquireNode(n, mflag);

    // use the internal wrapper type to encapsulate users from Lockable objects
    return data[n].getData();
  }
  //! [Get data]

  //! [Easy operator cautiousness]
  //************************************************************************
  // functions to access neighboring nodes, i.e. edges in a general graph
  //************************************************************************
  iterator upNeighbor(TorusNode n) {
    auto r = n / numCols, c = n % numCols;
    auto newR = (r + numRows - 1) % numRows;
    return iterator(newR * numCols + c);
  }

  iterator downNeighbor(TorusNode n) {
    auto r = n / numCols, c = n % numCols;
    auto newR = (r + 1) % numRows;
    return iterator(newR * numCols + c);
  }

  iterator leftNeighbor(TorusNode n) {
    auto r = n / numCols, c = n % numCols;
    auto newC = (c + numCols - 1) % numCols;
    return iterator(r * numCols + newC);  
  }

  iterator rightNeighbor(TorusNode n) {
    auto r = n / numCols, c = n % numCols;
    auto newC = (c + 1) % numCols;
    return iterator(r * numCols + newC);  
  }

  //************************************************************************
  // function to lock all neighbors of node n
  // similar to edge_begin(), edge_end() or edges() in a general graph
  //************************************************************************
  void acquireAllNeighbors(TorusNode n, galois::MethodFlag mflag = galois::MethodFlag::WRITE) {
    acquireNode(*upNeighbor(n), mflag);
    acquireNode(*downNeighbor(n), mflag);
    acquireNode(*leftNeighbor(n), mflag);
    acquireNode(*rightNeighbor(n), mflag);
  }
  //! [Easy operator cautiousness]
}; // end of class Torus2D

int main(int argc, char *argv[]) {
  galois::SharedMemSys G;

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <num_rows> <num_columns> <num_threads>" << std::endl;
    return 1;
  }

  galois::setActiveThreads(std::atoi(argv[3]));

  //! [Use torus]
  using Torus = Torus2D<unsigned int>;
  using TorusNode = Torus::TorusNode;

  Torus torus(std::atoi(argv[1]), std::atoi(argv[2]));

  galois::do_all(
      galois::iterate(0ul, torus.size()),                 // range as a pair of unsigned integers
      [&] (TorusNode n) { torus.getData(n) = 0; }         // operator
      , galois::loopname("do_all_torus_reset_self")       // options
  );

  galois::for_each(
      galois::iterate(torus),                             // range as a container. assuming begin() and end()
      [&] (TorusNode n, auto& ctx) {                      // operator
        // cautious point
        torus.acquireAllNeighbors(n);

        torus.getData(*torus.upNeighbor(n)) += 1;
        torus.getData(*torus.downNeighbor(n)) += 1;
        torus.getData(*torus.leftNeighbor(n)) += 1;
        torus.getData(*torus.rightNeighbor(n)) += 1;
      }
      , galois::loopname("for_each_torus_add_neighbors")  // options
      , galois::no_pushes()
  );
  //! [Use torus]

  //! [Turn off conflict detection]
  // serial verification, no conflict is possible
  size_t numWrongAnswer = 0;
  for (auto n: torus) {
    // use galois::MethodFlag::UNPROTECTED to notify Galois runtime
    // that do not acquire lock for this call
    if (torus.getData(n, galois::MethodFlag::UNPROTECTED) != 4) {
      numWrongAnswer++;
    }
  }
  std::cout << "# nodes of wrong answer: " << numWrongAnswer << std::endl;
  //! [Turn off conflict detection]

  return 0;
}
