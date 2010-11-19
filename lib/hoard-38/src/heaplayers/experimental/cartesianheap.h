#ifndef _CARTESIANHEAP_H_
#define _CARTESIANHEAP_H_

//#include "heaplayers.h"
//#include "treap.h"

// NB: All objects are rounded up to at least the size of a treap node
// (around 24 bytes on a 32-bit architecture).

template <class SuperHeap>
class CartesianHeap : public SuperHeap {

  // Provide a freelist wrapper for treap nodes.
  class MyTreap : public Treap<size_t, void *> {
  public:
    // class Node : public PerClassHeap<Treap<size_t, void *>::Node, FreelistHeap<SuperHeap> > {};
  };

public:

  ~CartesianHeap (void) {
    // FIX ME
    // free everything from the treap.
  }

  void * malloc (size_t sz) {
    // Round sz up.
    sz = (sz < sizeof(MyTreap::Node)) ? sizeof(MyTreap::Node) : sz;
    MyTreap::Node * n = (MyTreap::Node *) treap.lookupGreater (sz);
    if (n != NULL) {
      assert (n->getValue() == (void *) n);
      void * ptr = n->getValue();
      treap.remove (n);
      // delete n; // onto a freelist
      return ptr;
    } else {
      return SuperHeap::malloc (sz);
    }
  }

  void free (void * ptr) {
    // MyTreap::Node * n = new MyTreap::Node; // from a freelist
    // cout << "n = " << (void *) n << endl;
    MyTreap::Node * n = (MyTreap::Node *) ptr;
    treap.insert (n, getSize(ptr), ptr, (unsigned int) ptr);
  }

  // Removes a pointer from the treap.
  void remove (void * ptr) {
    treap.remove ((MyTreap::Node *) ptr);
  }
    
private:

  MyTreap treap;

};

#endif // _CARTESIANHEAP_H_
