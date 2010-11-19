#ifndef _TRYHEAP_H_
#define _TRYHEAP_H_

template <class Heap1, class Heap2>
class TryHeap : public Heap2 {
public:
  TryHeap (void)
  {}

  inline void * malloc (size_t sz) {
    void * ptr = heap1.malloc (sz);
    if (ptr == NULL)
      ptr = Heap2::malloc (sz);
    return ptr;
  }

  inline void free (void * ptr) {
    heap1.free (ptr);
  }

private:
	Heap1 heap1;
};


#endif
