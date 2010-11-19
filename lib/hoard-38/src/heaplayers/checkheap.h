#ifndef _CHECKHEAP_H_
#define _CHECKHEAP_H_


template <class SuperHeap>
class CheckHeap : public SuperHeap {
private:
	enum { RECEIVED_A_NULL_OBJECT_FROM_MALLOC = 0 };
	enum { RECEIVED_AN_UNALIGNED_OBJECT_FROM_MALLOC = 0 };
public:
	inline void * malloc (size_t sz) {
		void * addr = SuperHeap::malloc (sz);
		if (addr == NULL) {
			assert (RECEIVED_A_NULL_OBJECT_FROM_MALLOC);
			printf ("RECEIVED_A_NULL_OBJECT_FROM_MALLOC\n");
			abort();
		}
		if ((unsigned long) addr % sizeof(double) != 0) {
			assert (RECEIVED_AN_UNALIGNED_OBJECT_FROM_MALLOC);
			printf ("RECEIVED_AN_UNALIGNED_OBJECT_FROM_MALLOC\n");
			abort();
		}
		return addr;
	}
};



#endif
