#ifndef BLISS_HEAP_HH
#define BLISS_HEAP_HH
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
namespace bliss {
/** \internal
 * \brief A capacity bounded heap data structure.
 */
class Heap {
	unsigned int N;
	unsigned int n;
	unsigned int *array;
	//void upheap(unsigned int k);
	void upheap(unsigned int index) {
		const unsigned int v = array[index];
		array[0] = 0;
		while(array[index/2] > v) {
			array[index] = array[index/2];
			index = index/2;
		}
		array[index] = v;
	}
	//void downheap(unsigned int k);
	void downheap(unsigned int index) {
		const unsigned int v = array[index];
		const unsigned int lim = n/2;
		while(index <= lim) {
			unsigned int new_index = index + index;
			if((new_index < n) and (array[new_index] > array[new_index+1]))
				new_index++;
			if(v <= array[new_index])
				break;
			array[index] = array[new_index];
			index = new_index;
		}
		array[index] = v;
	}

public:
	/**
	 * Create a new heap.
	 * init() must be called after this.
	 */
	Heap() {array = 0; n = 0; N = 0; }
	~Heap() {
		if(array) {
			free(array);
			array = 0;
			n = 0;
			N = 0;
		}
	}
	/**
	 * Initialize the heap to have the capacity to hold \e size elements.
	 */
	//void init(const unsigned int size);
	void init(const unsigned int size) {
		if(size > N) {
			if(array) free(array);
			array = (unsigned int*)malloc((size + 1) * sizeof(unsigned int));
			N = size;
		}
	}
	/**
	 * Is the heap empty?
	 * Time complexity is O(1).
	 */
	bool is_empty() const { return (n==0); }

	/**
	 * Remove all the elements in the heap.
	 * Time complexity is O(1).
	 */
	void clear() { n = 0; }

	/**
	 * Insert the element \a e in the heap.
	 * Time complexity is O(log(N)), where N is the number of elements
	 * currently in the heap.
	 */
	//void insert(const unsigned int e);
	void insert(const unsigned int v) {
		array[++n] = v;
		upheap(n);
	}

	/**
	 * Remove and return the smallest element in the heap.
	 * Time complexity is O(log(N)), where N is the number of elements
	 * currently in the heap.
	 */
	//unsigned int remove();
	unsigned int remove() {
		const unsigned int v = array[1];
		array[1] = array[n--];
		downheap(1);
		return v;
	}

	/**
	 * Get the number of elements in the heap.
	 */
	unsigned int size() const {return n; }

};
} // namespace bliss

#endif
