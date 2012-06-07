// Galois Managed Conflict type wrapper -*- C++ -*-
/*
Galois, a framework to exploit amorphous data-parallelism in irregular
programs.

Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS SOFTWARE
AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY
PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY
WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF TRADE.
NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO THE USE OF THE
SOFTWARE OR DOCUMENTATION. Under no circumstances shall University be liable
for incidental, special, indirect, direct or consequential damages or loss of
profits, interruption of business, or related expenses which may arise from use
of Software or Documentation, including but not limited to those resulting from
defects in Software and/or Documentation, or loss or inaccuracy of data of any
kind.

@author rupesh nasre. <rupesh0508@gmail.com>
*/

#ifndef GALOIS_UTIL_SPARSEBITVECTOR_H
#define GALOIS_UTIL_SPARSEBITVECTOR_H

#include "Galois/Runtime/SimpleLock.h"

#include <vector>
#include <iostream>
#include <string>

#define ForEach(ii)	for (std::vector<SparseBitVectorElement *>::iterator ii = bits.begin(); ii != bits.end(); ++ii)
#define ForEachLock(ll)	for (std::vector<LockType>::iterator ll = kulup.begin(); ll != kulup.end(); ++ll)
#define null	0

namespace Galois {

//! Concurrent version of sparse bit vector.
/*! Stores objects as indices in sparse bit vectors.
    Saves space when the data to be stored is sparsely populated. */

typedef unsigned long WORD;
typedef GaloisRuntime::SimpleLock<int, true> LockType;

static const unsigned wordsize = sizeof(WORD)*8;

class SparseBitVectorElement {
public:

	SparseBitVectorElement() {
		bits = 0;
	}
	~SparseBitVectorElement() { }

	// not using the conditions on bit number to improve efficiency.
	bool set(unsigned bit) {	// returns true if the bit is newly set.
		// bit: 0 .. wordsize - 1.
		WORD oribits = bits;
		// if (bit >= wordsize) {
			bits |= ((WORD)1 << bit);
			//std::cout << "debug: " << bit << std::endl;
		// }
		return bits != oribits;
	}
	bool reset(unsigned bit) {	// returns true if the bit is newly reset.
		// bit: 0 .. wordsize - 1.
		WORD oribits = bits;
		// if (bit < wordsize) {
			bits &= ~(1 << bit);
		// }
		return bits != oribits;
	}
	bool test(unsigned bit) {
		// bit: 0 .. wordsize - 1.
		// if (bit >= wordsize) {
			return bits & (1 << bit);
		// }
	}
	void clear() {
		bits = 0;
	}
	bool isEmpty() {
		return bits == 0;
	}
	unsigned count() {
		unsigned numElements = 0;
		WORD powerof2 = 1;

		for (unsigned ii = 0; ii < wordsize; ++ii) {
			if (bits & powerof2) {
				++numElements;
			}
			powerof2 <<= 1;
		}
		return numElements;
	}
	unsigned size() {
		return count();
	}
	bool unify(SparseBitVectorElement &second) {
		return unify(second.bits);
	}
	bool unify(WORD second) {
		WORD oribits = bits;
		bits |= second;
		return bits != oribits;
	}
	bool equals(SparseBitVectorElement &second) {
		return equals(second.bits);
	}
	bool equals(WORD second) {
		return bits == second;
	}
	bool isSubsetEq(SparseBitVectorElement &second) {
		return isSubsetEq(second.bits);
	}
	bool isSubsetEq(WORD second) {
		return (bits & second) == bits;
	}
	void print(unsigned ventry, std::string prefix = std::string("")) {
		WORD powerof2 = 1;
		unsigned bitno = 0;

		for (unsigned ii = 0; ii < wordsize; ++ii) {
			if (bits & powerof2) {
				std::cout << prefix << ventry*wordsize + bitno << ", ";
			}
			powerof2 <<= 1;
			++bitno;
		}
	}
	void getAllSetBits(unsigned ventry, std::vector<unsigned> &setbits) {
		WORD powerof2 = 1;
		unsigned bitno = 0;

		for (unsigned ii = 0; ii < wordsize; ++ii) {
			if (bits & powerof2) {
				setbits.push_back(ventry*wordsize + bitno);
			}
			powerof2 <<= 1;
			++bitno;
		}
			
	}
private:
	WORD bits;
};

class SparseBitVector {

public:

	SparseBitVector(unsigned maxSize) { 
		init(maxSize);
	}
	SparseBitVector() { 
		bits.clear();
		kulup.clear();
	}
	~SparseBitVector() { 
		clear();
	}

	void init(unsigned maxSize) {
		bits.resize((maxSize + wordsize - 1) / wordsize);
		kulup.resize(bits.size());
		init();
	}
	
	bool set(unsigned bit) {
		unsigned ventry, wbit;
		getOffsets(bit, ventry, wbit);
		//std::cout << "debug: ventry=" << ventry << " wbit=" << wbit << std::endl;
		//std::cout << "debug: bits.size=" << bits.size() << std::endl;
		return set(ventry, wbit);
	}
	bool reset(unsigned bit) {
		unsigned ventry, wbit;
		getOffsets(bit, ventry, wbit);
		return reset(ventry, wbit);
	}
	bool test(unsigned bit) {
		unsigned ventry, wbit;
		getOffsets(bit, ventry, wbit);
		return test(ventry, wbit);
	}
	bool isEmpty() {
		ForEach(ii) {
			if (*ii && !(*ii)->isEmpty()) {
				return false;
			}
		}
		return true;
	}
	unsigned count() {	// costly, O(n).
		unsigned numElements = 0;
		ForEach(ii) {
			if (*ii) {
				numElements += (*ii)->count();
			}
		}
		return numElements;
	}
	unsigned size() {
		return count();
	}
	bool unify(SparseBitVector &second) {
		// code assumes same sizes of the two vectors.
		unsigned ventry = 0;
		bool changed = false;

		for (std::vector<SparseBitVectorElement *>::iterator ii = bits.begin(), jj = second.bits.begin(); ii != bits.end(); ++ii, ++jj) {
			if (*jj) {
				lock(ventry);
				if (!*ii) {
					*ii = getNew();
				}
				changed |= (*ii)->unify(**jj);
				unlock(ventry);
			}
			++ventry;
		}
		return changed;
	}
	bool equals(SparseBitVector &second) {
		if (this == &second) {	// both are the same vectors.
			return true;
		}
		if (bits.size() != second.bits.size()) {	// sizes are different.
			return false;
		}
		/*if (count() != second.count()) {	// no of set bits are different.
			return false;
		}*/
		for (std::vector<SparseBitVectorElement *>::iterator ii = bits.begin(), jj = second.bits.begin(); ii != bits.end(); ++ii, ++jj) {
			if (*ii && *jj && !(*ii)->equals(**jj)) {
				return false;
			} else if (*ii && !*jj || !*ii && *jj) {
				return false;
			}
		}
		return true;
	}
	bool isSubsetEq(SparseBitVector &second) {	// ptsto(this) <= ptsto(second)?
		if (this == &second) {	// both are the same vectors.
			return true;
		}
		if (bits.size() > second.bits.size()) {
			return false;
		}
		// bits size is smaller or equal to second.bits size.

		for (std::vector<SparseBitVectorElement *>::iterator ii = bits.begin(), jj = second.bits.begin(); ii != bits.end(); ++ii, ++jj) {
			if (*ii && *jj && !(*ii)->isSubsetEq(**jj)) {
				return false;
			} else if (*ii && !*jj) {
				return false;
			}
		}
		return true;
	}
	void getAllSetBits(std::vector<unsigned> &setbits) {
		unsigned ventry = 0;

		ForEach(ii) {
			if (*ii) {
				(*ii)->getAllSetBits(ventry, setbits);
			}
			++ventry;
		}
	}
	void print(std::string prefix = std::string("")) {
		unsigned ventry = 0;
		//std::cout << "Elements: ";
		ForEach(ii) {
			if (*ii) {
				(*ii)->print(ventry, prefix);
			}
			++ventry;
		}
		std::cout << std::endl;
	}

protected:
	void init() {	// no locking since this function is not public.
		ForEach(ii) {
			*ii = null;
		}
	}
	void clear() {	// no locking since this function is not public.
		ForEach(ii) {
			if (*ii) {
				delete *ii;
				*ii = null;
			}
		}
	}
	void getOffsets(unsigned bit, unsigned &ventry, unsigned &wbit) {
		ventry = bit / wordsize;
		wbit = bit % wordsize;
	}
	SparseBitVectorElement *getNew() {
		return new SparseBitVectorElement();
	}
	bool set(unsigned ventry, unsigned wbit) {
		//if (ventry < bits.size()) {
			lock(ventry);
			if (bits[ventry] == null) {
				bits[ventry] = getNew();
			}
			bool retval = bits[ventry]->set(wbit);
			unlock(ventry);
			return retval;
		//} else {
		//	error("SparseBitVector::set: Array out of bound.");
		//}
	}
	bool reset(unsigned ventry, unsigned wbit) {
		//if (ventry < bits.size()) {
			bool retval = false;
			lock(ventry);
			if (bits[ventry] != null) {
				retval = bits[ventry]->reset(wbit);
			}
			unlock(ventry);
			return retval;
		//} else {
		//	error("SparseBitVector::reset: Array out of bound.");
		//}
	}
	bool test(unsigned ventry, unsigned wbit) {
		//if (ventry < bits.size()) {
			return bits[ventry]->test(wbit);
		//} else {
		//	error("SparseBitVector::test: Array out of bound.");
		//}
	}
	void error(std::string &msg) {
		std::cerr << "ERROR: " << msg << std::endl;
		// exit(1);
	}

private:
	void lock(unsigned ventry) {
		// if (ventry < kulup.size()) {
			kulup[ventry].lock();
		// }
	}
	void unlock(unsigned ventry) {
		// if (ventry < kulup.size()) {
			kulup[ventry].unlock();
		// }
	}
	void lockAll() {
		ForEachLock(ll) {
			ll->lock();
		}
	}
	void unlockAll() {
		ForEachLock(ll) {
			ll->unlock();
		}
	}
	std::vector<SparseBitVectorElement *> bits;
	std::vector<LockType> kulup;	// Can't have it in SparseBitVectorElement since some entries of bits could be null.
};
}



#endif //  _GALOIS_SPARSEBITVECTOR_H

/*
int main() {
	Galois::SparseBitVector vv(100), vv2(100);
	vv.set(10);
	vv.set(20);
	vv.set(22);
	vv.set(11);
	vv.set(30);
	vv.reset(11);
	vv.reset(12);
	vv.reset(20);
	vv.set(98);
	vv.print();

	vv2.set(10);
	vv2.set(30);
	vv2.set(22);
	vv2.set(98);

	vv.unify(vv2);
	vv.print();

	return 0;
}
*/
