#ifndef EQUIVALENCE_HPP_
#define EQUIVALENCE_HPP_
/**
 * Code from on below link. Modified under Galois.
 *
 * https://github.com/qcri/Arabesque
 *
 * Copyright (c) 2015 Qatar Computing Research Institute
 * All rights reserved.
 * Reused/revised under 3-BSD
 */
#include "types.h"

class VertexPositionEquivalences {
friend std::ostream & operator<<(std::ostream & strm, const VertexPositionEquivalences& equ);
public:
	VertexPositionEquivalences() {
		numVertices = 0;
	}
	~VertexPositionEquivalences() {}
	void clear() {
		for (unsigned i = 0; i < equivalences.size(); ++i)
			equivalences[i].clear();
	}
	void set_size(unsigned n) {
		if (numVertices != n) {
			equivalences.resize(n);
			numVertices = n;
		}
	}
	void add_equivalence(unsigned pos1, unsigned pos2) {
		equivalences[pos1].insert(pos2);
	}
	UintSet get_equivalent_set(unsigned pos) const {
		return equivalences[pos];
	}
	void propagate_equivalences() {
		for (unsigned i = 0; i < numVertices; ++i) {
			UintSet currentEquivalences = equivalences[i];
			for (auto equivalentPosition : currentEquivalences) {
				if (equivalentPosition == i) continue;
				//equivalences[equivalentPosition];
			}
		}
	}
	unsigned get_size() const { return numVertices; }
	bool empty() const { return numVertices == 0; }

private:
	UintSets equivalences;
	unsigned numVertices;
};
#endif // EQUIVALENCE_HPP_
