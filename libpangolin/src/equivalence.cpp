#include "equivalence.h"

std::ostream & operator<<(std::ostream & strm, const VertexPositionEquivalences& equ) {
	if(equ.get_size() == 0) {
		strm << "(empty)";
		return strm;
	}
	strm << "VertexPositionEquivalences{equivalences=[";
	for (unsigned i = 0; i < equ.get_size(); ++i) {
		strm << "[";
		for (auto ele : equ.get_equivalent_set(i)) {
			strm << ele << ", ";
		}
		strm << "], ";
	}
	strm << "]; size=" << equ.get_size() << "}\n";
	return strm;
}

