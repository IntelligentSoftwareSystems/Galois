#ifndef EQUIVALENCE_HPP_
#define EQUIVALENCE_HPP_

typedef std::set<unsigned> UintSet;
typedef std::vector<UintSet> UintSets;

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
#endif // EQUIVALENCE_HPP_
