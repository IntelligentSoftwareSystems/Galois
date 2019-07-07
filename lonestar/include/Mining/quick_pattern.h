#ifndef QUICK_PATTERN_HPP_
#define QUICK_PATTERN_HPP_

#include "embedding.h"
#include "equivalence.h"

template <typename EmbeddingTy, typename ElementTy> class QuickPattern;
template <typename EmbeddingTy, typename ElementTy> std::ostream& operator<<(std::ostream& strm, const QuickPattern<EmbeddingTy, ElementTy>& qp);

template <typename EmbeddingTy, typename ElementTy>
class QuickPattern {
friend std::ostream & operator<< <>(std::ostream & strm, const QuickPattern<EmbeddingTy, ElementTy>& qp);
public:
	QuickPattern() { }
	QuickPattern(unsigned subgraph_size) {
		hash_value = 0;
		cg_id = 0;
		size = subgraph_size / sizeof(ElementTy);
		elements = new ElementTy[size];
	}
	QuickPattern(const EmbeddingTy & emb) {
		cg_id = 0;
		size = emb.size();
		unsigned bytes = size * sizeof(ElementTy);
		elements = new ElementTy[size];
		std::memcpy(elements, emb.data(), bytes);
		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;
		for(unsigned i = 0; i < size; i++) {
			auto& element = elements[i];
			VertexId old_id = element.get_vid();
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;
			} else element.set_vertex_id(iterator->second);
		}
		set_hash();
	}
	QuickPattern(unsigned n, std::vector<bool> connected) {
		cg_id = 0;
		size = std::count(connected.begin(), connected.end(), true) + 1; // number of edges + 1
		elements = new ElementTy[size];
		//std::cout << "num_vertices: " << n << ", num_edges: " << size-1 << "\n";
		std::vector<unsigned> pos(n, 0);
		pos[1] = 1; pos[2] = 2;
		elements[0].set_vertex_id(1);
		elements[0].set_history_info(0);
		elements[1].set_vertex_id(2);
		elements[1].set_history_info(0);
		//std::cout << "Constructing edge 1: 1 --> 2 \n";
		int count = 2;
		int l = 1;
		for (unsigned i = 2; i < n; i++) {
			if(i<n-2) pos[i+1] = pos[i];
			for (unsigned j = 0; j < i; j++) {
				if (connected[l++]) {
					if (i<n-2) pos[i+1] ++;
					//std::cout << "Constructing edge " << i <<": " << elements[pos[j]].get_vid() << " --> " << i+1 << "\n";
					elements[count].set_vertex_id(i+1);
					elements[count++].set_history_info(pos[j]);
				}
			}
		}
		set_hash();
	}
	~QuickPattern() {}
	void get_equivalences(VertexPositionEquivalences& equ) {
		equ.set_size(size);
		for (unsigned i = 0; i < size; ++i) equ.add_equivalence(i, i);
		findAutomorphisms(equ);
		//equ.propagateEquivalences();
	}
	//operator for map
	bool operator==(const QuickPattern& other) const {
		//compare edges
		assert(size == other.size);
		for (unsigned i = 0; i < size; ++i) {
			const ElementTy & t1 = elements[i];
			const ElementTy & t2 = other.elements[i];
			int cmp_element = t1.cmp(t2);
			if(cmp_element != 0) {
				return false;
			}
		}
		return true;
	}
	operator size_t() const {
		size_t a = 0;
		for (unsigned i = 0; i < size; ++i) {
			auto element = elements[i];
			a += element.get_vid();
		}
		return a; 
	}
	inline unsigned get_hash() const { return hash_value; }
	inline void set_hash() {
		bliss::UintSeqHash h;
		h.update(size);
		//hash vertex labels and edges
		for (unsigned i = 0; i < size; ++i) {
			auto element = elements[i];
			h.update(element.get_vid());
#ifdef ENABLE_LABEL
			h.update(element.get_vlabel());
#endif
#ifndef USE_SIMPLE 
			h.update(element.get_his());
#endif
		}
		hash_value = h.get_value();
		//return h.get_value();
	}
	ElementTy& at(unsigned index) const { return elements[index]; }
	inline unsigned get_size() const { return size; }
	inline ElementTy* get_elements() { return elements; }
	inline void clean() { delete[] elements; }
	inline unsigned get_id() const { return hash_value; }
	inline unsigned get_cgid() const { return cg_id; }
	void set_cgid(unsigned i) { cg_id = i; }

private:
	unsigned size;
	ElementTy* elements;
	unsigned hash_value; // quick pattern ID
	unsigned cg_id; // ID of the canonical pattern that this quick pattern belongs to

	void findAutomorphisms(VertexPositionEquivalences &eq_sets) {
		if (size == 2) { // single-edge
			if (at(0).get_vlabel() == at(1).get_vlabel()) {
				eq_sets.add_equivalence(0, 1);
				eq_sets.add_equivalence(1, 0);
			}
		} else if (size == 3) { // two-edge chain
			if (at(2).get_his() == 0) {
				if (at(1).get_vlabel() == at(2).get_vlabel()) {
					eq_sets.add_equivalence(1, 2);
					eq_sets.add_equivalence(2, 1);
				}
			} else if (at(2).get_his() == 1) {
				if (at(0).get_vlabel() == at(2).get_vlabel()) {
					eq_sets.add_equivalence(0, 2);
					eq_sets.add_equivalence(2, 0);
				}
			} else std::cout << "Error\n";
		} else if (size == 4) { // three-edge chain or star
			if (at(2).get_his() == 0) {
				if (at(3).get_his() == 0) {
					if (at(1).get_vlabel() == at(2).get_vlabel()) {
						eq_sets.add_equivalence(1, 2);
						eq_sets.add_equivalence(2, 1);
					}
					if (at(1).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(1, 3);
						eq_sets.add_equivalence(3, 1);
					}
					if (at(2).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(2, 3);
						eq_sets.add_equivalence(3, 2);
					}
				} else if (at(3).get_his() == 1) {
					if (at(2).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(2, 3);
						eq_sets.add_equivalence(3, 2);
					}
					if (at(0).get_vlabel() == at(1).get_vlabel()) {
						eq_sets.add_equivalence(0, 1);
						eq_sets.add_equivalence(1, 0);
					}
				} else if (at(3).get_his() == 2) {
					if (at(1).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(1, 3);
						eq_sets.add_equivalence(3, 1);
					}
					if (at(0).get_vlabel() == at(2).get_vlabel()) {
						eq_sets.add_equivalence(0, 2);
						eq_sets.add_equivalence(2, 0);
					}
				} else std::cout << "Error\n";
			} else if (at(2).get_his() == 1) {
				if (at(3).get_his() == 0) {
					if (at(2).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(2, 3);
						eq_sets.add_equivalence(3, 2);
					}
					if (at(0).get_vlabel() == at(1).get_vlabel()) {
						eq_sets.add_equivalence(0, 1);
						eq_sets.add_equivalence(1, 0);
					}
				} else if (at(3).get_his() == 1) {
					if (at(0).get_vlabel() == at(2).get_vlabel()) {
						eq_sets.add_equivalence(0, 2);
						eq_sets.add_equivalence(2, 0);
					}
					if (at(0).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(0, 3);
						eq_sets.add_equivalence(3, 0);
					}
					if (at(2).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(2, 3);
						eq_sets.add_equivalence(3, 2);
					}
				} else if (at(3).get_his() == 2) {
					if (at(0).get_vlabel() == at(3).get_vlabel()) {
						eq_sets.add_equivalence(0, 3);
						eq_sets.add_equivalence(3, 0);
					}
					if (at(1).get_vlabel() == at(2).get_vlabel()) {
						eq_sets.add_equivalence(1, 2);
						eq_sets.add_equivalence(2, 1);
					}
				} else std::cout << "Error\n";
			} else std::cout << "Error\n";
		} else { // four-edge and beyond
			std::cout << "Currently not supported\n";
		}
	}
};

template <typename EmbeddingTy, typename ElementTy>
std::ostream & operator<<(std::ostream & strm, const QuickPattern<EmbeddingTy, ElementTy>& qp) {
	if(qp.get_size() == 0) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned index = 0; index < qp.get_size() - 1; ++index)
		strm << qp.elements[index] << ", ";
	strm << qp.elements[qp.get_size() - 1];
	strm << ")";
	return strm;
}

namespace std {
template<typename EmbeddingTy, typename ElementTy>
struct hash<QuickPattern<EmbeddingTy, ElementTy> > {
	std::size_t operator()(const QuickPattern<EmbeddingTy, ElementTy>& qp) const {
		return std::hash<int>()(qp.get_hash());
	}
};
}
#endif // QUICK_PATTERN_HPP_
