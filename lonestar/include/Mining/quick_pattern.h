/*
 * quick_pattern.hpp
 *
 *  Created on: Aug 4, 2017
 *      Author: icuzzq
 */

#ifndef CORE_QUICK_PATTERN_HPP_
#define CORE_QUICK_PATTERN_HPP_

#include "type.h"

class QuickPattern {
friend std::ostream & operator<<(std::ostream & strm, const QuickPattern& quick_pattern);
public:
	QuickPattern() { }
	QuickPattern(unsigned subgraph_size) {
		id = -1;
		size = subgraph_size/ sizeof(ElementType);
		elements = new ElementType[size];
	}
	~QuickPattern() {}
	//operator for map
	bool operator==(const QuickPattern& other) const {
		//compare edges
		assert(size == other.size);
		for (unsigned i = 0; i < size; ++i) {
			const ElementType & t1 = elements[i];
			const ElementType & t2 = other.elements[i];
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
			a += element.vertex_id;
		}
		return a; 
	}
	unsigned get_hash() const {
		//TODO
		bliss::UintSeqHash h;
		h.update(size);
		//hash vertex labels and edges
		for (unsigned i = 0; i < size; ++i) {
			auto element = elements[i];
			h.update(element.vertex_id);
#ifdef ENABLE_LABEL
			h.update(element.vertex_label);
#endif
			h.update(element.history_info);
		}
		return h.get_value();
	}
	ElementType& at(unsigned index) const {
		return elements[index];
	}
	inline unsigned get_size() const { return size; }
	inline ElementType* get_elements() { return elements; }
	inline void clean() { delete[] elements; }
	inline int get_id() const { return id; }
	void set_id(int i) { id = i; }

private:
	unsigned size;
	ElementType* elements;
	int id; // quick pattern ID
};

std::ostream & operator<<(std::ostream & strm, const QuickPattern& quick_pattern) {
	if(quick_pattern.get_size() == 0) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned index = 0; index < quick_pattern.get_size() - 1; ++index)
		strm << quick_pattern.elements[index] << ", ";
	strm << quick_pattern.elements[quick_pattern.get_size() - 1];
	strm << ")";
	return strm;
}

namespace std {
	template<>
	struct hash<QuickPattern> {
		std::size_t operator()(const QuickPattern& qp) const {
			return std::hash<int>()(qp.get_hash());
		}
	};
}
#endif /* CORE_QUICK_PATTERN_HPP_ */
