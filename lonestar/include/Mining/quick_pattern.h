/*
 * quick_pattern.hpp
 *
 *  Created on: Aug 4, 2017
 *      Author: icuzzq
 */

#ifndef CORE_QUICK_PATTERN_HPP_
#define CORE_QUICK_PATTERN_HPP_

#include "type.h"

class Quick_Pattern {
friend std::ostream & operator<<(std::ostream & strm, const Quick_Pattern& quick_pattern);
public:
	Quick_Pattern() { }
	Quick_Pattern(unsigned int size_of_tuple) {
		size = size_of_tuple / sizeof(Element_In_Tuple);
		elements = new Element_In_Tuple[size];
	}
	~Quick_Pattern() {}
	//operator for map
	bool operator==(const Quick_Pattern& other) const {
		//compare edges
		assert(size == other.size);
		for (unsigned int i = 0; i < size; ++i) {
			const Element_In_Tuple & t1 = elements[i];
			const Element_In_Tuple & t2 = other.elements[i];
			int cmp_element = t1.cmp(t2);
			if(cmp_element != 0) {
				return false;
			}
		}
		return true;
	}
	operator size_t() const {
		size_t a = 0;
		for (unsigned int i = 0; i < size; ++i) {
			auto element = elements[i];
			a += element.vertex_id;
		}
		return a; 
	}
	//operator int() const { return 0; }
	unsigned int get_hash() const {
		//TODO
		bliss::UintSeqHash h;
		h.update(size);
		//hash vertex labels and edges
		for (unsigned int i = 0; i < size; ++i) {
			auto element = elements[i];
			h.update(element.vertex_id);
			h.update(element.vertex_label);
			h.update(element.history_info);
		}
		return h.get_value();
	}
	Element_In_Tuple& at(unsigned int index) const {
		return elements[index];
	}
	inline unsigned int get_size() const { return size; }
	inline Element_In_Tuple* get_elements() { return elements; }
	inline void clean() { delete[] elements; }

private:
	unsigned int size;
	Element_In_Tuple* elements;
};

std::ostream & operator<<(std::ostream & strm, const Quick_Pattern& quick_pattern) {
	if(quick_pattern.get_size() == 0) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned int index = 0; index < quick_pattern.get_size() - 1; ++index) {
		strm << quick_pattern.elements[index] << ", ";
	}
	strm << quick_pattern.elements[quick_pattern.get_size() - 1];
	strm << ")";
	return strm;
}

namespace std {
	template<>
	struct hash<Quick_Pattern> {
		std::size_t operator()(const Quick_Pattern& qp) const {
			return std::hash<int>()(qp.get_hash());
		}
	};
}
#endif /* CORE_QUICK_PATTERN_HPP_ */
