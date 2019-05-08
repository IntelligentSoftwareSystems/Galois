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
		hash_value = 0;
		cg_id = 0;
		size = subgraph_size / sizeof(ElementType);
		elements = new ElementType[size];
	}
	QuickPattern(const Embedding & emb) {
		cg_id = 0;
		size = emb.size();
		unsigned bytes = size * sizeof(ElementType);
		elements = new ElementType[size];
		std::memcpy(elements, emb.data(), bytes);
		std::unordered_map<VertexId, VertexId> map;
		VertexId new_id = 1;
		for(unsigned i = 0; i < size; i++) {
			auto& element = elements[i];
			VertexId old_id = element.vertex_id;
			auto iterator = map.find(old_id);
			if(iterator == map.end()) {
				element.set_vertex_id(new_id);
				map[old_id] = new_id++;
			} else element.set_vertex_id(iterator->second);
		}
		set_hash();
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
	inline unsigned get_hash() const { return hash_value; }
	inline void set_hash() {
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
		hash_value = h.get_value();
		//return h.get_value();
	}
	ElementType& at(unsigned index) const { return elements[index]; }
	inline unsigned get_size() const { return size; }
	inline ElementType* get_elements() { return elements; }
	inline void clean() { delete[] elements; }
	inline unsigned get_id() const { return hash_value; }
	inline unsigned get_cgid() const { return cg_id; }
	void set_cgid(unsigned i) { cg_id = i; }

private:
	unsigned size;
	ElementType* elements;
	unsigned hash_value; // quick pattern ID
	unsigned cg_id; // ID of the canonical pattern that this quick pattern belongs to
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
