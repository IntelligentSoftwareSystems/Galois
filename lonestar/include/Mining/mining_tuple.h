
#ifndef STRUCT_MINING_TUPLE_HPP_
#define STRUCT_MINING_TUPLE_HPP_
#include "type.h"

class MTuple {
	friend std::ostream & operator<<(std::ostream & strm, const MTuple& cg);
public:
	MTuple(unsigned size_of_t) {
		size = size_of_t / sizeof(Element_In_Tuple);
		elements = nullptr;
	}
	virtual ~MTuple() {};
	void init(std::vector<Element_In_Tuple> & tuple) { elements = tuple.data(); }
	virtual Element_In_Tuple& at(unsigned index) { return elements[index]; }
	inline unsigned get_size() const { return size; }
	inline Element_In_Tuple* get_elements() { return elements; }
	inline virtual unsigned get_num_vertices() { return elements[size - 1].key_index; }
protected:
	unsigned size;
	Element_In_Tuple* elements;
};

std::ostream & operator<<(std::ostream & strm, const MTuple& tuple) {
	if(tuple.get_size() == 0){
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	for(unsigned index = 0; index < tuple.get_size() - 1; ++index) {
		strm << tuple.elements[index] << ", ";
	}
	strm << tuple.elements[tuple.get_size() - 1];
	strm << ")";
	return strm;
}

class MTuple_join : public MTuple {
public:
	MTuple_join(unsigned size_of_t): MTuple(size_of_t) {
		capacity = size + 1;
		added_element = nullptr;
	}
	virtual ~MTuple_join() {};
	void init(std::vector<Element_In_Tuple> & tuple, std::unordered_set<VertexId>& vertices_set) {
		elements = tuple.data();
		vertices_set.reserve(size);
		for(unsigned index = 0; index < size; index ++) {
			vertices_set.insert(elements[index].vertex_id);
		}
	}
	Element_In_Tuple& at(unsigned index) {
		if(index == capacity - 1) { return *added_element; }
		return elements[index];
	}
	void push(Element_In_Tuple* element) {
		added_element = element;
		size++;
	}
	void pop() {
		added_element = nullptr;
		size--;
	}
	inline Element_In_Tuple* get_added_element() {
		return added_element;
	}
	inline void set_num_vertices(unsigned num) {
		added_element->key_index = (BYTE)num;
	}
	inline unsigned get_num_vertices() {
		return added_element->key_index;
	}
protected:
	unsigned capacity;
	Element_In_Tuple* added_element;
};

#endif /* STRUCT_MINING_TUPLE_HPP_ */
