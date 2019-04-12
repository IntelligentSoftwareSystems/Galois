
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

class MTuple_simple {
	friend std::ostream & operator<<(std::ostream & strm, const MTuple_simple& cg);
public:
	MTuple_simple(unsigned size_of_t) {
		size = size_of_t / sizeof(Base_Element);
		elements = nullptr;
	}
	virtual ~MTuple_simple() {}
	void init(std::vector<Base_Element> & tuple) { elements = tuple.data(); }
	virtual Base_Element& at(unsigned index) { return elements[index];}
	bool operator==(const MTuple_simple& other) const {
		assert(size == other.size);
		for(unsigned int i = 0; i < size; ++i)
			if(elements[i].id != other.elements[i].id)
				return false;
		return true;
	}
	virtual inline unsigned get_hash() const {
		bliss::UintSeqHash h;
		for(unsigned i = 0; i < size; ++i)
			h.update(elements[i].id);
		return h.get_value();
	}
	inline unsigned get_size() const {
		return size;
	}
	inline Base_Element* get_elements() {
		return elements;
	}
protected:
	unsigned size;
	Base_Element* elements;
};

std::ostream & operator<<(std::ostream & strm, const MTuple_simple& tuple) {
	if(tuple.get_size() == 0) {
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

class MTuple_join_simple : public MTuple_simple {
	friend std::ostream & operator<<(std::ostream & strm, const MTuple_join_simple& cg);
public:
	MTuple_join_simple(unsigned size_of_t) : MTuple_simple(size_of_t) {
		capacity = size + 1;
		added_element = nullptr;
	}
	virtual ~MTuple_join_simple() {}
	Base_Element& at(unsigned index) {
		if(index == capacity - 1)
			return *added_element;
		return elements[index];
	}
	void push(Base_Element* element) {
		added_element = element;
		size++;
	}
	void pop() {
		added_element = nullptr;
		size--;
	}
	inline unsigned get_hash() {
		bliss::UintSeqHash h;
		for(unsigned int i = 0; i < size; ++i)
			h.update(elements[i].id);
		h.update(added_element->id);
		return h.get_value();
	}
	inline Base_Element* get_added_element() {
		return added_element;
	}
private:
	unsigned int capacity;
	Base_Element* added_element;
};

namespace std {
	template<>
	struct hash<MTuple_simple> {
		std::size_t operator()(const MTuple_simple& qp) const {
			return std::hash<int>()(qp.get_hash());
		}
	};
}

std::ostream & operator<<(std::ostream & strm, const MTuple_join_simple& tuple) {
	if(tuple.get_size() == 0) {
		strm << "(empty)";
		return strm;
	}
	strm << "(";
	if(tuple.size == tuple.capacity) {
		for(unsigned int index = 0; index < tuple.get_size() - 1; ++index)
			strm << tuple.elements[index] << ", ";
		strm << *(tuple.added_element);
	}
	else {
		for(unsigned int index = 0; index < tuple.get_size() - 1; ++index)
			strm << tuple.elements[index] << ", ";
		strm << tuple.elements[tuple.get_size() - 1];
	}
	strm << ")";
	return strm;
}

#endif /* MINING_TUPLE_HPP_ */
