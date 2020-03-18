#pragma once
#include "embedding.h"
#include "bliss/uintseqhash.hh"

// Vertex-induced embedding with hash value
class VertexInducedEmbedding: public Embedding<SimpleElement> {
friend std::ostream & operator<<(std::ostream & strm, const VertexInducedEmbedding& emb);
public:
	VertexInducedEmbedding() : Embedding() { hash_value = 0; }
	VertexInducedEmbedding(size_t n) : Embedding(n) { hash_value = 0; }
	VertexInducedEmbedding(const VertexInducedEmbedding &emb) : Embedding() {
		elements = emb.get_elements();
		hash_value = emb.get_pid();
	}
	~VertexInducedEmbedding() {}
	SimpleElement operator[](size_t i) const { return elements[i]; }
	VertexInducedEmbedding& operator=(const VertexInducedEmbedding& other) {
		if(this == &other) return *this;
		elements = other.get_elements();
		hash_value = other.get_pid();
		return *this;
	}
	inline unsigned get_pid() const { return hash_value; } // get the pattern id
	inline void set_pid(unsigned i) { hash_value = i; } // set the pattern id
	inline unsigned get_hash() const {
		bliss::UintSeqHash h;
		for(unsigned i = 0; i < size(); ++i)
			h.update(elements[i].get_vid());
		return h.get_value();
	}
	friend bool operator==(const VertexInducedEmbedding &e1, const VertexInducedEmbedding &e2) {
		return e1.elements == e2.elements;
	}

protected:
	unsigned hash_value;
};

namespace std {
	template<>
	struct hash<VertexInducedEmbedding> {
		std::size_t operator()(const VertexInducedEmbedding& emb) const {
			return std::hash<int>()(emb.get_hash());
		}
	};
}

typedef VertexInducedEmbedding VertexEmbedding;
