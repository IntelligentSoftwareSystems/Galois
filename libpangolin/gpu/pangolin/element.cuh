#ifndef ELEMENT_CUH_
#define ELEMENT_CUH_
#include "types.cuh"

struct SimpleElement {
protected:
	IndexT vertex_id;
public:
	SimpleElement() : vertex_id(0) { }
	SimpleElement(IndexT _vertex_id) : vertex_id(_vertex_id) { }
	SimpleElement(IndexT _vertex_id, edge_data_type _edge_label, node_data_type _vertex_label, history_type _history) : vertex_id(_vertex_id) { }
	SimpleElement(IndexT _vertex_id, key_type _key_index, edge_data_type _edge_label, node_data_type _vertex_label, history_type _history) : vertex_id(_vertex_id) { }
	~SimpleElement() { }
	inline __device__ void set_vertex_id(IndexT new_id) { vertex_id = new_id; }
	inline __device__ void set_history_info(history_type his) { }
	inline __device__ void set_vertex_label(node_data_type lab) { }
	inline __device__ IndexT get_vid() const { return vertex_id; }
	inline __device__ history_type get_his() const { return 0; }
	inline __device__ key_type get_key() const { return 0; }
	inline __device__ int cmp(const SimpleElement& other) const {
		if(vertex_id < other.get_vid()) return -1;
		if(vertex_id > other.get_vid()) return 1;
		return 0;
	}
	//friend bool operator==(const SimpleElement &e1, const SimpleElement &e2) {
	//	return e1.get_vid() == e2.get_vid();
	//}
};

#ifdef USE_SIMPLE
typedef SimpleElement ElementType;
#endif
#endif
