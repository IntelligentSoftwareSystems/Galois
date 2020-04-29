#ifndef EMBEDDING_CUH_
#define EMBEDDING_CUH_

#include "element.cuh"

template <typename ElementTy>
class Embedding {
public:
	Embedding() { size_ = 0; }
	Embedding(size_t n) { size_ = n; elements = new ElementTy[size_]; } // TODO
	//Embedding(const Embedding &emb) { size_ = emb.size(); elements = emb.elements; }
	~Embedding() { }
	__device__ IndexT get_vertex(unsigned i) const { return elements[i].get_vid(); }
	__device__ history_type get_history(unsigned i) const { return elements[i].get_his(); }
	__device__ node_data_type get_label(unsigned i) const { return elements[i].get_vlabel(); }
	__device__ key_type get_key(unsigned i) const { return elements[i].get_key(); }
	__device__ bool empty() const { return size_ == 0; }
	__device__ size_t size() const { return size_; }
	__device__ ElementTy& back() { return elements[size_-1]; }
	__device__ const ElementTy& back() const { return elements[size_-1]; }
	__device__ ElementTy get_element(unsigned i) const { return elements[i]; }
	__device__ void set_element(unsigned i, ElementTy &ele) { elements[i] = ele; }
	__device__ void set_vertex(unsigned i, IndexT vid) { elements[i].set_vertex_id(vid); }
	//__device__ unsigned insert(unsigned pos, const ElementTy& value ) { return elements[pos] = value; }
	//__device__ ElementTy* data() { return elements; }
	//__device__ const ElementTy* data() const { return elements; }
	//__device__ ElementTy* get_elements() const { return elements; }
protected:
	ElementTy *elements;
	size_t size_;
};


class BaseEmbedding : public Embedding<SimpleElement> {
public:
	BaseEmbedding() {}
	BaseEmbedding(size_t n) : Embedding(n) {}
	~BaseEmbedding() {}
};

#ifdef USE_BASE_TYPES
typedef BaseEmbedding EmbeddingType;
#endif

template <typename EmbeddingTy>
class EmbeddingQueue{
public:
	EmbeddingQueue() {}
	~EmbeddingQueue() {}
	void init(int nedges, unsigned max_size = 2, bool use_dag = true) {
		int nnz = nedges;
		if (!use_dag) nnz = nnz / 2;
		size = nedges;
	}
	EmbeddingTy *queue;
	int size;
};

class EmbeddingList {
public:
	EmbeddingList() {}
	~EmbeddingList() {}
	void init(int nedges, unsigned max_size = 2, bool use_dag = true) {
		last_level = 1;
		assert(max_size > 1);
		max_level = max_size;
		h_vid_lists = (IndexT **)malloc(max_level * sizeof(IndexT*));
		h_idx_lists = (IndexT **)malloc(max_level * sizeof(IndexT*));
		check_cuda(cudaMalloc(&d_vid_lists, max_level * sizeof(IndexT*)));
		check_cuda(cudaMalloc(&d_idx_lists, max_level * sizeof(IndexT*)));
		#ifdef ENABLE_LABEL
		h_his_lists = (history_type **)malloc(max_level * sizeof(history_type*));
		check_cuda(cudaMalloc(&d_his_lists, max_level * sizeof(history_type*)));
		#endif
		sizes = new size_t[max_level];
		sizes[0] = 0;
		int nnz = nedges;
		if (!use_dag) nnz = nnz / 2;
		sizes[1] = nnz;
		check_cuda(cudaMalloc((void **)&h_vid_lists[1], nnz * sizeof(IndexT)));
		check_cuda(cudaMalloc((void **)&h_idx_lists[1], nnz * sizeof(IndexT)));
		check_cuda(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		check_cuda(cudaMalloc((void **)&h_his_lists[1], nnz * sizeof(history_type)));
		check_cuda(cudaMemcpy(d_his_lists, h_his_lists, max_level * sizeof(history_type*), cudaMemcpyHostToDevice));
		#endif
	}
	void init_cpu(CSRGraph *graph, bool is_dag = false) {
		int nnz = graph->get_nedges();
		if (!is_dag) nnz = nnz / 2;
		IndexT *vid_list = (IndexT *)malloc(nnz*sizeof(IndexT));
		IndexT *idx_list = (IndexT *)malloc(nnz*sizeof(IndexT));
		int eid = 0;
		for (int src = 0; src < graph->get_nnodes(); src ++) {
			IndexT row_begin = graph->edge_begin(src);
			IndexT row_end = graph->edge_end(src);
			for (IndexT e = row_begin; e < row_end; e++) {
				IndexT dst = graph->getEdgeDst(e);
				if (is_dag || src < dst) {
					vid_list[eid] = dst;
					idx_list[eid] = src;
					eid ++;
				}
			}
		}
		check_cuda(cudaMemcpy(h_vid_lists[1], vid_list, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(h_idx_lists[1], idx_list, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		check_cuda(cudaMemset(h_his_lists[1], 0, nnz * sizeof(history_type)));
		#endif
	}
	__device__ IndexT get_vid(unsigned level, IndexT id) const { return d_vid_lists[level][id]; }
	__device__ IndexT get_idx(unsigned level, IndexT id) const { return d_idx_lists[level][id]; }
	__device__ history_type get_his(unsigned level, IndexT id) const { return d_his_lists[level][id]; }
	__device__ unsigned get_pid(IndexT id) const { return pid_list[id]; }
	__device__ void set_vid(unsigned level, IndexT id, IndexT vid) { d_vid_lists[level][id] = vid; }
	__device__ void set_idx(unsigned level, IndexT id, IndexT idx) { d_idx_lists[level][id] = idx; }
	__device__ void set_his(unsigned level, IndexT id, history_type lab) { d_his_lists[level][id] = lab; }
	__device__ void set_pid(IndexT id, unsigned pid) { pid_list[id] = pid; }
	size_t size() const { return sizes[last_level]; }
	size_t size(unsigned level) const { return sizes[level]; }
	//__device__ VertexList get_vid_list(unsigned level) { return vid_lists[level]; }
	//__device__ UintList get_idx_list(unsigned level) { return idx_lists[level]; }
	//__device__ ByteList get_his_list(unsigned level) { return his_lists[level]; }
	void add_level(unsigned size) { // TODO: this size could be larger than 2^32, when running LiveJournal and even larger graphs
		last_level ++;
		assert(last_level < max_level);
		check_cuda(cudaMalloc((void **)&h_vid_lists[last_level], size * sizeof(IndexT)));
		check_cuda(cudaMalloc((void **)&h_idx_lists[last_level], size * sizeof(IndexT)));
		#ifdef ENABLE_LABEL
		check_cuda(cudaMalloc((void **)&h_his_lists[last_level], size * sizeof(history_type)));
		#endif
		#ifdef USE_PID
		check_cuda(cudaMalloc((void **)&pid_list, size * sizeof(unsigned)));
		#endif
		check_cuda(cudaMemcpy(d_vid_lists, h_vid_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(d_idx_lists, h_idx_lists, max_level * sizeof(IndexT*), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		check_cuda(cudaMemcpy(d_his_lists, h_his_lists, max_level * sizeof(history_type*), cudaMemcpyHostToDevice));
		#endif
		sizes[last_level] = size;
	}
	void remove_tail(unsigned idx) { sizes[last_level] = idx; }
	void reset_level() {
		for (size_t i = 2; i <= last_level; i ++) {
			check_cuda(cudaFree(h_vid_lists[i]));
			check_cuda(cudaFree(h_idx_lists[i]));
		}
		last_level = 1;
	}

	/*
	void printout_embeddings(int level, bool verbose = false) {
		std::cout << "number of embeddings in level " << level << ": " << size() << std::endl;
		if(verbose) {
			for (size_t pos = 0; pos < size(); pos ++) {
				embeddingtype emb(last_level+1);
				get_embedding(last_level, pos, emb);
				std::cout << emb << "\n";
			}
		}
	}
	*/
	__device__ void get_embedding(unsigned level, unsigned pos, IndexT *emb) {
		IndexT vid = get_vid(level, pos);
		IndexT idx = get_idx(level, pos);
		emb[level] = vid;
		for (unsigned l = 1; l < level; l ++) {
			vid = get_vid(level-l, idx);
			emb[level-l] = vid;
			idx = get_idx(level-l, idx);
		}
		emb[0] = idx;
	}
	__device__ void get_edge_embedding(unsigned level, unsigned pos, IndexT *vids, history_type *hiss) {
		IndexT vid = get_vid(level, pos);
		IndexT idx = get_idx(level, pos);
		history_type his = get_his(level, pos);
		vids[level] = vid;
		hiss[level] = his;
		for (unsigned l = 1; l < level; l ++) {
			vid = get_vid(level-l, idx);
			his = get_his(level-l, idx);
			vids[level-l] = vid;
			hiss[level-l] = his;
			idx = get_idx(level-l, idx);
		}
		vids[0] = idx;
		hiss[0] = 0;
	}

private:
	unsigned max_level;
	unsigned last_level;
	size_t *sizes;
	unsigned *pid_list;
	IndexT** h_idx_lists;
	IndexT** h_vid_lists;
	history_type** h_his_lists;
	IndexT** d_idx_lists;
	IndexT** d_vid_lists;
	history_type** d_his_lists;
};

__global__ void init_gpu_dag(int m, CSRGraph graph, EmbeddingList emb_list) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			emb_list.set_vid(1, e, dst);
			emb_list.set_idx(1, e, src);
		}
	}
}

__global__ void init_alloc(int m, CSRGraph graph, EmbeddingList emb_list, IndexT *num_emb) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		num_emb[src] = 0;
		#ifdef ENABLE_LABEL
		node_data_type src_label = graph.getData(src);
		#endif
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			#ifdef ENABLE_LABEL
			node_data_type dst_label = graph.getData(dst);
			#endif
			#ifdef ENABLE_LABEL
			if (src_label <= dst_label) num_emb[src] ++;
			#else
			if (src < dst) num_emb[src] ++;
			#endif
		}
	}
}

__global__ void init_insert(int m, CSRGraph graph, EmbeddingList emb_list, IndexT *indices) {
	unsigned src = blockIdx.x * blockDim.x + threadIdx.x;
	if (src < m) {
		#ifdef ENABLE_LABEL
		node_data_type src_label = graph.getData(src);
		#endif
		IndexT start = indices[src];
		IndexT row_begin = graph.edge_begin(src);
		IndexT row_end = graph.edge_end(src);
		for (IndexT e = row_begin; e < row_end; e++) {
			IndexT dst = graph.getEdgeDst(e);
			#ifdef ENABLE_LABEL
			node_data_type dst_label = graph.getData(dst);
			#endif
			#ifdef ENABLE_LABEL
			if (src_label <= dst_label) {
			#else
			if (src < dst) {
			#endif
				emb_list.set_vid(1, start, dst);
				emb_list.set_idx(1, start, src);
				#ifdef ENABLE_LABEL
				emb_list.set_his(1, start, 0);
				#endif
				start ++;
			}
		}
	}
}

#endif // EMBEDDING_CUH_
