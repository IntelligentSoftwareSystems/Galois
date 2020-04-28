#pragma once
#include "gg.h"
#include "types.h"
#include "csr_graph.h"
#include "embedding.cuh"

class CUDA_Context_Common {
public:
	int device;
	int id;
	CSRGraph gg;
	MGraph *hg;
	void build_graph_gpu() {
		int m = hg->num_vertices();
		int nnz = hg->num_edges();
		index_type *row_offsets = hg->out_rowptr();
		index_type *column_indices = hg->out_colidx();
		ValueT *labels = (ValueT *)malloc(m * sizeof(ValueT));
		for (int i = 0; i < m; i++) labels[i] = hg->getData(i);
		gg.init_from_mgraph(m, nnz, row_offsets, column_indices, labels);
	}
};

class CUDA_Context_Mining : public CUDA_Context_Common {
public:
	unsigned max_level;
	EmbeddingList emb_list;
};

