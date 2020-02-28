#pragma once
#include <cassert>
#include <fstream>
#include "checker.h"

typedef unsigned IndexT;
typedef unsigned index_type;
typedef int edge_data_type;
typedef int node_data_type;

class GraphGPU {
protected:
	IndexT *d_row_offsets;
	IndexT *d_column_indices;
	BYTE *d_labels;
	int *d_degrees;
	int num_vertices;
	int num_edges;
	int nedges;
	int nnodes;
public:
	GraphGPU() {}
	~GraphGPU() {}
	void clean() {
		CUDA_SAFE_CALL(cudaFree(d_row_offsets));
		CUDA_SAFE_CALL(cudaFree(d_column_indices));
	}
	/*
	void init(Graph *hg) {
		int m = hg->num_vertices();
		int nnz = hg->num_edges();
		num_vertices = m;
		num_edges = nnz;
		IndexT *h_row_offsets = hg->out_rowptr();
		IndexT *h_column_indices = hg->out_colidx();
		int *h_degrees = (int *)malloc(m * sizeof(int));
		for (int i = 0; i < m; i++) h_degrees[i] = h_row_offsets[i + 1] - h_row_offsets[i];
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_row_offsets, (m + 1) * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_column_indices, nnz * sizeof(IndexT)));
		CUDA_SAFE_CALL(cudaMemcpy(d_row_offsets, h_row_offsets, (m + 1) * sizeof(IndexT), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_column_indices, h_column_indices, nnz * sizeof(IndexT), cudaMemcpyHostToDevice));
		#ifdef ENABLE_LABEL
		BYTE *h_labels = (BYTE *)malloc(m * sizeof(BYTE));
		for (int i = 0; i < m; i++) h_labels[i] = hg->getData(i);
		CUDA_SAFE_CALL(cudaMalloc((void **)&d_labels, m * sizeof(BYTE)));
		CUDA_SAFE_CALL(cudaMemcpy(d_labels, h_labels, m * sizeof(BYTE), cudaMemcpyHostToDevice));
		#endif
	}
	*/
	__device__ __host__ bool valid_node(IndexT node) { return (node < num_vertices); }
	__device__ __host__ bool valid_edge(IndexT edge) { return (edge < num_edges); }
	__device__ __host__ IndexT getOutDegree(unsigned src) {
		assert(src < num_vertices);
		return d_row_offsets[src+1] - d_row_offsets[src];
	};
	__device__ __host__ IndexT getDestination(unsigned src, unsigned edge) {
		assert(src < num_vertices);
		assert(edge < getOutDegree(src));
		IndexT abs_edge = d_row_offsets[src] + edge;
		assert(abs_edge < num_edges);
		return d_column_indices[abs_edge];
	};
	__device__ __host__ IndexT getAbsDestination(unsigned abs_edge) {
		assert(abs_edge < num_edges);
		return d_column_indices[abs_edge];
	};
	inline __device__ __host__ IndexT getEdgeDst(unsigned edge) {
		assert(edge < num_edges);
		return d_column_indices[edge];
	};
	inline __device__ __host__ BYTE getData(unsigned vid) {
		return d_labels[vid];
	}
	inline __device__ __host__ IndexT edge_begin(unsigned src) {
		assert(src <= num_vertices);
		return d_row_offsets[src];
	};
	inline __device__ __host__ IndexT edge_end(unsigned src) {
		assert(src <= num_vertices);
		return d_row_offsets[src+1];
	};

	void readFromGR(const char file[], bool read_edge_data) {
		std::ifstream cfile;
		cfile.open(file);
		// copied from GaloisCpp/trunk/src/FileGraph.h
		int masterFD = open(file, O_RDONLY);
		if (masterFD == -1) {
			printf("FileGraph::structureFromFile: unable to open %s.\n", file);
			return 1;
		}
		struct stat buf;
		int f = fstat(masterFD, &buf);
		if (f == -1) {
			printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
			abort();
		}
		size_t masterLength = buf.st_size;
		int _MAP_BASE = MAP_PRIVATE;
		void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
		if (m == MAP_FAILED) {
			m = 0;
			printf("FileGraph::structureFromFile: mmap failed.\n");
			abort();
		}
		//ggc::Timer t("graphreader");
		t.start();
		//parse file
		uint64_t* fptr = (uint64_t*)m;
		__attribute__((unused)) uint64_t version = le64toh(*fptr++);
		assert(version == 1);
		uint64_t sizeEdgeTy = le64toh(*fptr++);
		uint64_t numNodes = le64toh(*fptr++);
		uint64_t numEdges = le64toh(*fptr++);
		uint64_t *outIdx = fptr;
		fptr += numNodes;
		uint32_t *fptr32 = (uint32_t*)fptr;
		uint32_t *outs = fptr32; 
		fptr32 += numEdges;
		if (numEdges % 2) fptr32 += 1;
		edge_data_type  *edgeData = (edge_data_type *)fptr32;
		
		// cuda.
		nnodes = numNodes;
		nedges = numEdges;
		printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", nnodes, nedges, sizeEdgeTy);
		allocOnHost(!read_edge_data);
		row_start[0] = 0;
		for (unsigned ii = 0; ii < nnodes; ++ii) {
			row_start[ii+1] = le64toh(outIdx[ii]);
			index_type degree = row_start[ii+1] - row_start[ii];
			for (unsigned jj = 0; jj < degree; ++jj) {
				unsigned edgeindex = row_start[ii] + jj;
				unsigned dst = le32toh(outs[edgeindex]);
				if (dst >= nnodes) printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);
				edge_dst[edgeindex] = dst;
				if(sizeEdgeTy && read_edge_data)
					edge_data[edgeindex] = edgeData[edgeindex];
			}
		}
		cfile.close();	// probably galois doesn't close its file due to mmap.
		//t.stop();
		//printf("read %lld bytes in %d ms (%0.2f MB/s)\n\r\n", masterLength, t.duration_ms(), (masterLength / 1000.0) / (t.duration_ms()));
		return 0;
	}

	void copy_to_gpu(struct CSRGraph &copygraph) {
		copygraph.nnodes = nnodes;
		copygraph.nedges = nedges;
		assert(copygraph.allocOnDevice(edge_data == NULL));
		check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst, nedges * sizeof(index_type), cudaMemcpyHostToDevice));
		if (edge_data != NULL) check_cuda(cudaMemcpy(copygraph.edge_data, edge_data, nedges * sizeof(edge_data_type), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(copygraph.node_data, node_data, nnodes * sizeof(node_data_type), cudaMemcpyHostToDevice));
		check_cuda(cudaMemcpy(copygraph.row_start, row_start, (nnodes+1) * sizeof(index_type), cudaMemcpyHostToDevice));
	}

};
