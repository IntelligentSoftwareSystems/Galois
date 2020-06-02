#pragma once
#include <set>
#include <vector>
#include <string>
#include <cassert>
#include <fstream>
#include <fcntl.h>
#include <cassert>
#include <unistd.h>
#include <stdint.h>
#include <algorithm>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include "pangolin/types.cuh"
#include "pangolin/checker.h"
#include "pangolin/timer.h"

struct Edge {
  Edge(IndexT from, IndexT to) : src(from), dst(to) {}
  IndexT src;
  IndexT dst;
};

std::vector<std::vector<Edge>> vertices;

class CSRGraph {
protected:
  IndexT* row_start;
  IndexT* edge_dst;
  node_data_type* node_data;
  int nnodes;
  int nedges;
  bool need_dag;
  bool device_graph;
  bool use_node_data;

public:
  CSRGraph() { init(); }
  //~CSRGraph() {}
  void init() {
    row_start = edge_dst = NULL;
    node_data            = NULL;
    nnodes = nedges = 0;
    need_dag        = false;
    device_graph    = false;
    use_node_data   = false;
  }
  void enable_dag() { need_dag = true; }
  int get_nnodes() { return nnodes; }
  int get_nedges() { return nedges; }
  void clean() {
    check_cuda(cudaFree(row_start));
    check_cuda(cudaFree(edge_dst));
  }
  __device__ __host__ bool valid_node(IndexT node) { return (node < nnodes); }
  __device__ __host__ bool valid_edge(IndexT edge) { return (edge < nedges); }
  __device__ __host__ IndexT getOutDegree(unsigned src) {
    assert(src < nnodes);
    return row_start[src + 1] - row_start[src];
  };
  __device__ __host__ IndexT getDestination(unsigned src, unsigned edge) {
    assert(src < nnodes);
    assert(edge < getOutDegree(src));
    IndexT abs_edge = row_start[src] + edge;
    assert(abs_edge < nedges);
    return edge_dst[abs_edge];
  };
  __device__ __host__ IndexT getAbsDestination(unsigned abs_edge) {
    assert(abs_edge < nedges);
    return edge_dst[abs_edge];
  };
  inline __device__ __host__ IndexT getEdgeDst(unsigned edge) {
    assert(edge < nedges);
    return edge_dst[edge];
  };
  inline __device__ __host__ node_data_type getData(unsigned vid) {
    return node_data[vid];
  }
  inline __device__ __host__ IndexT edge_begin(unsigned src) {
    assert(src <= nnodes);
    return row_start[src];
  };
  inline __device__ __host__ IndexT edge_end(unsigned src) {
    assert(src <= nnodes);
    return row_start[src + 1];
  };
  int read(std::string file, bool read_node_data = true, bool dag = false) {
    std::cout << "Reading graph fomr file: " << file << "\n";
    need_dag = dag;
    if (read_node_data) {
      use_node_data = true;
      return read_adj(file.c_str());
    } else {
      use_node_data = false;
      readFromGR(file.c_str());
    }
    return 0;
  }
  void readFromGR(const char file[]) {
    std::ifstream cfile;
    cfile.open(file);
    int masterFD = open(file, O_RDONLY);
    if (masterFD == -1) {
      printf("FileGraph::structureFromFile: unable to open %s.\n", file);
      return;
    }
    struct stat buf;
    int f = fstat(masterFD, &buf);
    if (f == -1) {
      printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
      abort();
    }
    size_t masterLength = buf.st_size;
    int _MAP_BASE       = MAP_PRIVATE;
    void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
    if (m == MAP_FAILED) {
      m = 0;
      printf("FileGraph::structureFromFile: mmap failed.\n");
      abort();
    }
    Timer t;
    t.Start();
    uint64_t* fptr                           = (uint64_t*)m;
    __attribute__((unused)) uint64_t version = le64toh(*fptr++);
    assert(version == 1);
    uint64_t sizeEdgeTy = le64toh(*fptr++);
    uint64_t numNodes   = le64toh(*fptr++);
    uint64_t numEdges   = le64toh(*fptr++);
    uint64_t* outIdx    = fptr;
    fptr += numNodes;
    uint32_t* fptr32 = (uint32_t*)fptr;
    uint32_t* outs   = fptr32;
    fptr32 += numEdges;
    if (numEdges % 2)
      fptr32 += 1;
    nnodes = numNodes;
    nedges = numEdges;
    printf("nnodes=%d, nedges=%d, sizeEdge=%d.\n", nnodes, nedges, sizeEdgeTy);
    row_start    = (index_type*)calloc(nnodes + 1, sizeof(index_type));
    edge_dst     = (index_type*)calloc(nedges, sizeof(index_type));
    row_start[0] = 0;
    for (unsigned ii = 0; ii < nnodes; ++ii) {
      row_start[ii + 1] = le64toh(outIdx[ii]);
      index_type degree = row_start[ii + 1] - row_start[ii];
      for (unsigned jj = 0; jj < degree; ++jj) {
        unsigned edgeindex = row_start[ii] + jj;
        unsigned dst       = le32toh(outs[edgeindex]);
        if (dst >= nnodes)
          printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj,
                 edgeindex);
        edge_dst[edgeindex] = dst;
      }
    }
    cfile.close(); // probably galois doesn't close its file due to mmap.
    t.Stop();
    double runtime = t.Millisecs();
    printf("read %lld bytes in %.1f ms (%0.2f MB/s)\n\r\n", masterLength,
           runtime, (masterLength / 1000.0) / runtime);
    if (need_dag) {
      reconstruct_from_csr();
      SquishGraph();
      MakeCSR(vertices);
      vertices.clear();
    }
    return;
  }
  void reconstruct_from_csr() {
    vertices.resize(nnodes);
    std::cout << "Reconstructing from CSR graph ... ";
    for (int i = 0; i < nnodes; i++) {
      std::vector<Edge> neighbors;
      for (IndexT j = row_start[i]; j < row_start[i + 1]; j++)
        neighbors.push_back(Edge(i, edge_dst[j]));
      vertices[i] = neighbors;
    }
    std::cout << "Done\n";
  }
  int read_adj(const char* filename) {
    FILE* fd = fopen(filename, "r");
    assert(fd != NULL);
    char buf[2048];
    unsigned size = 0, maxsize = 0;
    int numNodes = 0;
    while (fgets(buf, 2048, fd) != NULL) {
      auto len = strlen(buf);
      size += len;
      if (buf[len - 1] == '\n') {
        maxsize = std::max(size, maxsize);
        size    = 0;
        numNodes++;
      }
    }
    fclose(fd);
    nnodes = numNodes;
    printf("nnodes=%d.\n", nnodes);
    std::ifstream is;
    is.open(filename, std::ios::in);
    char* line = new char[maxsize + 1];
    std::vector<std::string> result;
    nedges    = 0;
    node_data = (node_data_type*)calloc(nnodes, sizeof(node_data_type));
    vertices.resize(nnodes);
    std::vector<Edge> neighbors;
    for (size_t i = 0; i < nnodes; i++)
      vertices.push_back(neighbors);
    int line_count = 0;
    while (is.getline(line, maxsize + 1)) {
      result.clear();
      split(line, result);
      IndexT src = atoi(result[0].c_str());
      assert(src == line_count);
      assert(src < nnodes);
      node_data[src] = atoi(result[1].c_str());
      for (size_t i = 2; i < result.size(); i++) {
        IndexT dst = atoi(result[i].c_str());
        if (src == dst)
          continue; // remove self-loop
        vertices[src].push_back(Edge(src, dst));
        nedges++;
      }
      line_count++;
    }
    is.close();
    printf("nedges=%d\n", nedges);
    int num_labels = count_unique_labels();
    std::cout << "Number of unique vertex label values: " << num_labels
              << std::endl;
    SquishGraph();
    printf("nedges after clean: %d\n", nedges);
    row_start = (index_type*)calloc(nnodes + 1, sizeof(index_type));
    edge_dst  = (index_type*)calloc(nedges, sizeof(index_type));
    MakeCSR(vertices);
    vertices.clear();
    return num_labels;
  }

  void MakeCSR(const std::vector<std::vector<Edge>> vert) {
    printf("Constructing CSR graph ... ");
    std::vector<IndexT> offsets(nnodes + 1);
    IndexT total = 0;
    for (int i = 0; i < nnodes; i++) {
      offsets[i] = total;
      total += vert[i].size();
    }
    offsets[nnodes] = total;
    assert(nedges == offsets[nnodes]);
    assert(row_start != NULL);
    for (size_t i = 0; i < nnodes + 1; i++)
      row_start[i] = offsets[i];
    for (size_t i = 0; i < nnodes; i++) {
      for (auto e : vert[i]) {
        if (i != e.src)
          std::cout << "[debug] i = " << i << ", src = " << e.src
                    << ", dst = " << e.dst << "\n";
        assert(i == e.src);
        edge_dst[offsets[e.src]++] = e.dst;
      }
    }
    printf("Done\n");
  }

  static bool compare_id(Edge a, Edge b) { return (a.dst < b.dst); }
  void SquishGraph(bool remove_selfloops  = true,
                   bool remove_redundents = true) {
    printf("Sorting the neighbor lists...");
    for (size_t i = 0; i < nnodes; i++)
      std::sort(vertices[i].begin(), vertices[i].end(), compare_id);
    printf(" Done\n");
    // remove self loops
    int num_selfloops = 0;
    if (remove_selfloops) {
      printf("Removing self loops...");
      for (size_t i = 0; i < nnodes; i++) {
        for (unsigned j = 0; j < vertices[i].size(); j++) {
          if (i == vertices[i][j].dst) {
            vertices[i].erase(vertices[i].begin() + j);
            num_selfloops++;
            j--;
          }
        }
      }
      printf(" %d selfloops are removed\n", num_selfloops);
      nedges -= num_selfloops;
    }
    // remove redundent
    int num_redundents = 0;
    if (remove_redundents) {
      printf("Removing redundent edges...");
      for (size_t i = 0; i < nnodes; i++) {
        for (unsigned j = 1; j < vertices[i].size(); j++) {
          if (vertices[i][j].dst == vertices[i][j - 1].dst) {
            vertices[i].erase(vertices[i].begin() + j);
            num_redundents++;
            j--;
          }
        }
      }
      printf(" %d redundent edges are removed\n", num_redundents);
      nedges -= num_redundents;
    }
    if (need_dag) {
      int num_dag = 0;
      std::cout << "Constructing DAG...";
      IndexT* degrees = new IndexT[nnodes];
      for (size_t i = 0; i < nnodes; i++)
        degrees[i] = vertices[i].size();
      for (size_t i = 0; i < nnodes; i++) {
        for (unsigned j = 0; j < vertices[i].size(); j++) {
          IndexT to = vertices[i][j].dst;
          auto di   = degrees[i];
          if (degrees[to] < di || (degrees[to] == di && to < i)) {
            vertices[i].erase(vertices[i].begin() + j);
            num_dag++;
            j--;
          }
        }
      }
      delete degrees;
      printf(" %d dag edges are removed\n", num_dag);
      nedges -= num_dag;
    }
  }

  int count_unique_labels() {
    std::set<node_data_type> s;
    int res = 0;
    for (int i = 0; i < nnodes; i++) {
      if (s.find(node_data[i]) == s.end()) {
        s.insert(node_data[i]);
        res++;
      }
    }
    return res;
  }

  inline void split(const std::string& str, std::vector<std::string>& tokens,
                    const std::string& delimiters = " ") {
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      lastPos = str.find_first_not_of(delimiters, pos);
      pos     = str.find_first_of(delimiters, lastPos);
    }
  }

  void copy_to_gpu(struct CSRGraph& copygraph) {
    copygraph.nnodes = nnodes;
    copygraph.nedges = nedges;
    auto error       = copygraph.allocOnDevice(use_node_data);
    if (error == 0) {
      std::cout << "GPU memory allocation failed\n";
      exit(0);
    }
    printf("edge_dst: host_ptr %x device_ptr %x \n", edge_dst,
           copygraph.edge_dst);
    check_cuda(cudaMemcpy(copygraph.edge_dst, edge_dst,
                          nedges * sizeof(index_type), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(copygraph.row_start, row_start,
                          (nnodes + 1) * sizeof(index_type),
                          cudaMemcpyHostToDevice));
    if (use_node_data)
      check_cuda(cudaMemcpy(copygraph.node_data, node_data,
                            nnodes * sizeof(node_data_type),
                            cudaMemcpyHostToDevice));
  }

  unsigned allocOnHost() {
    assert(nnodes > 0);
    assert(!device_graph);
    if (row_start != NULL)
      return true;
    std::cout << "Allocating memory on CPU\n";
    if (use_node_data)
      std::cout << "Need node data\n";
    size_t mem_usage = ((nnodes + 1) + nedges) * sizeof(index_type);
    if (use_node_data)
      mem_usage += (nnodes) * sizeof(node_data_type);
    printf("Host memory for graph: %3u MB\n", mem_usage / 1048756);
    row_start = (index_type*)calloc(nnodes + 1, sizeof(index_type));
    edge_dst  = (index_type*)calloc(nedges, sizeof(index_type));
    if (use_node_data)
      node_data = (node_data_type*)calloc(nnodes, sizeof(node_data_type));
    std::cout << "Memory allocation done\n";
    return ((!use_node_data || node_data) && row_start && edge_dst);
  }

  unsigned allocOnDevice(bool use_label) {
    if (edge_dst != NULL) {
      std::cout << "already allocated\n";
      exit(0);
    }
    assert(edge_dst == NULL); // make sure not already allocated
    device_graph = true;
    std::cout << "Allocating memory on GPU\n";
    check_cuda(cudaMalloc((void**)&edge_dst, nedges * sizeof(index_type)));
    check_cuda(
        cudaMalloc((void**)&row_start, (nnodes + 1) * sizeof(index_type)));
    if (use_label)
      check_cuda(
          cudaMalloc((void**)&node_data, nnodes * sizeof(node_data_type)));
    return (edge_dst && (!use_node_data || node_data) && row_start);
  }
};
