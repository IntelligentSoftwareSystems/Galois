#include "deepgalois/lgraph.h"
#include "deepgalois/utils.h"
#include "galois/Galois.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>    /* For O_RDWR */
#include <unistd.h>   /* For open(), creat() */
#include <fstream>
#include <iostream>
#include <cassert>

namespace deepgalois {

bool LearningGraph::isLocal(index_t vid) { return true; }

index_t LearningGraph::getLID(index_t vid) { return 0; }

bool LearningGraph::is_vertex_cut() {return true; }

std::vector<std::vector<size_t>>& LearningGraph::getMirrorNodes() {
  return mirrorNodes;
}

uint64_t LearningGraph::numMasters() { return 0; }

uint64_t LearningGraph::globalSize() { return 0; }

void LearningGraph::progressPrint(unsigned maxii, unsigned ii) {
  const unsigned nsteps = 10;
  unsigned ineachstep = (maxii / nsteps);
  if(ineachstep == 0) ineachstep = 1;
  if (ii % ineachstep == 0) {
    int progress = ((size_t) ii * 100) / maxii + 1;
    printf("\t%3d%%\r", progress);
    fflush(stdout);
  }
}

void LearningGraph::allocateFrom(index_t nv, index_t ne) {
  //printf("Allocating num_vertices_=%d, num_edges_=%d.\n", num_vertices_, num_edges_);
/*
  if (num_vertices_ != nv) {
    if (rowptr_ != NULL) delete [] rowptr_;
    if (degrees_ != NULL) delete [] degrees_;
    if (vertex_data_ != NULL) delete [] vertex_data_;
    num_vertices_ = nv;
  }
  if (num_edges_ != ne) {
    if (colidx_ != NULL) delete [] colidx_;
    if (edge_data_ != NULL) delete [] edge_data_;
    num_edges_ = ne;
  } 
  if (rowptr_ == NULL) rowptr_ = new index_t[num_vertices_+1];
  if (colidx_ == NULL) colidx_ = new index_t[num_edges_];
*/
  num_vertices_ = nv;
  num_edges_ = ne;
  rowptr_.resize(num_vertices_+1);
  colidx_.resize(num_edges_);
  degrees_.resize(num_vertices_);
  rowptr_[0] = 0;
}

void LearningGraph::constructNodes() {
}

void LearningGraph::fixEndEdge(index_t vid, index_t row_end) {
  rowptr_[vid+1] = row_end;
}

void LearningGraph::constructEdge(index_t eid, index_t dst, edata_t edata) {
  assert(dst < num_vertices_);
  assert(eid < num_edges_);
  colidx_[eid] = dst;
}

void LearningGraph::degree_counting() {
  //if (degrees_ != NULL) return;
  //degrees_ = new index_t[num_vertices_];
  galois::do_all(galois::iterate(size_t(0), size_t(num_vertices_)), [&] (auto v) {
    degrees_[v] = rowptr_[v+1] - rowptr_[v];
  }, galois::loopname("DegreeCounting"));
}

void LearningGraph::readGraph(std::string path, std::string dataset) {
  std::string filename = path + dataset + ".csgr";
}

void LearningGraph::readGraphFromGRFile(const std::string& filename) {
  std::ifstream ifs;
  ifs.open(filename);
  int masterFD = open(filename.c_str(), O_RDONLY);
  if (masterFD == -1) {
    std::cout << "LearningGraph: unable to open" << filename << "\n";
    exit(1);
  }
  struct stat buf;
  int f = fstat(masterFD, &buf);
  if (f == -1) {
    std::cout << "LearningGraph: unable to stat" << filename << "\n";
    exit(1);
  }
  size_t masterLength = buf.st_size;
  int _MAP_BASE = MAP_PRIVATE;
  void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
  if (m == MAP_FAILED) {
    m = 0;
    std::cout << "LearningGraph: mmap failed.\n";
    exit(1);
  }
  Timer t;
  t.Start();

  uint64_t* fptr = (uint64_t*)m;
  __attribute__((unused)) uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t nv = le64toh(*fptr++);
  uint64_t ne = le64toh(*fptr++);
  uint64_t *outIdx = fptr;
  fptr += nv;
  uint32_t *fptr32 = (uint32_t*)fptr;
  uint32_t *outs = fptr32; 
  fptr32 += ne;
  if (ne % 2) fptr32 += 1;
  num_vertices_ = nv;
  num_edges_ = ne;
  if (sizeEdgeTy != 0) {
    std::cout << "LearningGraph: currently edge data not supported.\n";
    exit(1);
  }

  printf("num_vertices_=%d, num_edges_=%d.\n", num_vertices_, num_edges_);
  allocateFrom(nv, ne);
  //degrees_ = new index_t[num_vertices_];
  //rowptr_ = new index_t[num_vertices_+1];
  //colidx_ = new index_t[num_edges_];
  //rowptr_[0] = 0;
  for (unsigned ii = 0; ii < num_vertices_; ++ii) {
    rowptr_[ii+1] = le64toh(outIdx[ii]);
    degrees_[ii] = rowptr_[ii+1] - rowptr_[ii];
    for (unsigned jj = 0; jj < degrees_[ii]; ++jj) {
      unsigned eid = rowptr_[ii] + jj;
      unsigned dst = le32toh(outs[eid]);
      if (dst >= num_vertices_) {
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, eid);
        exit(0);
      }
      colidx_[eid] = dst;
    }
    progressPrint(num_vertices_, ii);
  }
  ifs.close();

/*
  std::string file_dims = path + dataset + "-dims.bin";
  std::string file_rowptr = path + dataset + "-rowptr.bin";
  std::string file_colidx = path + dataset + "-colidx.bin";
  index_t dims[2];
  ifs.open(file_dims, std::ios::binary|std::ios::in);
  ifs.read((char*)dims, sizeof(index_t) * 2);
  ifs.close();
  num_vertices_ = dims[0];
  num_edges_ = dims[1];
  degrees_ = new index_t[num_vertices_];
  rowptr_ = new index_t[num_vertices_+1];
  colidx_ = new index_t[num_edges_];
  ifs.open(file_rowptr, std::ios::binary|std::ios::in);
  ifs.read((char*)rowptr_, sizeof(index_t) * (num_vertices_+1));
  ifs.close();
  ifs.open(file_colidx, std::ios::binary|std::ios::in);
  ifs.read((char*)colidx_, sizeof(index_t) * num_edges_);
  ifs.close();
*/
  t.Stop();
  double runtime = t.Millisecs();
  std::cout << "read " << masterLength << " bytes in " << runtime << " ms (" 
            << masterLength/1000.0/runtime << " MB/s)\n\n"; 
}

#ifdef CPU_ONLY
void LearningGraph::dealloc() {
/*
  assert (!is_device);
  if (rowptr_ != NULL) delete [] rowptr_;
  if (colidx_ != NULL) delete [] colidx_;
  if (degrees_ != NULL) delete [] degrees_;
  if (vertex_data_ != NULL) delete [] vertex_data_;
  if (edge_data_ != NULL) delete [] edge_data_;
//*/
}
#endif

} // end namespace
