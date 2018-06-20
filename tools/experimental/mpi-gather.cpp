/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

#include <mpi.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#ifdef __linux__
#include <linux/mman.h>
#endif
#include <sys/mman.h>
#include <iostream>
#include <string>
#include <system_error>

void printHelp(std::ostream& out) { out << "gather <galois graph.gr>\n"; }

class FileGraph {
  class slice {
    static const int pageSize = 4096;
    void* m;
    void* b;
    size_t len;

  public:
    slice() : m(MAP_FAILED) {}

    void read(size_t begin, size_t size, int fd) {
      size_t align = begin & (pageSize - 1);
      len          = size + align;
      m            = mmap(0, len, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd,
               begin - align);
      if (m == MAP_FAILED)
        throw std::system_error(
            std::error_code(errno, std::generic_category()));
      b = static_cast<void*>(static_cast<char*>(m) + align);
    }

    ~slice() {
      if (m != MAP_FAILED)
        munmap(m, len);
    }

    template <typename T>
    T* begin(size_t idx) {
      return static_cast<T*>(b) + idx;
    }
    template <typename T>
    T* end() {
      return reinterpret_cast<T*>(static_cast<char*>(m) + len);
    }
  };

  int fd;
  uint64_t numNodes;
  uint64_t numEdges;

  slice outIdx;
  size_t outIdxBegin;
  slice outs;
  size_t outsBegin;

public:
  FileGraph(const FileGraph&) = delete;
  FileGraph& operator=(const FileGraph&) = delete;

  explicit FileGraph(const std::string& f) {
    fd = open(f.c_str(), O_RDONLY);
    if (fd == -1)
      throw std::system_error(std::error_code(errno, std::generic_category()));
    slice s;
    s.read(0, 4 * sizeof(uint64_t), fd);
    uint64_t* ptr = s.begin<uint64_t>(0);
    numNodes      = ptr[2];
    numEdges      = ptr[3];
  }

  ~FileGraph() { close(fd); }

  void read(int rank, int total) {
    size_t bsize = (numNodes + total - 1) / total;
    outIdxBegin  = rank == 0 ? 0 : (rank * bsize) - 1;
    size_t outIdxEnd =
        std::min((rank + 1) * bsize, static_cast<size_t>(numNodes));
    size_t len = outIdxEnd - outIdxBegin;
    outIdx.read(4 * sizeof(uint64_t) + outIdxBegin * sizeof(uint64_t),
                len * sizeof(uint64_t), fd);
    uint64_t* pOutIdx = outIdx.begin<uint64_t>(0);

    if (!len)
      return;

    outsBegin      = rank == 0 ? 0 : pOutIdx[0];
    size_t outsEnd = pOutIdx[len - 1];
    size_t nedges  = outsEnd - outsBegin;
    size_t base    = 4 * sizeof(uint64_t) + numNodes * sizeof(uint64_t) +
                  outsBegin * sizeof(uint32_t);
    outs.read(base, nedges * sizeof(uint32_t), fd);
  }

  void putMine(int rank, int total, MPI_Win window) {
    size_t offset1 = rank == 0 ? 0 : (outIdxBegin + 1) * sizeof(uint64_t);
    uint64_t* source1 =
        rank == 0 ? outIdx.begin<uint64_t>(0) : outIdx.begin<uint64_t>(1);
    size_t len1 =
        std::distance(source1, outIdx.end<uint64_t>()) * sizeof(uint64_t);
    MPI_Put(source1, len1, MPI_BYTE, 0, offset1, len1, MPI_BYTE, window);

    size_t offset2 = numNodes * sizeof(uint64_t) + outsBegin * sizeof(uint32_t);
    uint32_t* source2 = outs.begin<uint32_t>(0);
    size_t len2 =
        std::distance(source2, outs.end<uint32_t>()) * sizeof(uint32_t);
    MPI_Put(source2, len2, MPI_BYTE, 0, offset2, len2, MPI_BYTE, window);
  }

  size_t numBytes() {
    return numNodes * sizeof(uint64_t) + numEdges * sizeof(uint32_t);
  }

  void verify(void* base) {
    slice s;
    s.read(4 * sizeof(uint64_t), numBytes(), fd);
    int r = memcmp(base, s.begin<char>(0), numBytes());

    if (r == 0) {
      std::cout << "Verified\n";
    } else {
      throw std::logic_error("not perfect copy");
    }
  }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    printHelp(std::cout);
    exit(1);
  }

  MPI_Init(&argc, &argv);

  int rank;
  int total;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &total);

  FileGraph g(argv[1]);

  g.read(rank, total);

  void* base  = 0;
  size_t size = 0;
  if (rank == 0) {
    size = g.numBytes();
    MPI_Alloc_mem(size, MPI_INFO_NULL, &base);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  MPI_Win window;
  MPI_Win_create(base, size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &window);
  MPI_Win_fence(0, window);
  g.putMine(rank, total, window);
  MPI_Win_fence(0, window);

  double t2 = MPI_Wtime();
  if (rank == 0) {
    std::cout << "Gather time: " << (t2 - t1) << "\n";
    g.verify(base);
  }

  MPI_Win_free(&window);
  MPI_Finalize();

  if (base)
    free(base);

  return 0;
}
