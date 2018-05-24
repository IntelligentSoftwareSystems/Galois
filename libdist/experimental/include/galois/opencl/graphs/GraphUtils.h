#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <cassert>
#ifdef _WIN32
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#else
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include "galois/opencl/CL_Util.h"

#ifndef GRAPHUTILS_H_
#define GRAPHUTILS_H_

#ifdef _WIN32
#  define le64toh(x) (x) // OSSwapLittleToHostInt64(x)
#  define le32toh(x) (x) //  OSSwapLittleToHostInt32(x)
#endif

#ifdef __APPLE__
#include <libkern/OSByteOrder.h>
#  define le64toh(x) (x) // OSSwapLittleToHostInt64(x)
#  define le32toh(x) (x) //  OSSwapLittleToHostInt32(x)
#elif __FreeBSD__
#  include <sys/endian.h>
#elif __linux__
#  include <endian.h>
#  ifndef le64toh
#    if __BYTE_ORDER == __LITTLE_ENDIAN
#      define le64toh(x) (x)
#      define le32toh(x) (x)
#    else
#      define le64toh(x) __bswap_64 (x)
#      define le32toh(x) __bswap_32 (x)
#    endif
#  endif
#endif

#ifdef _WIN32
/********************************************************************
 *
 *********************************************************************/
template<typename GraphType>
unsigned inline readFromGR(GraphType & g, const char * file) {
   DEBUG_CODE(std::cout<<"Reading binary .gr file " << file << "\n";);
   int filebuf = _open(file, _O_BINARY | _O_RDONLY);

   //int masterFD = _open(file, O_RDONLY);
   if (filebuf ==-1) {
      printf("FileGraph::structureFromFile: unable to open %s.\n", file);
      abort();
   }
   struct stat buf;
   int f = fstat(filebuf, &buf);
   if (f == -1) {
      printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
      abort();
   }
   size_t masterLength = buf.st_size;

   //int _MAP_BASE = MAP_PRIVATE;
   void* m = malloc(masterLength*sizeof(char));
   //mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
   size_t size_read = _read(filebuf,m,masterLength);
   if (size_read==0 ) {
      m = 0;
      printf("FileGraph::structureFromFile: mmap failed.\n");
      abort();
   }

   //parse file
   uint64_t* fptr = (uint64_t*) m;
   uint64_t version = le64toh(*fptr++);
   assert(version == 1);
   uint64_t sizeEdgeTy = le64toh(*fptr++);
   uint64_t numNodes = le64toh(*fptr++);
   uint64_t numEdges = le64toh(*fptr++);
   uint64_t *outIdx = fptr;
   fptr += numNodes;
   uint32_t *fptr32 = (uint32_t*) fptr;
   uint32_t *outs = fptr32;
   fptr32 += numEdges;
   if (numEdges % 2)
   fptr32 += 1;
   unsigned *edgeData = (unsigned *) fptr32;

   g._num_nodes = numNodes;
   g._num_edges = numEdges;
   DEBUG_CODE(std::cout<<"Sizes read from file :: Edge (" << sizeEdgeTy << ")\n";);
   g.init(g._num_nodes, g._num_edges);
   //node_data
   memset(g.node_data(), 0, sizeof(unsigned int) * g._num_nodes);
   for (unsigned int i = 0; i < g._num_edges; ++i) {
      g.out_neighbors()[i] = *le64toh(outs+i);
   }
   g.outgoing_index()[0] = 0;
   for (unsigned int i = 0; i < g._num_nodes; ++i) {
      g.outgoing_index()[i + 1] = *le64toh(outIdx+i);
   }
   unsigned int start = 0;
   unsigned int displacement = 0;

   for (unsigned int i = 0; i < g._num_nodes; ++i) {
      unsigned int end = *le64toh(outIdx+i);
      for (unsigned int idx = start; idx < end; ++idx) {
         //node i's idx neighbor is to be populated here.
         g.out_edge_data()[displacement] = *le64toh(edgeData+idx);
         g.out_neighbors()[displacement] = *le64toh(outs+idx);
         displacement++;
      }
      start = end;
   }
   for (size_t i = 0; i < g._num_nodes; ++i)
   g.node_data()[i] = std::numeric_limits<unsigned int>::max() / 2;
   g.update_in_neighbors();
   return 0;
}

#else
/********************************************************************
 *
 *********************************************************************/
template<typename GraphType>
unsigned inline readFromGR(GraphType & g, const char * file) {
   DEBUG_CODE(std::cout<<"Reading binary .gr file " << file << "\n";);
   std::ifstream cfile;
   cfile.open(file);

   // copied from GaloisCpp/trunk/src/FileGraph.h
   int masterFD = open(file, O_RDONLY);
   if (masterFD == -1) {
      printf("FileGraph::structureFromFile: unable to open %s.\n", file);
      abort();
   }

   struct stat buf;
   int f = fstat(masterFD, &buf);
   if (f == -1) {
      printf("FileGraph::structureFromFile: unable to stat %s.\n", file);
      abort();
   }
   size_t masterLength = buf.st_size;

   int _MAP_BASE = MAP_PRIVATE;
//#ifdef MAP_POPULATE
//  _MAP_BASE  |= MAP_POPULATE;
//#endif

   void* m = mmap(0, masterLength, PROT_READ, _MAP_BASE, masterFD, 0);
   if (m == MAP_FAILED ) {
      m = 0;
      printf("FileGraph::structureFromFile: mmap failed.\n");
      abort();
   }

   //parse file
   uint64_t* fptr = (uint64_t*) m;
   __attribute__((unused))      uint64_t version = le64toh(*fptr++);
   assert(version == 1);
   __attribute__((unused))    uint64_t sizeEdgeTy = le64toh(*fptr++);
   uint64_t numNodes = le64toh(*fptr++);
   uint64_t numEdges = le64toh(*fptr++);
   uint64_t *outIdx = fptr;
   fptr += numNodes;
   uint32_t *fptr32 = (uint32_t*) fptr;
   uint32_t *outs = fptr32;
   fptr32 += numEdges;
   if (numEdges % 2)
      fptr32 += 1;
   unsigned *edgeData = (unsigned *) fptr32;

   g._num_nodes = numNodes;
   g._num_edges = numEdges;
   DEBUG_CODE(std::cout<<"Sizes read from file :: Edge (" << sizeEdgeTy << ")\n";);
   g.init(g._num_nodes, g._num_edges);
   //node_data
   memset(g.node_data(), 0, sizeof(unsigned int) * g._num_nodes);
   for (unsigned int i = 0; i < g._num_edges; ++i) {
      g.out_neighbors()[i] = *le64toh(outs+i);
   }
   g.outgoing_index()[0] = 0;
   for (unsigned int i = 0; i < g._num_nodes; ++i) {
      g.outgoing_index()[i + 1] = *le64toh(outIdx+i);
   }
   unsigned int start = 0;
   unsigned int displacement = 0;
   for (unsigned int i = 0; i < g._num_nodes; ++i) {
      unsigned int end = *le64toh(outIdx+i);
      for (unsigned int idx = start; idx < end; ++idx) {
         //node i's idx neighbor is to be populated here.
         g.out_edge_data()[displacement] = *le64toh(edgeData+idx);
         g.out_neighbors()[displacement] = *le64toh(outs+idx);
         displacement++;
      }
      start = end;
   }
/*   for (size_t i = 0; i < g._num_nodes; ++i)
      g.node_data()[i] = std::numeric_limits<unsigned int>::max() / 2;*/
   cfile.close();
   g.update_in_neighbors();
   return 0;
}
#endif
/********************************************************************
 *
 *********************************************************************/
template<typename GraphTy>
void check_graph(GraphTy & graph) {
   double node_data_sum = 0;
   double edge_data_sum = 0;
   unsigned int max_edge = 0;
   size_t sink_nodes = 0; //no outgoing edge
   size_t max_degree = 0;
   size_t one_degree = 0;
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      node_data_sum += graph.node_data()[i];
      if (graph.num_neighbors(i) == 0)
         sink_nodes++;
      if (graph.num_neighbors(i) == 1)
         one_degree++;
      max_degree = std::max(max_degree, (size_t)graph.num_neighbors(i));
   }
   for (size_t i = 0; i < graph.num_edges(); ++i) {
      edge_data_sum += graph.out_edge_data()[i];
      max_edge = std::max(max_edge, graph.out_edge_data()[i]);
   }
   std::cout << "Nodes:" << graph.num_nodes() << ", Sinks:" << sink_nodes << ", Edges:" << graph.num_edges() << ", 1-degree:" << one_degree << ", Max-degree:" << max_degree
         << "\n";
   std::cout << "Node data sum : " << node_data_sum << " \nEdge data sum : " << edge_data_sum << " Max weight : " << max_edge << "\n";
   return;
} //End check_graph!
template<typename GraphType>
void graph_stat(const char * p_filename) {
   GraphType graph;
   graph.read(p_filename);
   check_graph(graph);
}
/********************************************************************
 *
 *********************************************************************/
template<typename GraphTy>
void edge_distribution(std::string filename, GraphTy & g) {
   int max_degree = 0;
   for (unsigned int i = 0; i < g.num_nodes(); ++i) {
      max_degree = std::max(max_degree, (int) g.num_neighbors(i));
   }
   max_degree += 1;
   std::vector<unsigned int> edge_count(max_degree);
   for (size_t i = 0; i < g.num_nodes(); ++i) {
      edge_count[g.num_neighbors(i)]++;
   }
   std::ofstream out_file(filename);
   for (int i = 0; i < max_degree; ++i) {
      out_file << i << ", " << edge_count[i] << "\n";
   }
   out_file.close();
}
/********************************************************************
 *
 *********************************************************************/
template<typename GraphTy>
void check_duplicate_edges(GraphTy & g) {
   int max_wt = 0;
   for (unsigned int j = g.outgoing_index()[0]; j < g.outgoing_index()[g.num_nodes()]; ++j) {
      max_wt = std::max(max_wt, (int) g.out_edge_data()[j]);
   }
   std::vector<int> edge_wt(max_wt + 1, 0);
   unsigned int fail_counter = 0;
   typename GraphTy::EdgeDataType max_duplicate_wt;
   int max_duplicate_counter = 0;
   for (unsigned int i = 0; i < g.num_nodes(); ++i) {
      unsigned int src = i;
      for (unsigned int dst_idx = g.outgoing_index()[src]; dst_idx < g.outgoing_index()[src + 1]; ++dst_idx) {
         unsigned int dst = g.out_neighbors()[dst_idx];
         unsigned int wt = g.out_edge_data()[dst_idx];
         edge_wt[wt] += 1;
         if (edge_wt[wt] > max_duplicate_counter) {
            max_duplicate_counter = edge_wt[wt];
            max_duplicate_wt = wt;
         }
         bool found = false;
         for (unsigned int rev_idx = g.outgoing_index()[dst]; rev_idx < g.outgoing_index()[dst + 1]; ++rev_idx) {
            if (src == g.out_neighbors()[rev_idx]) {
               found = true;
               break;
            }
         }
         if (found == false)
            fail_counter++;
      }
   }
   int max_duplicates = 0;
   for (auto it = edge_wt.begin(); it != edge_wt.end(); ++it) {
      max_duplicates = std::max(max_duplicates, *it);
   }
   if (max_duplicates != 0 || fail_counter != 0) {
      std::cout << "Max edge_wt duplication : " << max_duplicates << "\tFailed asymmetry : " << fail_counter << ", max_wt:: " << max_wt << ", " << log2(max_wt) << "-bits, ";
      std::cout << "Max duplications :: " << max_duplicate_counter << "\tfor wt:: " << max_duplicate_wt << ", req::" << log2(max_duplicate_counter) << "-bits\n";
      std::cout << "#INFO#:: Total bits required:: " << (ceil(log2(max_wt)) + ceil(log2(max_duplicate_counter))) << "\n";
   }
}

/********************************************************************
 *
 *********************************************************************/
template<typename GraphTy>
GraphTy * create_symmetric(GraphTy & graph) {
   typedef std::pair<int, int> EdgeType;
   typedef typename std::vector<EdgeType> EdgeList;
   std::vector<EdgeList *> edges;
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      edges.push_back(new EdgeList());
   }
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      EdgeList & curr_list = *edges[i];
      for (unsigned int e = graph.outgoing_index()[i]; e < graph.outgoing_index()[i + 1]; ++e) {
         EdgeType edge;
         edge.first = graph.out_neighbors()[e];
         edge.second = graph.out_edge_data()[e];
         assert(edge.first < (int )(graph.num_nodes()));
         {
            bool found = false;
            for (size_t j = 0; j < curr_list.size(); ++j) {
               if (curr_list[j].first == edge.first) {
                  curr_list[j].second = std::min(curr_list[j].second, edge.second);
                  found = true;
               }
            }
            if (found == false)
               curr_list.push_back(edge);
         }
         {
            EdgeList &rev_list = *edges[edge.first];
            EdgeType rev_edge;
            rev_edge.first = i;
            rev_edge.second = edge.second;
            bool found = false;
            for (size_t j = 0; j < rev_list.size(); ++j) {
               if (rev_list[j].first == rev_edge.first) {
                  rev_list[j].second = std::min(rev_list[j].second, rev_edge.second);
                  found = true;
               }
            }
            if (found == false)
               rev_list.push_back(rev_edge);
         }
      } //End for-edges

   } //End for-nodes
     /////////Now create next graph;
   int total_edges = 0;
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      total_edges += edges[i]->size();
   }
   GraphTy * next_graph = new GraphTy();
   std::cout << "Creating symmetric graph: " << graph.num_nodes() << " nodes , " << total_edges << " edges. \n";
   next_graph->init(graph.num_nodes(), total_edges);
   int run_counter = 0;
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      next_graph->outgoing_index()[i] = run_counter;
      EdgeList & curr_list = *edges[i];
      for (unsigned int j = 0; j < edges[i]->size(); ++j) {
         next_graph->out_neighbors()[j + run_counter] = curr_list[j].first;
         next_graph->out_edge_data()[j + run_counter] = curr_list[j].second;
      }
      run_counter += curr_list.size();
   }
   next_graph->outgoing_index()[graph.num_nodes()] = run_counter;
   for (unsigned int i = 0; i < graph.num_nodes(); ++i) {
      delete edges[i];
   }
   return next_graph;
}
/********************************************************************
 *
 *********************************************************************/
#endif /* GRAPHUTILS_H_ */
