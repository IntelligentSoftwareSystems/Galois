/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/*
 * RandomPartitioner.h
 *
 *  Created on: Jun 15, 2016
 *      Author: rashid
 */

#ifndef GDIST_PARTITIONER_GREEDY_BALANCED_PARTITIONER_DISK_H_
#define GDIST_PARTITIONER_GREEDY_BALANCED_PARTITIONER_DISK_H_
#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>
#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <random>

#include "galois/graphs/FileGraph.h"
#include "galois/graphs/OfflineGraph.h"
#include <sys/mman.h>
#include <mcheck.h>

/******************************************************************
 *
 *****************************************************************/

struct GBPartitionerDisk {
  typedef short PartitionIDType;
  struct NewEdgeData {
    size_t src, dst;
#if _HAS_EDGE_DATA
    EdgeDataType data;
#endif

#if _HAS_EDGE_DATA
    NewEdgeData(size_t s, size_t d, EdgeDataType dt)
        : src(s), dst(d), data(dt) {}
#else

#endif
    NewEdgeData(size_t s, size_t d) : src(s), dst(d) {}
    NewEdgeData() : src(~0), dst(~0) {}
  };
  struct VertexCutInfo {
    // Per host data structures - small size
    std::vector<size_t> edgesPerHost;
    std::vector<size_t> verticesPerHost;
    std::vector<size_t> mastersPerHost;

    // Per vertex data structure - large size
    //      std::vector<std::unordered_set<PartitionIDType>> vertexOwnersPacked;
    std::vector<std::vector<bool>>* vertexOwnersPacked;
    std::vector<PartitionIDType> vertexMasters;
    std::vector<std::unordered_map<size_t, size_t>> global2Local;

    std::vector<std::FILE*> tmpPartitionFiles;
    std::vector<std::FILE*> tmpGraphFiles;
    std::vector<std::string> tmpPartitionFiles_names;
    std::vector<std::string> tmpGraphFiles_names;

    void print_size_stats(size_t num_hosts) {
      long long total_size = 0;
      //         total_size+=vertexOwners[0].size()*num_hosts +
      //         vertexOwnersPacked[0].size()*num_hosts + vertexMasters.size() +
      //         global2Local[0].size()*num_hosts;
      std::cout << "Size == "
                << total_size * sizeof(size_t) / (1024 * 1024 * 1024.0f)
                << " GB\n";
      return;
    }
    void init(size_t nn, size_t ne, size_t numHosts) {
      global2Local.resize(numHosts);
      vertexOwnersPacked = new std::vector<std::vector<bool>>(nn);
      for (auto& n : *vertexOwnersPacked) {
        n.resize(numHosts, false);
      }
      char tmpBuffer[L_tmpnam];
      for (size_t h = 0; h < numHosts; ++h) {
        tmpnam(tmpBuffer);
        std::string tmpfileName = "/net/ohm/export/cdgc/rashid/" +
                                  std::string(tmpBuffer) + ".P." +
                                  std::to_string(h);
        tmpPartitionFiles_names.push_back(tmpfileName);
        std::FILE* file_handle = std::fopen(tmpfileName.c_str(), "wb+x");
        if (!file_handle) {
          std::cerr << "Unable to open tmp file :: " << tmpfileName.c_str()
                    << "\n";
          exit(-1);
        }
        tmpPartitionFiles.push_back(file_handle);
        std::cout << "Pushed file : " << tmpfileName << "\n";
        {
          tmpnam(tmpBuffer);
          std::string tmpGraphfileName = "/net/ohm/export/cdgc/rashid/" +
                                         std::string(tmpBuffer) + ".G." +
                                         std::to_string(h);
          tmpGraphFiles_names.push_back(tmpGraphfileName);
          std::FILE* file_handle = std::fopen(tmpGraphfileName.c_str(), "wb+x");
          if (!file_handle) {
            std::cerr << "Unable to open tmp file :: "
                      << tmpGraphfileName.c_str() << "\n";
            exit(-1);
          }
          tmpGraphFiles.push_back(file_handle);
        }
        // tmpPartitionFiles.push_back(std::tmpfile());
        //            global2Local[h].resize(nn, ~0);
      }
      std::cout << "Init : Loop 1 \n";

      //         vertexMasters.resize(nn, ~0);
      mastersPerHost.resize(numHosts, 0);
      std::cout << "Init : Loop 2 \n";
      edgesPerHost.resize(numHosts, 0);
      verticesPerHost.resize(numHosts, 0);
      std::cout << "Done Init \n";
    }
    static void writeEdgeToTempFile(std::FILE* f, size_t src, size_t dst) {
      NewEdgeData ed(src, dst);
      fwrite(&ed, sizeof(NewEdgeData), 1, f);
    }
    template <typename EType>
    static void writeEdgeToTempFile(std::FILE* f, size_t src, size_t dst,
                                    EType e) {
      NewEdgeData ed(src, dst, e);
      fwrite(&ed, sizeof(NewEdgeData), 1, f);
    }
    static bool readEdgeFromTempFile(std::FILE* f, size_t& src, size_t& dst) {
      NewEdgeData ed;
      fread(&ed, sizeof(NewEdgeData), 1, f);
      src = ed.src;
      dst = ed.dst;
      return feof(f);
    }
    template <typename EType>
    static bool readEdgeFromTempFile(std::FILE* f, size_t& src, size_t& dst,
                                     EType& e) {
      NewEdgeData ed;
      fread(&ed, sizeof(NewEdgeData), 1, f);
      src = ed.src;
      dst = ed.dst;
      e   = ed.data;
      return feof(f);
    }

    /*
     *
     * */
    void assignEdge(galois::graphs::OfflineGraph& g, NodeItType& _src,
                    galois::graphs::OfflineGraph::GraphNode& dst, size_t& eIdx,
                    EdgeItType& e, PartitionIDType owner) {
      auto src = *_src;
      edgesPerHost[owner]++;
      auto& _vertexOwnersPacked       = *vertexOwnersPacked;
      _vertexOwnersPacked[src][owner] = 1;
      _vertexOwnersPacked[dst][owner] = 1;
#if _HAS_EDGE_DATA
      //         writeEdgeToTempFile(tmpPartitionFiles[owner],new_src, new_dst,
      //         g.getEdgeData<EdgeDataType>(e));
      writeEdgeToTempFile(tmpPartitionFiles[owner], src, dst,
                          g.getEdgeData<EdgeDataType>(e));
#else
      //         writeEdgeToTempFile(tmpPartitionFiles[owner], new_src,
      //         new_dst);
      writeEdgeToTempFile(tmpPartitionFiles[owner], src, dst);
#endif
    }
    /*
     * Go over all the nodes, and check if it has an edge
     * in a given host - if yes, assign a local-id to it.
     * This will ensure that localids and globalids are sequential
     * */

    void assignLocalIDs(size_t numhost, galois::graphs::OfflineGraph& g) {
      auto& _vertexOwnersPacked = *vertexOwnersPacked;
#if 1 // Currently some error in stats - use threaded code after it is fixed.
      std::vector<size_t> hosts;
      for (size_t i = 0; i < numhost; ++i)
        hosts.push_back(i);
      galois::do_all(
          hosts.begin(), hosts.end(),
          [&](size_t h) {
            for (size_t n = 0; n < g.size(); ++n) {
              if (_vertexOwnersPacked[n][h]) {
                global2Local[h][n] = verticesPerHost[h]++;
              }
            }
          },
          galois::loopname("localIDs"));
#else
      for (size_t h = 0; h < numhost; ++h) {
        for (size_t n = 0; n < g.size(); ++n) {
          if (global2Local[h][n] == 1) {
            global2Local[h][n] = verticesPerHost[h]++;
          }
        }
      }
#endif
    }

    void writeReplicaInfo(std::string& basename,
                          galois::graphs::OfflineGraph& g, size_t numhosts) {
#if 0
         this->print_size_stats(numhosts);
         for (size_t n = 0; n < g.size(); ++n) {
            for (size_t h = 0; h < numhosts; ++h) {
               if (vertexOwners[n][h]) {
                  vertexOwnersPacked[n].insert(h);
               }
            }
            vertexOwners[n].clear();
         }
         vertexOwners.clear();
         std::cout<<"VertexOwners "<< vertexOwners.size()<<"\n";
         this->print_size_stats(numhosts);
         std::ofstream replica_file(getReplicaInfoFileName(basename, numhosts));
         auto numEntries = g.size();
         //         replica_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         replica_file << numEntries << ", " << numhosts << std::endl;
         for (size_t n = 0; n < g.size(); ++n) {
//            auto owner = &vertexOwnersPacked[n];
            size_t num_replicas = vertexOwnersPacked[n].size();
            //            replica_file.write(reinterpret_cast<const char*>(&num_replicas), sizeof(size_t));
            replica_file << num_replicas << ", " << std::distance(g.edge_begin(n), g.edge_end(n)) << std::endl;
         }
         replica_file.close();
#endif
    }
    /*
     * The assignment of masters to each vertex is done in a greedy manner -
     * the list of hosts with a copy of each vertex is scanned, and the one with
     * smallest number of masters is selected to be the master of the current
     * node, and the masters-count for the host is updated.
     * */
    void assignMasters(size_t nn, size_t numhost,
                       galois::graphs::OfflineGraph& g) {
      auto& _vertexOwnersPacked = *vertexOwnersPacked;

      vertexMasters.resize(nn, ~0);
      for (size_t n = 0; n < nn; ++n) {
        if (vertexMasters[n] != ~0) {
          std::cout << "ERRR " << vertexMasters[n] << " Not eq " << ~0 << "\n";
        }
        assert(vertexMasters[n] == ~0);
#if 1
        { // begin change
          size_t minID     = ~0;
          size_t min_count = mastersPerHost[0];

          for (size_t h = 0; h < numhost; ++h) {
            if (_vertexOwnersPacked[n][h]) {
              if (minID == ~0) {
                minID     = h;
                min_count = mastersPerHost[minID];
              } else {
                if (min_count > mastersPerHost[h]) { // found something smaller!
                  minID     = h;
                  min_count = mastersPerHost[h];
                }
              }
            }
          }

          // Vertex does not have any edges - pick the least loaded partition
          // anyway!
          if (minID == ~0) {
            minID     = 0;
            min_count = mastersPerHost[minID];
            for (size_t h = 1; h < numhost; ++h) {
              if (min_count > mastersPerHost[h]) {
                min_count = mastersPerHost[h];
                minID     = h;
              }
            }
          }
          assert(minID != ~0);
          vertexMasters[n] = minID;
          mastersPerHost[minID]++;
          _vertexOwnersPacked[n].clear();
        } // end change
#else
        if (vertexOwnersPacked[n].size() == 0) {
          size_t minID     = 0;
          size_t min_count = mastersPerHost[minID];
          for (size_t h = 1; h < numhost;
               ++h) { /*Note - 0 is default host, so start at 1*/
            if (min_count > mastersPerHost[h]) {
              min_count = mastersPerHost[h];
              minID     = h;
            } // end if
          }   // end for
          vertexMasters[n] = minID;
          mastersPerHost[minID]++;
          // std::cout<<"No edges for "<< n <<" ,
          // "<<std::distance(g.edge_begin(n), g.edge_end(n))<<std::endl;
        } else {
          assert(vertexOwnersPacked[n].size() > 0);
          size_t minID     = *vertexOwnersPacked[n].begin();
          size_t min_count = mastersPerHost[minID];
          for (auto host : vertexOwnersPacked[n]) {
            if (mastersPerHost[host] < min_count) {
              min_count = mastersPerHost[host];
              minID     = host;
            } // end if
          }   // end for
          assert(minID != ~0);
          vertexMasters[n] = minID;
          mastersPerHost[minID]++;
        } // end else
        _vertexOwnersPacked[n].clear();
#endif
      } // end for
      _vertexOwnersPacked.clear();
      _vertexOwnersPacked.shrink_to_fit();
      delete vertexOwnersPacked;
    } // end assignMasters
    /////////////////////////////////////////////////////
    void print_stats() {
      for (size_t i = 0; i < mastersPerHost.size(); ++i) {
        std::cout << "Masters " << i << ":: " << mastersPerHost[i] << std::endl;
      }
      for (size_t i = 0; i < edgesPerHost.size(); ++i) {
        std::cout << "Edges " << i << ":: " << edgesPerHost[i] << std::endl;
      }
    }
    ~VertexCutInfo() {
      std::cout << "Cleaning up....\n";
      for (auto F : tmpPartitionFiles) {
        std::fclose(F);
      }
      for (auto F : tmpGraphFiles) {
        std::fclose(F);
      }
      for (auto N : tmpPartitionFiles_names) {
        std::remove(N.c_str());
      }
      for (auto N : tmpGraphFiles_names) {
        std::remove(N.c_str());
      }
      std::cout << "Done cleaning up....\n";
    }
  };
  /******************************************************************
   *
   *****************************************************************/
  VertexCutInfo vcInfo;
  /*
   * Overload this method for different implementations of the partitioning.
   * */
  PartitionIDType getEdgeOwner_old(size_t src, size_t dst,
                                   PartitionIDType num) {
    return rand() % num;
  }
  PartitionIDType getEdgeOwner(size_t src, size_t dst, PartitionIDType num) {
    static std::mt19937 gen;
    static std::uniform_int_distribution<PartitionIDType> dist(0, num - 1);
    return dist(gen);
  }
  /*
   * Partitioning routine.
   * */
  void operator()(std::string& basename, galois::graphs::OfflineGraph& g,
                  size_t num_hosts) {
    galois::Timer T_edge_assign, T_write_replica, T_assign_masters,
        T_write_partition, T_total, T_assign_localIDs;

    std::cout << "Partitioning: |V|= " << g.size()
              << " , |E|= " << g.sizeEdges() << " |P|= " << num_hosts << "\n";
    //      mtrace();
    T_total.start();
    T_edge_assign.start();
    vcInfo.init(g.size(), g.sizeEdges(), num_hosts);
    auto prev_nbr_end   = g.edge_begin(*g.begin());
    auto curr_nbr_end   = g.edge_end(*g.begin());
    size_t edge_counter = 0;
    for (auto n = g.begin(); n != g.end(); ++n) {
      auto src     = *n;
      curr_nbr_end = g.edge_end(*n);
      for (auto nbr = prev_nbr_end; nbr != curr_nbr_end;
           ++nbr, ++edge_counter) {
        auto dst              = g.getEdgeDst(nbr);
        PartitionIDType owner = getEdgeOwner(src, dst, num_hosts);
        //            void assignEdge(OfflineGraph & g, NodeItType & _src,
        //            NodeItType & dst, size_t & eIdx, EdgeItType & e, size_t
        //            owner) {
        vcInfo.assignEdge(g, n, dst, edge_counter, nbr, owner);
      }
      prev_nbr_end = curr_nbr_end;
    }
    T_edge_assign.stop();
    std::cout << "STEP#1-EdgesAssign:: " << T_edge_assign.get() << "\n";
    for (size_t i = 0; i < num_hosts; ++i) {
      std::cout << "HOST#:: " << i << " , " << vcInfo.verticesPerHost[i] << ", "
                << vcInfo.edgesPerHost[i] << "\n";
    }
    std::cout << "Assigning local ids\n";
    T_assign_localIDs.start();
    vcInfo.assignLocalIDs(num_hosts, g);
    T_assign_localIDs.stop();
    std::cout << "Writing replica info \n";
    T_write_replica.start();
    //      vcInfo.writeReplicaInfo(basename, g, num_hosts);
    T_write_replica.stop();
    std::cout << "Assigning masters\n";
    T_assign_masters.start();
    vcInfo.assignMasters(g.size(), num_hosts, g);
    T_assign_masters.stop();
    std::cout << "Writing partitions\n";
    T_write_partition.start();
    writePartitionsMem(basename, g, num_hosts);
    T_write_partition.stop();

    T_total.stop();
    std::cout
        << "STAT,EdgeAssig, WriteReplica,AssignMasters,WritePartition, Total\n";
    std::cout << num_hosts << "," << T_edge_assign.get() << ","
              << T_write_replica.get() << "," << T_assign_masters.get() << ","
              << T_write_partition.get() << "," << T_total.get() << "\n";
    //      muntrace();
  }

  /*
   * Optimized implementation for memory usage.
   * Write both the metadata as well as the partition information.
   * */

  void writePartitionsMem(std::string& basename,
                          galois::graphs::OfflineGraph& g, size_t num_hosts) {
    // Create graph
    // TODO RK - Fix edgeData
    std::cout << " Low mem version\n";
    //      std::vector<size_t> &vertexOwners = vcInfo.vertexMasters;
    for (size_t h = 0; h < num_hosts; ++h) {
      std::cout << "Building partition " << h << "...\n";
      std::cout << "Analysis :: " << vcInfo.verticesPerHost[h] << " , "
                << vcInfo.edgesPerHost[h] << "\n";
      using namespace galois::graphs;
      FileGraphWriter* ng       = new FileGraphWriter();
      FileGraphWriter& newGraph = *ng;
      newGraph.setNumNodes(vcInfo.verticesPerHost[h]);
      newGraph.setNumEdges(vcInfo.edgesPerHost[h]);
#if _HAS_EDGE_DATA
      newGraph.setSizeofEdgeData(sizeof(EdgeDataType));
#endif
      newGraph.phase1();
      std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
      std::cout << "Writing meta-file " << h << " to disk..." << meta_file_name
                << "\n";
      std::ofstream meta_file(meta_file_name, std::ofstream::binary);
      auto numEntries = vcInfo.verticesPerHost[h];
      meta_file.write(reinterpret_cast<char*>(&(numEntries)),
                      sizeof(numEntries));
      for (auto entry : vcInfo.global2Local[h]) {
        size_t n     = entry.first;
        size_t owner = vcInfo.vertexMasters[n];
        meta_file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        meta_file.write(reinterpret_cast<const char*>(&entry.second),
                        sizeof(size_t));
        meta_file.write(reinterpret_cast<const char*>(&owner), sizeof(owner));
      }
      meta_file.close();
      {
        rewind(vcInfo.tmpPartitionFiles[h]);
        size_t s, d;
#if _HAS_EDGE_DATA
        EdgeDataType ed;
        while (!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d,
                                            ed)) {
          vcInfo.writeEdgeToTempFile(vcInfo.tmpGraphFiles[h],
                                     vcInfo.global2Local[h][s],
                                     vcInfo.global2Local[h][d], ed);
          newGraph.incrementDegree(vcInfo.global2Local[h][s]);
          //               std::cout<<i++<<", "<<newEdges.back().src << " , " <<
          //               newEdges.back().dst << std::endl;
        }
#else
        while (
            !vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d)) {
          vcInfo.writeEdgeToTempFile(vcInfo.tmpGraphFiles[h],
                                     vcInfo.global2Local[h][s],
                                     vcInfo.global2Local[h][d]);
          newGraph.incrementDegree(vcInfo.global2Local[h][s]);
        }
#endif
      }
      vcInfo.global2Local[h].clear();
      //         assert(newEdges.size() == vcInfo.edgesPerHost[h]);
      newGraph.phase2();
      rewind(vcInfo.tmpGraphFiles[h]);
#if _HAS_EDGE_DATA
      std::vector<EdgeDataType> newEdgeData(vcInfo.edgesPerHost[h]);
      size_t s, d;
      EdgeDataType ed;
      while (!vcInfo.readEdgeFromTempFile(vcInfo.tmpGraphFiles[h], s, d, ed)) {
        size_t idx       = newGraph.addNeighbor(s, d);
        newEdgeData[idx] = ed;
      }
      EdgeDataType* edgeDataPtr = newGraph.finish<EdgeDataType>();
      memcpy(edgeDataPtr, newEdgeData.data(),
             sizeof(EdgeDataType) * newEdgeData.size());
      newEdgeData.clear();
#else
      size_t s, d;
      while (!vcInfo.readEdgeFromTempFile(vcInfo.tmpGraphFiles[h], s, d)) {
        size_t idx = newGraph.addNeighbor(s, d);
      }
      newGraph.finish<void>();
#endif

      std::string gFileName = getPartitionFileName(basename, h, num_hosts);
      std::cout << "Writing partition " << h << " to disk... " << gFileName
                << "\n";
      newGraph.toFile(gFileName);
      delete ng;
    } // End for-hosts
    vcInfo.print_stats();
  } // end writePartitionsMem method
};
#endif /* GDIST_PARTITIONER_GREEDY_BALANCED_PARTITIONER_DISK_H_ */
