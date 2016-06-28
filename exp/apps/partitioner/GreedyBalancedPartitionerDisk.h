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
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>
#include <cstdio>

#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/OfflineGraph.h"

/******************************************************************
 *
 *****************************************************************/

struct GBPartitionerDisk {
   struct NewEdgeData {
      size_t src, dst;
#if _HAS_EDGE_DATA
      EdgeDataType data;
#endif

#if _HAS_EDGE_DATA
      NewEdgeData(size_t s, size_t d, EdgeDataType dt ):src(s), dst(d),data(dt) {}
#else

#endif
      NewEdgeData(size_t s, size_t d) :
            src(s), dst(d) {
      }
      NewEdgeData() :
            src(-1), dst(-1) {
      }
   };
   struct VertexCutInfo {
//      std::vector<size_t> edgeOwners;
      std::vector<size_t> edgesPerHost;
      std::vector<size_t> verticesPerHost;
      std::vector<std::set<size_t>> vertexOwnersPacked;
      std::vector<std::vector<bool>> vertexOwners;
      std::vector<size_t> vertexMasters;
      std::vector<int> mastersPerHost;
      std::vector<std::FILE*> tmpPartitionFiles;
      std::vector<std::vector<size_t> > global2Local;
//      std::vector<std::vector<size_t> > globalIDPerHost;

      void init(size_t nn, size_t ne, size_t numHosts) {
         global2Local.resize(numHosts);
         vertexOwners.resize(nn);
         vertexOwnersPacked.resize(nn);
         for (auto h = 0; h < numHosts; ++h) {
            tmpPartitionFiles.push_back(std::tmpfile());
            global2Local[h].resize(nn, -1);
         }
         for (int i = 0; i < nn; ++i) {
            vertexOwners[i].resize(numHosts);
         }
         mastersPerHost.resize(numHosts, 0);

         vertexMasters.resize(nn, -1);
         edgesPerHost.resize(numHosts, 0);
         verticesPerHost.resize(numHosts, 0);
      }
      static void writeEdgeToTempFile(std::FILE * f, size_t src, size_t dst) {
         NewEdgeData ed(src, dst);
         fwrite(&ed, sizeof(NewEdgeData), 1, f);
      }
      template<typename EType>
      static void writeEdgeToTempFile(std::FILE * f, size_t src, size_t dst, EType e) {
         NewEdgeData ed(src, dst, e);
         fwrite(&ed, sizeof(NewEdgeData), 1, f);
      }
      static bool readEdgeFromTempFile(std::FILE * f, size_t &src, size_t &dst) {
         NewEdgeData ed;
         fread(&ed, sizeof(NewEdgeData), 1, f);
         src = ed.src;
         dst = ed.dst;
         return feof(f);
      }
      template<typename EType>
      static bool readEdgeFromTempFile(std::FILE * f, size_t &src, size_t &dst, EType &e) {
         NewEdgeData ed;
         fread(&ed, sizeof(NewEdgeData), 1, f);
         src = ed.src;
         dst = ed.dst;
         e = ed.data;
         return feof(f);
      }

      /*
       *
       * */
      void assignEdge(OfflineGraph & g, NodeItType & _src, OfflineGraph::GraphNode & dst, size_t & eIdx, EdgeItType & e, size_t owner) {
         auto src = *_src;
         edgesPerHost[owner]++;
         vertexOwners[src][owner] = true;
         vertexOwners[dst][owner] = true;
         if (global2Local[owner][src] == -1) {
            global2Local[owner][src] = 1;//verticesPerHost[owner]++;
         } //if g2l[src]
         if (global2Local[owner][dst] == -1) {
            global2Local[owner][dst] = 1;//verticesPerHost[owner]++;
         } //if g2l[dst]
//         size_t new_src = global2Local[owner][src];
//         size_t new_dst = global2Local[owner][dst];
//         assert(new_src != -1 && new_dst != -1);
#if _HAS_EDGE_DATA
//         writeEdgeToTempFile(tmpPartitionFiles[owner],new_src, new_dst, g.getEdgeData<EdgeDataType>(e));
         writeEdgeToTempFile(tmpPartitionFiles[owner],src, dst, g.getEdgeData<EdgeDataType>(e));
#else
//         writeEdgeToTempFile(tmpPartitionFiles[owner], new_src, new_dst);
         writeEdgeToTempFile(tmpPartitionFiles[owner], src, dst);
#endif

      }
      /*
       * Go over all the nodes, and check if it has an edge
       * in a given host - if yes, assign a local-id to it.
       * This will ensure that localids and globalids are sequential
       * */
      void assignLocalIDs(size_t numhost, OfflineGraph &g) {
         for(size_t h=0; h<numhost; ++h){
            for(size_t n = 0; n < g.size(); ++n){
               if(global2Local[h][n]==1){
                  global2Local[h][n]=verticesPerHost[h]++;
               }
            }
         }

      }
      void writeReplicaInfo(std::string &basename, OfflineGraph &g, size_t numhosts) {
         for (size_t n = 0; n < g.size(); ++n) {
            for (int h = 0; h < numhosts; ++h) {
               if (vertexOwners[n][h]) {
                  vertexOwnersPacked[n].insert(h);
               }
            }
         }
         std::ofstream replica_file(getReplicaInfoFileName(basename, numhosts));
         auto numEntries = g.size();
         //         replica_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         replica_file << numEntries << ", " << numhosts << std::endl;
         for (size_t n = 0; n < g.size(); ++n) {
            auto owner = vertexOwnersPacked[n];
            size_t num_replicas = vertexOwnersPacked[n].size();
            //            replica_file.write(reinterpret_cast<const char*>(&num_replicas), sizeof(size_t));
            replica_file << num_replicas << ", " << std::distance(g.edge_begin(n), g.edge_end(n)) << std::endl;
         }

         replica_file.close();
      }
      /*
       * The assignment of masters to each vertex is done in a greedy manner -
       * the list of hosts with a copy of each vertex is scanned, and the one with
       * smallest number of masters is selected to be the master of the current
       * node, and the masters-count for the host is updated.
       * */
      void assignMasters(size_t nn, size_t numhost, OfflineGraph &g) {
         for (size_t n = 0; n < nn; ++n) {
            assert(vertexMasters[n] == -1);
            if (vertexOwnersPacked[n].size() == 0) {
               size_t minID = 0;
               size_t min_count = mastersPerHost[minID];
               for (int h = 1; h < numhost; ++h) {
                  if (min_count > mastersPerHost[h]) {
                     min_count = mastersPerHost[h];
                     minID = h;
                  }
               }
               vertexMasters[n] = minID;
               mastersPerHost[minID]++;
               //std::cout<<"No edges for "<< n <<" , "<<std::distance(g.edge_begin(n), g.edge_end(n))<<std::endl;
            } else {
               assert(vertexOwnersPacked[n].size() > 0);
               size_t minID = *vertexOwnersPacked[n].begin();
               size_t min_count = mastersPerHost[minID];
               for (auto host : vertexOwnersPacked[n]) {
                  if (mastersPerHost[host] < min_count) {
                     min_count = mastersPerHost[host];
                     minID = host;
                  }
               }
               assert(minID != -1);
               vertexMasters[n] = minID;
               mastersPerHost[minID]++;
            }

         }
      }
      void print_stats() {
         for (int i = 0; i < mastersPerHost.size(); ++i) {
            std::cout << "Masters " << i << ":: " << mastersPerHost[i] << std::endl;
         }
         for (int i = 0; i < edgesPerHost.size(); ++i) {
            std::cout << "Edges " << i << ":: " << edgesPerHost[i] << std::endl;
         }
      }
      ~VertexCutInfo() {
      }
   };
   /******************************************************************
    *
    *****************************************************************/
   VertexCutInfo vcInfo;
   /*
    * Overload this method for different implementations of the partitioning.
    * */
   size_t getEdgeOwner(size_t src, size_t dst, size_t num) {
      return rand() % num;
   }
   /*
    * Partitioning routine.
    * */
   void operator()(std::string & basename, OfflineGraph & g, size_t num_hosts) {
      Galois::Timer T_edge_assign, T_write_replica, T_assign_masters, T_write_partition, T_total, T_assign_localIDs;

      std::cout << "Partitioning: |V|= " << g.size() << " , |E|= " << g.sizeEdges() << " |P|= " << num_hosts << "\n";
      T_total.start();
      T_edge_assign.start();
      vcInfo.init(g.size(), g.sizeEdges(), num_hosts);
      auto prev_nbr_end = g.edge_begin(*g.begin());
      auto curr_nbr_end = g.edge_end(*g.begin());
      size_t edge_counter = 0;
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         curr_nbr_end = g.edge_end(*n);
         for (auto nbr = prev_nbr_end; nbr != curr_nbr_end; ++nbr, ++edge_counter) {
            auto dst = g.getEdgeDst(nbr);
            size_t owner = getEdgeOwner(src, dst, num_hosts);
//            void assignEdge(OfflineGraph & g, NodeItType & _src, NodeItType & dst, size_t & eIdx, EdgeItType & e, size_t owner) {
            vcInfo.assignEdge(g, n, dst, edge_counter, nbr, owner);
         }
         prev_nbr_end = curr_nbr_end;
      }
      T_edge_assign.stop();
      std::cout << "STEP#1-EdgesAssign:: " << T_edge_assign.get() << "\n";
      for (auto i = 0; i < num_hosts; ++i) {
         std::cout << "HOST#:: " << i << " , " << vcInfo.verticesPerHost[i] << ", " << vcInfo.edgesPerHost[i] << "\n";
      }
      T_assign_localIDs.start();
      vcInfo.assignLocalIDs(num_hosts, g);
      T_assign_localIDs.stop();
      T_write_replica.start();
      vcInfo.writeReplicaInfo(basename, g, num_hosts);
      T_write_replica.stop();

      T_assign_masters.start();
      vcInfo.assignMasters(g.size(), num_hosts, g);
      T_assign_masters.stop();

      T_write_partition.start();
      writePartitionsMem(basename, g, num_hosts);
      T_write_partition.stop();

      T_total.stop();
      std::cout << "STAT,EdgeAssig, WriteReplica,AssignMasters,WritePartition, Total\n";
      std::cout << num_hosts << "," << T_edge_assign.get() << "," << T_write_replica.get() << "," << T_assign_masters.get() << "," << T_write_partition.get() << ","
            << T_total.get() << "\n";
   }

   /*
    * Optimized implementation for memory usage.
    * Write both the metadata as well as the partition information.
    * */

   void writePartitionsMem(std::string & basename, OfflineGraph & g, size_t num_hosts) {
      //Create graph
      //TODO RK - Fix edgeData
      std::cout << " Low mem version\n";
      std::vector<size_t> &vertexOwners = vcInfo.vertexMasters;
      for (size_t h = 0; h < num_hosts; ++h) {
         std::cout << "Building partition " << h << "...\n";
         std::cout << "Analysis :: " << vcInfo.verticesPerHost[h] << " , " << vcInfo.edgesPerHost[h] << "\n";
         using namespace Galois::Graph;
         FileGraphWriter newGraph;
         newGraph.setNumNodes(vcInfo.verticesPerHost[h]);
         newGraph.setNumEdges(vcInfo.edgesPerHost[h]);
#if _HAS_EDGE_DATA
         newGraph.setSizeofEdgeData(sizeof(EdgeDataType));
#endif
         newGraph.phase1();
         std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
         std::cout << "Writing meta-file " << h << " to disk..." << meta_file_name << "\n";
         std::ofstream meta_file(meta_file_name, std::ofstream::binary);
         auto numEntries = vcInfo.verticesPerHost[h];
         meta_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         for (size_t n = 0; n < g.size(); ++n) {
            if (vcInfo.global2Local[h][n] != -1) {
               auto owner = vertexOwners[n];
               meta_file.write(reinterpret_cast<const char*>(&n), sizeof(n));
               meta_file.write(reinterpret_cast<const char*>(&vcInfo.global2Local[h][n]), sizeof(size_t));
               meta_file.write(reinterpret_cast<const char*>(&owner), sizeof(owner));
            }
         }
         meta_file.close();

         std::vector<NewEdgeData> newEdges;
         newEdges.clear();
         {
            rewind(vcInfo.tmpPartitionFiles[h]);
            int i = 0;
            size_t s, d;
#if _HAS_EDGE_DATA
            EdgeDataType ed;
            while(!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d, ed)) {
//               newEdges.push_back(NewEdgeData(s, d, ed));
               newEdges.push_back(NewEdgeData(vcInfo.global2Local[h][s], vcInfo.global2Local[h][d], ed));
//               std::cout<<i++<<", "<<newEdges.back().src << " , " << newEdges.back().dst << std::endl;
            }
#else
            while (!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d)) {
               newEdges.push_back(NewEdgeData(vcInfo.global2Local[s], vcInfo.global2Local[d]));
            }
#endif

         }
         assert(newEdges.size() == vcInfo.edgesPerHost[h]);
         for (auto e : newEdges) {
            newGraph.incrementDegree(e.src);
         }
         newGraph.phase2();
#if _HAS_EDGE_DATA
         std::vector<EdgeDataType> newEdgeData(newEdges.size());
#endif
         for (auto e : newEdges) {
            size_t idx = newGraph.addNeighbor(e.src, e.dst);
#if _HAS_EDGE_DATA
            newEdgeData[idx] = e.data;
#endif

         }
#if _HAS_EDGE_DATA
         memcpy(newGraph.finish<EdgeDataType>(), newEdgeData.data(), sizeof(EdgeDataType)*newEdges.size());
#else
         newGraph.finish<void>();
#endif
         std::string gFileName = getPartitionFileName(basename, h, num_hosts);
         std::cout << "Writing partition " << h << " to disk... " << gFileName << "\n";
         newGraph.toFile(gFileName);
      }      //End for-hosts
      vcInfo.print_stats();
   }      //end writePartitionsMem method
};
#endif /* GDIST_PARTITIONER_GREEDY_BALANCED_PARTITIONER_DISK_H_ */
