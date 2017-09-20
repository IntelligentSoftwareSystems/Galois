/*
 * RandomPartitioner.h
 *
 *  Created on: Jun 15, 2016
 *      Author: rashid
 */


#ifndef GDIST_EXP_APPS_PARTITIONER_RANDOMPARTITIONER_H_
#define GDIST_EXP_APPS_PARTITIONER_RANDOMPARTITIONER_H_
#include <iostream>
#include <limits>
#include "galois/Galois.h"
#include "galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>

#include "galois/Graphs/FileGraph.h"
#include "galois/runtime/OfflineGraph.h"

/******************************************************************
 *
 *****************************************************************/

struct RandomPartitioner {
   struct VertexCutInfo {
      std::vector<size_t> edgeOwners;
      std::vector<size_t> edgesPerHost;

      void init(size_t ne, size_t numHosts) {
         edgeOwners.resize(ne);
         edgesPerHost.resize(numHosts);
      }
      /*
       *
       * */
      void assignEdge(OfflineGraph & g, NodeItType & src, EdgeItType & e, size_t owner) {
         size_t eIdx = std::distance(g.edge_begin(*g.begin()), e);
         auto dst = g.getEdgeDst(e);
         edgeOwners[eIdx] = owner;
         edgesPerHost[owner]++;
      }
      size_t getEdgeOwner(OfflineGraph & g, EdgeItType & e) const {
         size_t eIdx = std::distance(g.edge_begin(*g.begin()), e);
         return edgeOwners[eIdx];
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
      std::cout << "Partitioning: |V|= "<< g.size() << " , |E|= " << g.sizeEdges() << " |P|= " << num_hosts << "\n";
      vcInfo.init(g.sizeEdges(), num_hosts);
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         for (auto nbr = g.edge_begin(*n); nbr != g.edge_end(*n); ++nbr) {
            auto dst = g.getEdgeDst(nbr);
            size_t owner = getEdgeOwner(src, dst, num_hosts);
            vcInfo.assignEdge(g, n, nbr, owner);
         }
      }
//      assignVertices(g, vcInfo, num_hosts);
      writePartitionsMem(basename, g, num_hosts);
   }
   /*
    * Edges have been assigned. Now, go over each partition, for any vertex in the partition
    * create a new local-id, and update all the edges to the new local-ids.
    * */
//   void assignVertices(OfflineGraph & g, VertexCutInfo & vcInfo, size_t num_hosts) {
//      size_t verticesSum = 0;
//      if (false) {
//         for (size_t h = 0; h < num_hosts; ++h) {
//            for (size_t v = 0; v < vcInfo.verticesPerHost[h].size(); ++v) {
//               auto vertex = vcInfo.verticesPerHost[h][v];
//               std::cout << "Host " << h << " Mapped Global:: " << vertex << " to Local:: " << vcInfo.hostGlobalToLocalMapping[h][vertex] << "\n";
//            }
//         }
//
//      }
//      std::vector<size_t> hostVertexCounters(num_hosts);
//      for (auto i : vcInfo.vertexOwners) {
//         verticesSum += i.second.size();
//      }
//      for (size_t i = 0; i < num_hosts; ++i) {
//         std::cout << "Host :: " << i << " , Vertices:: " << vcInfo.verticesPerHost[i].size() << ", Edges:: " << vcInfo.edgesPerHost[i] << "\n";
//      }
//      std::cout << "Vertices - Created ::" << verticesSum << " , Actual :: " << g.size() << ", Ratio:: " << verticesSum / (float) (g.size()) << "\n";
//   }
   /*
    * Write both the metadata as well as the partition information.
    * */
  /* void writePartitions(std::string & basename, OfflineGraph & g, VertexCutInfo & vcInfo, size_t num_hosts) {
      //Create graph
      //TODO RK - Fix edgeData
      std::cout << " Regular version\n";
      std::vector<std::vector<std::pair<size_t, size_t>>>newEdges(num_hosts);
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
            auto dst = g.getEdgeDst(e);
            size_t owner = vcInfo.getEdgeOwner(g, e);
            size_t new_src = vcInfo.hostGlobalToLocalMapping[owner][src];
            size_t new_dst = vcInfo.hostGlobalToLocalMapping[owner][dst];
            newEdges[owner].push_back(std::pair<size_t, size_t>(new_src, new_dst));
         }
      }
      for (size_t i = 0; i < num_hosts; ++i) {
         using namespace galois::Graph;
         FileGraphWriter newGraph;
         newGraph.setNumNodes(vcInfo.hostGlobalToLocalMapping[i].size());
         newGraph.setNumEdges(newEdges[i].size());
         newGraph.phase1();
//         char filename[256];
//         sprintf(filename, "partition_%zu_%zu.dimacs", i, num_hosts);
         std::string meta_file_name = getMetaFileName(basename, i, num_hosts);
//         char meta_file_name[256];
//         sprintf(meta_file_name, "partition_%zu_of_%zu.gr.meta", i, num_hosts);
         std::ofstream meta_file(meta_file_name, std::ofstream::binary);
         auto numEntries = vcInfo.hostGlobalToLocalMapping[i].size();
         meta_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         for (auto n : vcInfo.hostGlobalToLocalMapping[i]) {
            auto owner = *vcInfo.vertexOwners[n.first].begin();
            meta_file.write(reinterpret_cast<const char*>(&n.first), sizeof(n.first));
            meta_file.write(reinterpret_cast<const char*>(&n.second), sizeof(n.second));
            meta_file.write(reinterpret_cast<const char*>(&owner), sizeof(owner));
         }
         meta_file.close();

         for (auto e : newEdges[i]) {
            newGraph.incrementDegree(e.first);
         }
         newGraph.phase2();
         for (auto e : newEdges[i]) {
            newGraph.addNeighbor(e.first, e.second);
         }
         newGraph.finish<void>();
//         char gFileName[256];
//         sprintf(gFileName, "partition_%zu_of_%zu.gr", i, num_hosts);
         std::string gFileName = getPartitionFileName(basename, i, num_hosts);
         newGraph.toFile(gFileName);
      }
   }*/
   /*
    * Optimized implementation for memory usage.
    * Write both the metadata as well as the partition information.
    * */
   struct NewEdgeData{
      size_t src, dst;
#if _HAS_EDGE_DATA
      EdgeDataType data;
#endif

#if _HAS_EDGE_DATA
      NewEdgeData(size_t s, size_t d, EdgeDataType dt ):src(s), dst(d),data(dt){}
#else
      NewEdgeData(size_t s, size_t d):src(s), dst(d){}

#endif
   };
   void writePartitionsMem(std::string & basename, OfflineGraph & g, size_t num_hosts) {
      //Create graph
      //TODO RK - Fix edgeData
      std::cout << " Low mem version\n";
      std::vector<size_t> vertexOwners(g.size(), -1);
      {
         //Go over all the vertices and assign an owner.

      }
      for (size_t h = 0; h < num_hosts; ++h) {
         std::cout << "Building partition " << h << "...\n";
         std::vector<size_t> global2Local(g.size(), -1);
         size_t newNodeCounter = 0;
         for (auto n = g.begin(); n != g.end(); ++n) {
            auto src = *n;
            for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
               auto dst = g.getEdgeDst(e);
               size_t owner = vcInfo.getEdgeOwner(g, e);
               if (owner == h) {
                  if (global2Local[src] == -1) {
                     if (vertexOwners[src] == -1) {
                        vertexOwners[src] = h;
                     }
                     global2Local[src] = newNodeCounter++;
                  }
                  if (global2Local[dst] == -1) {
                     if (vertexOwners[dst] == -1) {
                        vertexOwners[dst] = h;
                     }
                     global2Local[dst] = newNodeCounter++;
                  }
               }
            }
         }//For each node
         std::vector<NewEdgeData> newEdges;
         for (auto n = g.begin(); n != g.end(); ++n) {
            auto src = *n;
            for (auto e = g.edge_begin(*n); e != g.edge_end(*n); ++e) {
               auto dst = g.getEdgeDst(e);
               size_t owner = vcInfo.getEdgeOwner(g, e);
               if (owner == h) {
                  size_t new_src = global2Local[src];
                  size_t new_dst = global2Local[dst];
                  assert(new_src!=-1 && new_dst!=-1);
#if _HAS_EDGE_DATA
                  newEdges.push_back(NewEdgeData(new_src, new_dst, g.getEdgeData<EdgeDataType>(e)));
#else
                  newEdges.push_back(NewEdgeData(new_src, new_dst));
#endif
               }      //End if
            }      //End for neighbors
         }      //end for nodes
         std::cout << "Analysis :: " << newNodeCounter << " , " << newEdges.size () << "\n";
         using namespace galois::Graph;
         FileGraphWriter newGraph;
         newGraph.setNumNodes(newNodeCounter);
         newGraph.setNumEdges(newEdges.size());
#if _HAS_EDGE_DATA
         newGraph.setSizeofEdgeData(sizeof(EdgeDataType));
#endif
         newGraph.phase1();
         std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
         std::cout << "Writing meta-file " << h << " to disk..." << meta_file_name<< "\n";
         std::ofstream meta_file(meta_file_name, std::ofstream::binary);
         auto numEntries = newNodeCounter;
         meta_file.write(reinterpret_cast<char*>(&(numEntries)), sizeof(numEntries));
         for (size_t n = 0; n < g.size(); ++n) {
            if (global2Local[n] != -1) {
               auto owner = vertexOwners[n];
               meta_file.write(reinterpret_cast<const char*>(&n), sizeof(n));
               meta_file.write(reinterpret_cast<const char*>(&global2Local[n]), sizeof(global2Local[n]));
               meta_file.write(reinterpret_cast<const char*>(&owner), sizeof(owner));
            }
         }

         meta_file.close();
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
         std::cout << "Writing partition " << h << " to disk... " << gFileName<<"\n";
         newGraph.toFile(gFileName);
      }      //End for-hosts
   }      //end writePartitionsMem method
};



#endif /* GDIST_EXP_APPS_PARTITIONER_RANDOMPARTITIONER_H_ */
