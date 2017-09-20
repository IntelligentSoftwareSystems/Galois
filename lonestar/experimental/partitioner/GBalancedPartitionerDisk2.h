/*
 * RandomPartitioner.h
 *
 *  Created on: Jun 15, 2016
 *      Author: rashid
 */

#ifndef GPGB_PARTITIONER_DISK_H_
#define GPGB_PARTITIONER_DISK_H_
#include <iostream>
#include <limits>
#include "Galois/Galois.h"
#include "Galois/gstl.h"
#include "Lonestar/BoilerPlate.h"
#include <set>
#include <vector>
#include <string>
#include <cstdio>
#include <unordered_set>
#include <unordered_map>
#include <random>


#include "Galois/Graphs/FileGraph.h"
#include "Galois/Runtime/OfflineGraph.h"
#include <sys/mman.h>
#include <mcheck.h>

/******************************************************************
 *
 *****************************************************************/

struct GBPD2 {
   typedef short PartitionIDType;
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
            src(~0), dst(~0) {
      }
   };
   struct VertexCutInfo {
      //Per host data structures - small size
      std::vector<size_t> edgesPerHost;
      std::vector<size_t> verticesPerHost;
      std::vector<size_t> mastersPerHost;


      //Per vertex data structure - large size
      std::vector<std::vector<bool> > _vertexOwnersPacked;
      std::vector<PartitionIDType> vertexMasters;

      std::vector<std::FILE*> tmpPartitionFiles;
      std::vector<std::string> tmpPartitionFiles_names;

      void print_size_stats(size_t num_hosts){
         long long total_size = 0;
         std::cout<<"Size == " << total_size*sizeof(size_t)/(1024*1024*1024.0f)<<" GB\n";
         return;
      }
      void init(size_t nn, size_t ne, size_t numHosts, std::string prefix_tmpFileName) {
         _vertexOwnersPacked.resize(nn);
         for(auto & n : _vertexOwnersPacked){
            n.resize(numHosts);
         }
         char tmpBuffer[L_tmpnam];
         for (size_t h = 0; h < numHosts; ++h) {
            tmpnam(tmpBuffer);
            //std::string tmpfileName = "/workspace/rashid/"+std::string(tmpBuffer)+".TMP." + std::to_string(h);
            std::string tmpfileName = prefix_tmpFileName + std::string(tmpBuffer)+".TMP." + std::to_string(h);
            tmpPartitionFiles_names.push_back(tmpfileName);
            std::FILE* file_handle = std::fopen(tmpfileName.c_str(), "wb+x");
            if(!file_handle){
               std::cerr<<"Unable to open tmp file :: "<<tmpfileName.c_str()<<"\n";
               exit(-1);
            }
            tmpPartitionFiles.push_back(file_handle);
            std::cout << "Pushed file : " << tmpfileName << "\n";
         }
         std::cout << "Init : Loop 1 \n";

         mastersPerHost.resize(numHosts, 0);
         std::cout << "Init : Loop 2 \n";
         edgesPerHost.resize(numHosts, 0);
         verticesPerHost.resize(numHosts, 0);
         std::cout << "Done Init \n";
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
      void assignEdge(galois::graphs::OfflineGraph & g, NodeItType & _src, galois::graphs::OfflineGraph::GraphNode & dst, size_t & eIdx, EdgeItType & e, PartitionIDType owner) {
         auto src = *_src;
         edgesPerHost[owner]++;
         _vertexOwnersPacked[src][owner]=1;
         _vertexOwnersPacked[dst][owner]=1;
//         fwrite(&owner, sizeof(owner), 1, edgeTmpFile);
#if _HAS_EDGE_DATA
//         writeEdgeToTempFile(tmpPartitionFiles[owner],new_src, new_dst, g.getEdgeData<EdgeDataType>(e));
         writeEdgeToTempFile(tmpPartitionFiles[owner],src, dst, g.getEdgeData<EdgeDataType>(e));
#else
//         writeEdgeToTempFile(tmpPartitionFiles[owner], new_src, new_dst);
         writeEdgeToTempFile(tmpPartitionFiles[owner], src, dst);
#endif

      }
      /*
       * The assignment of masters to each vertex is done in a greedy manner -
       * the list of hosts with a copy of each vertex is scanned, and the one with
       * smallest number of masters is selected to be the master of the current
       * node, and the masters-count for the host is updated.
       * */
      void assignMasters(size_t nn, size_t numhost, galois::graphs::OfflineGraph &g) {
         vertexMasters.resize(nn, ~0);
         for (size_t n = 0; n < nn; ++n) {
            if(vertexMasters[n] != ~0){
               std::cout<<"ERRR " << vertexMasters[n] << " Not eq " <<  ~0 << "\n";
            }
            assert(vertexMasters[n] == ~0);
#if 1
            {//begin change
                          size_t minID=~0;
                          size_t min_count=mastersPerHost[0];

                          for(size_t h = 0; h < numhost; ++h){
                             if(_vertexOwnersPacked[n][h]){
                                if(minID==~0){
                                   minID=h;
                                   min_count = mastersPerHost[minID];
                                }else{
                                   if(min_count > mastersPerHost[h]){ //found something smaller!
                                      minID = h;
                                      min_count = mastersPerHost[h];
                                   }
                                }
                             }
                          }

                          //Vertex does not have any edges - pick the least loaded partition anyway!
                          if(minID==~0){
                             minID=0;
                             min_count = mastersPerHost[minID];
                             for(size_t h=1; h<numhost ; ++h){
                                if (min_count > mastersPerHost[h]){
                                   min_count = mastersPerHost[h];
                                   minID=h;
                                }
                             }
                          }
                          assert(minID != ~0);
                          vertexMasters[n] = minID;
                          mastersPerHost[minID]++;
                          _vertexOwnersPacked[n].clear();
                       }//end change
#else
            if (vertexOwnersPacked[n].size() == 0) {
               size_t minID = 0;
               size_t min_count = mastersPerHost[minID];
               for (size_t h = 1; h < numhost; ++h) { /*Note - 0 is default host, so start at 1*/
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
                  }//end if
               }// end for
               assert(minID != ~0);
               vertexMasters[n] = minID;
               mastersPerHost[minID]++;
            }//end else
#endif
         }//end for

      }//end assignMasters
      void print_stats() {
         for (size_t i = 0; i < mastersPerHost.size(); ++i) {
            std::cout << "Masters " << i << ":: " << mastersPerHost[i] << std::endl;
         }
         for (size_t i = 0; i < edgesPerHost.size(); ++i) {
            std::cout << "Edges " << i << ":: " << edgesPerHost[i] << std::endl;
         }
      }
      ~VertexCutInfo() {
         std::cout<<"Cleaning up....\n";
        for(auto F : tmpPartitionFiles){
          std::fclose(F);
        }

        for(auto N : tmpPartitionFiles_names){
          std::remove(N.c_str());
        }
//        std::fclose(edgeTmpFile);
        std::cout<<"Done cleaning up....\n";
      }
   };
   /******************************************************************
    *
    *****************************************************************/
   VertexCutInfo vcInfo;
   /*
    * Overload this method for different implementations of the partitioning.
    * */
   PartitionIDType  getEdgeOwner_old(size_t src, size_t dst, PartitionIDType num) {
      return rand() % num;
   }
   PartitionIDType getEdgeOwner(size_t src, size_t dst, PartitionIDType num) {
      static std::mt19937 gen;
      static std::uniform_int_distribution<PartitionIDType> dist(0,num-1);
      return dist(gen);
   }
   /*
    * Partitioning routine.
    * */
   void operator()(std::string & basename, galois::graphs::OfflineGraph & g, size_t num_hosts, std::string prefix_tmpFileName) {
      galois::Timer T_edge_assign, T_write_replica, T_assign_masters, T_write_partition, T_total, T_assign_localIDs;

      std::cout << "Partitioning: |V|= " << g.size() << " , |E|= " << g.sizeEdges() << " |P|= " << num_hosts << "\n";
//      mtrace();
      T_total.start();
      T_edge_assign.start();
      vcInfo.init(g.size(), g.sizeEdges(), num_hosts, prefix_tmpFileName);
      auto prev_nbr_end = g.edge_begin(*g.begin());
      auto curr_nbr_end = g.edge_end(*g.begin());
      size_t edge_counter = 0;
      for (auto n = g.begin(); n != g.end(); ++n) {
         auto src = *n;
         curr_nbr_end = g.edge_end(*n);
         for (auto nbr = prev_nbr_end; nbr != curr_nbr_end; ++nbr, ++edge_counter) {
            auto dst = g.getEdgeDst(nbr);
            PartitionIDType owner = getEdgeOwner(src, dst, num_hosts);
//            void assignEdge(OfflineGraph & g, NodeItType & _src, NodeItType & dst, size_t & eIdx, EdgeItType & e, size_t owner) {
            vcInfo.assignEdge(g, n, dst, edge_counter, nbr, owner);
         }
         prev_nbr_end = curr_nbr_end;
      }
      T_edge_assign.stop();
      std::cout << "STEP#1-EdgesAssign:: " << T_edge_assign.get() << "\n";
      for (size_t i = 0; i < num_hosts; ++i) {
         std::cout << "HOST#:: " << i << " , " << vcInfo.verticesPerHost[i] << ", " << vcInfo.edgesPerHost[i] << "\n";
      }
      ///////////////////////////////////////////
      std::cout<<"Assigning masters\n";
      T_assign_masters.start();
      vcInfo.assignMasters(g.size(), num_hosts, g);
      T_assign_masters.stop();

      ///////////////////////////////////////////

      std::cout<<"Writing partitions\n";
      T_write_partition.start();
      writePartitionsMem(basename, g, num_hosts);
      T_write_partition.stop();

      T_total.stop();
      std::cout << "STAT,EdgeAssig, WriteReplica,AssignMasters,WritePartition, Total\n";
      std::cout << num_hosts << "," << T_edge_assign.get() << "," << T_write_replica.get() << "," << T_assign_masters.get() << "," << T_write_partition.get() << ","
            << T_total.get() << "\n";
//      muntrace();
   }

   /*
    * Optimized implementation for memory usage.
    * Write both the metadata as well as the partition information.
    * */
   void writePartitionsMem(std::string & basename,galois::graphs::OfflineGraph & g, size_t num_hosts) {
      //Create graph
      std::cout << " Low mem version\n";
      std::vector<std::FILE *> meta_files;
      for (size_t h = 0; h < num_hosts; ++h) {
         std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
         std::cout << "Analysis :: " << vcInfo.verticesPerHost[h] << " , " << vcInfo.edgesPerHost[h] << "\n";
         std::cout << "Writing meta-file " << h << " to disk..." << meta_file_name << " ";
         std::FILE* m = std::fopen(meta_file_name.c_str(), "wb+x");
         if(!m){
            std::cout<<"Failed to create meta file : " << meta_file_name << "\n";
            exit(-1);
         }
         meta_files.push_back(m);
         size_t dummy=0;
         fwrite(&dummy, sizeof(size_t),1, m);
      }
      for(size_t n=0; n < g.size(); ++n){
         size_t owner = vcInfo.vertexMasters[n];
         assert(owner<num_hosts);
         for(size_t h=0; h<num_hosts; ++h){
//         for(auto h : vcInfo._vertexOwnersPacked[n]){
            if(vcInfo._vertexOwnersPacked[n][h]){
               size_t localID = vcInfo.verticesPerHost[h]++;
               fwrite(&n, sizeof(size_t), 1, meta_files[h]);
               fwrite(&localID, sizeof(size_t), 1, meta_files[h]);
               fwrite(&owner, sizeof(size_t), 1, meta_files[h]);
            }
         }
      }
      for (size_t h = 0; h < num_hosts; ++h) {
//         std::cout<<"@ " << ftell(meta_files[h]) << " ";
         fseek(meta_files[h], 0,std::ios_base::beg);
         size_t numEntries = vcInfo.verticesPerHost[h];
         fwrite(&numEntries, sizeof(size_t), 1, meta_files[h]);
//         std::cout<<"@ " << ftell(meta_files[h]) << " ";
//         std::cout<<"Meta " << h << " has " << numEntries << "vertices\n";
         fclose(meta_files[h]);
      }

      for (size_t h = 0; h < num_hosts; ++h) {
         std::unordered_map<size_t,size_t> global2Local;
         size_t num_entries;
         std::string meta_file_name = getMetaFileName(basename, h, num_hosts);
         std::FILE * in_meta_file = std::fopen(meta_file_name.c_str(), "rb");
         if(!in_meta_file){
            std::cout<<"Failed to reload meta_file " << meta_file_name << "\n";
            exit(-1);
         }
         fseek(in_meta_file, 0,std::ios_base::beg);
         fread(&num_entries, sizeof(size_t),1, in_meta_file);
         std::cout << "Reloading partition " << h << "... Loading : " << meta_file_name<< "- Entries=["<< num_entries<<"] \n";

         while(!feof(in_meta_file)){
            size_t gid, lid, owner;
            fread(&gid, sizeof(size_t),1, in_meta_file);
            fread(&lid, sizeof(size_t),1, in_meta_file);
            fread(&owner, sizeof(size_t),1, in_meta_file);
            global2Local[gid]=lid;
//            if(owner >=num_hosts) std::cout<<"ERR " <<gid << " , " << lid<<", " << owner<<"\n";
            assert(owner<num_hosts );
//            std::cout<<gid << " , " << lid<<", " << owner<<"\n";
         }

         std::cout << "Analysis :: " << vcInfo.verticesPerHost[h] << " , " << vcInfo.edgesPerHost[h] << "\n";
         using namespace galois::graphs;
         FileGraphWriter * ng = new FileGraphWriter();
         FileGraphWriter &newGraph = *ng;
         newGraph.setNumNodes(vcInfo.verticesPerHost[h]);
         newGraph.setNumEdges(vcInfo.edgesPerHost[h]);
#if _HAS_EDGE_DATA
         newGraph.setSizeofEdgeData(sizeof(EdgeDataType));
#endif
         newGraph.phase1();
         {
            rewind(vcInfo.tmpPartitionFiles[h]);
            size_t s, d;
#if _HAS_EDGE_DATA
            EdgeDataType ed;
            while(!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d, ed)) {
//               vcInfo.writeEdgeToTempFile(vcInfo.tmpGraphFiles[h],vcInfo.global2Local[h][s], vcInfo.global2Local[h][d], ed);
               newGraph.incrementDegree(global2Local[s]);
//               std::cout<<i++<<", "<<newEdges.back().src << " , " << newEdges.back().dst << std::endl;
            }
#else
            while (!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d)) {
               vcInfo.writeEdgeToTempFile(vcInfo.tmpGraphFiles[h],vcInfo.global2Local[h][s], vcInfo.global2Local[h][d]);
               newGraph.incrementDegree(vcInfo.global2Local[h][s]);

            }
#endif

         }
         newGraph.phase2();
         rewind(vcInfo.tmpPartitionFiles[h]);
#if _HAS_EDGE_DATA
         std::vector<EdgeDataType> newEdgeData(vcInfo.edgesPerHost[h]);
         size_t s, d;
         EdgeDataType ed;
         while(!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d, ed)){
            size_t idx = newGraph.addNeighbor(global2Local[s],global2Local[d]);
            newEdgeData[idx] = ed;
         }
         EdgeDataType * edgeDataPtr = newGraph.finish<EdgeDataType>();
         memcpy(edgeDataPtr, newEdgeData.data(), sizeof(EdgeDataType)*newEdgeData.size());
         newEdgeData.clear();
#else
         size_t s, d;
         while(!vcInfo.readEdgeFromTempFile(vcInfo.tmpPartitionFiles[h], s, d))
         {
            size_t idx = newGraph.addNeighbor(global2Local[s],global2Local[d]);
         }
         newGraph.finish<void>();
#endif

         std::string gFileName = getPartitionFileName(basename, h, num_hosts);
         std::cout << "Writing partition " << h << " to disk... " << gFileName << "\n";
         newGraph.toFile(gFileName);
         delete ng;
      }      //End for-hosts
      vcInfo.print_stats();
   }      //end writePartitionsMem method
};
#endif /* GPGB_PARTITIONER_DISK_H_ */
