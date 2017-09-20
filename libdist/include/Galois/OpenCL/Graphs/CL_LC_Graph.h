/*
 * CL_LC_Graph.h
 *
 *  Created on: Nov 19, 2015
 *      Author: Rashid Kaleem (rashid.kaleem@gmail.com)
 */
#include "Galois/OpenCL/CL_Header.h"
#include "Galois/OpenCL/CL_Kernel.h"
#include <boost/iterator/counting_iterator.hpp>
//#include "Galois/Runtime/hGraph.h"
#ifndef _GDIST_CL_LC_Graph_H_
#define _GDIST_CL_LC_Graph_H_

namespace galois {
   namespace OpenCL {
      namespace Graphs {

      /*####################################################################################################################################################*/
      template<typename DataType>
      struct BufferWrapperImpl{
         typedef DataType MessageType;
         typedef size_t LocalIDType;
         std::vector<std::vector<MessageType>> inMessageBuffers;
         std::vector<std::vector<MessageType>> outMessageBuffers;
         std::vector<std::vector<LocalIDType>> inMessageIDs;
         std::vector<std::vector<LocalIDType>> outMessageIDs;
         std::vector<cl_mem> devInMessageBuffers;
         std::vector<cl_mem> devOutMessageBuffers;
         std::vector<cl_mem> devInMessageIDs;
         std::vector<cl_mem> devOutMessageIDs;

         int num_hosts;
         struct BufferWrapperGPU{
            cl_mem counters; //how many entries per peer
            cl_mem messages; // space for messages
            cl_mem localIDs; // localIDs for messages
            int numHosts;
            cl_mem buffer_ptr; // pointer for self/structure
         };
         BufferWrapperGPU inMessages, outMessages;
         CL_Kernel inMsgApply, outMsgGen;
         BufferWrapperImpl():num_hosts(0){
            auto * ctx = galois::OpenCL::getCLContext();
            inMsgApply.init(ctx->get_default_device(), "buffer_wrapper.cl", "recvApply");
            outMsgGen.init(ctx->get_default_device(), "buffer_wrapper.cl", "prepSend");
         }
         /*
          * hGraph - slaveNodes - nodes on other hosts that have a master here. Hence, outMessageIDs
          * hGraph - masterNodes - nodes on othor hosts that are slaves here. Hence, inMessageIDs
          * */
         void init(int nh, std::vector<std::vector<size_t>> slaveNodes, std::vector<std::vector<size_t>> masterNodes ){
            num_hosts = nh;
            inMessageBuffers.resize(num_hosts);
            outMessageBuffers.resize(num_hosts);
            inMessageIDs.resize(num_hosts);
            outMessageIDs.resize(num_hosts);
            for(size_t h = 0;  h < num_hosts; ++h){
               for(auto val : masterNodes[h]){
                  inMessageIDs[h].push_back(val);
               }
               for(auto val : slaveNodes[h]){
                  outMessageIDs[h].push_back(val);
               }
               inMessageBuffers[h].resize(masterNodes[h].size());
               outMessageBuffers[h].resize(slaveNodes[h].size());
            }
            allocate();
         }
         template<typename GraphType, typename Fn>
         void sync_outgoing(GraphType & g){
            //void (hGraph::*fn)(galois::Runtime::RecvBuffer&) = &hGraph::syncPullRecvApply<FnTy>;
            auto& net = galois::Runtime::getSystemNetworkInterface();

            for(size_t h = 0 ; h < num_hosts; ++h){
               for(size_t i=0; i< outMessageIDs[h].size(); ++i){
                  //static unsigned int extract(uint32_t node_id, const struct NodeData & node) {
                  outMessageBuffers[h][i]=Fn::extract(outMessageIDs[h][i], g.getData(outMessageIDs[h][i]));
               }
               //TODO RK - send messages.
            }
         }

         template<typename GraphType, typename Fn>
         void sync_incoming(GraphType & g, std::vector<std::vector<unsigned int>> & incomingMessages){
            auto& net = galois::Runtime::getSystemNetworkInterface();
            cl_int err;
            auto * ctx = galois::OpenCL::getCLContext();
            cl_command_queue queue = ctx->get_default_device()->command_queue();
            //
            for(size_t h = 0 ; h < num_hosts; ++h){
               //err = clEnqueueWriteBuffer(queue, devInMessageBuffers[h], CL_TRUE, 0, sizeof(unsigned int)*(incomingMessages[h],size()), incomingMessages[h].data(), 0, NULL, NULL);
               /*for(size_t i=0; i< outMessageIDs[h].size(); ++i){
                  //      static void setVal (uint32_t node_id, struct NodeData & node, unsigned int y) {
                  Fn::setVal(inMessageIDs[h][i], g.getData(inMessageIDs[h][i]), inMessageBuffers[h][i]);
               }*/
               //TODO RK - send messages.
            }//end for host
            this->inMsgApply();
         }

         void allocate(){
            auto * ctx = galois::OpenCL::getCLContext();
            cl_command_queue queue = ctx->get_default_device()->command_queue();

            assert(ctx!=nullptr && "CL Context should be non-null");
            cl_int err;
            CL_Kernel initBuffer(getCLContext()->get_default_device(),"buffer_wrapper.cl", "initBufferWrapper", false);
            CL_Kernel registerBuffer(getCLContext()->get_default_device(),"buffer_wrapper.cl", "registerBuffer",false);
            {
               inMessages.counters = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(cl_uint) *( num_hosts), 0, &err);
               inMessages.messages= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(cl_mem) *( num_hosts), 0, &err);
               inMessages.localIDs= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(cl_mem) *( num_hosts), 0, &err);
               inMessages.buffer_ptr= clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(BufferWrapperGPU), 0, &err);
               initBuffer.set_arg_raw(0, inMessages.buffer_ptr);
               initBuffer.set_arg_raw(1, inMessages.counters);
               initBuffer.set_arg_raw(2, inMessages.messages);
               initBuffer.set_arg_raw(3, inMessages.localIDs);
//               initBuffer.set_arg(4, num_hosts);
//               galois::OpenCL::CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &val), "Arg-1, is NOT set!");
               clSetKernelArg(initBuffer.kernel, 4, sizeof(cl_int), &num_hosts);
               initBuffer.run_task();

               registerBuffer.set_arg_raw(0, inMessages.buffer_ptr);
               for(int i=0; i < num_hosts; ++i){
                  cl_mem msg_buffer = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(MessageType) * (inMessageIDs[i].size()), 0, &err);
                  cl_mem id_buffer = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(MessageType) * (inMessageIDs[i].size()), 0, &err);
                  err = clEnqueueWriteBuffer(queue, id_buffer, CL_TRUE, 0, sizeof(unsigned int)*(inMessageIDs[i].size()), inMessageIDs[i].data(), 0, NULL, NULL);

                  devInMessageBuffers.push_back(msg_buffer);
                  devInMessageIDs.push_back(id_buffer);

                  registerBuffer.set_arg_raw(1, id_buffer);
                  registerBuffer.set_arg_raw(2, msg_buffer);
//                  registerBuffer.set_arg(3, i);
                  clSetKernelArg(registerBuffer.kernel, 3, sizeof(cl_int), &i);
                  size_t numItems = inMessageIDs[i].size();
                  clSetKernelArg(registerBuffer.kernel, 4, sizeof(cl_int), &numItems);
                  registerBuffer.run_task();
               }
               // TODO resume here
               //inMsgApply.set_arg(???GRAPH, inMessages);
            }
            {
               outMessages.counters = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(cl_uint) *( num_hosts), 0, &err);
               outMessages.messages= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(cl_mem) *( num_hosts), 0, &err);
               outMessages.localIDs= clCreateBuffer(ctx->get_default_device()->context(), CL_MEM_READ_WRITE, sizeof(cl_mem) *( num_hosts), 0, &err);
               outMessages.buffer_ptr= clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(BufferWrapperGPU), 0, &err);
               initBuffer.set_arg_raw(0, outMessages.buffer_ptr);
               initBuffer.set_arg_raw(1, outMessages.counters);
               initBuffer.set_arg_raw(2, outMessages.messages);
               initBuffer.set_arg_raw(3, outMessages.localIDs);
//               initBuffer.set_arg(4, num_hosts);
               clSetKernelArg(initBuffer.kernel, 4, sizeof(cl_int), &num_hosts);
               initBuffer.run_task();

               registerBuffer.set_arg_raw(0, inMessages.buffer_ptr);
               for(int i=0; i < num_hosts; ++i){
                  cl_mem msg_buffer = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(MessageType) * (outMessageIDs[i].size()), 0, &err);
                  cl_mem id_buffer = clCreateBuffer(ctx->get_default_device()->context(),CL_MEM_READ_WRITE, sizeof(MessageType) * (outMessageIDs[i].size()), 0, &err);
                  err = clEnqueueWriteBuffer(queue, id_buffer, CL_TRUE, 0, sizeof(unsigned int)*(outMessageIDs[i].size()), outMessageIDs[i].data(), 0, NULL, NULL);

                  devOutMessageBuffers.push_back(msg_buffer);
                  devOutMessageIDs.push_back(id_buffer);

                  registerBuffer.set_arg_raw(1, id_buffer);
                  registerBuffer.set_arg_raw(2, msg_buffer);
//                  registerBuffer.set_arg(3, i);
                  clSetKernelArg(registerBuffer.kernel, 3, sizeof(cl_int), &i);
                  registerBuffer.run_task();
               }
            }

         }
         void deallocate(){
            for(auto b : devInMessageBuffers){
               clReleaseMemObject(b);
            }
            for(auto b : devInMessageIDs){
               clReleaseMemObject(b);
            }
            clReleaseMemObject(inMessages.counters);
            clReleaseMemObject(inMessages.messages);
            clReleaseMemObject(inMessages.buffer_ptr);
         }
         /*
template<typename FnTy>
  void sync_push() {
    void (hGraph::*fn)(galois::Runtime::RecvBuffer&) = &hGraph::syncRecvApply<FnTy>;
    auto& net = galois::Runtime::getSystemNetworkInterface();
    for (unsigned x = 0; x < hostNodes.size(); ++x) {
      if (x == id) continue;
      uint32_t start, end;
      std::tie(start, end) = nodes_by_host(x);
      if (start == end) continue;
      galois::Runtime::SendBuffer b;
      gSerialize(b, idForSelf(), fn, (uint32_t)(end-start));
      for (; start != end; ++start) {
        auto gid = L2G(start);
        if( gid > totalNodes){
          assert(gid < totalNodes);
        }
        gSerialize(b, gid, FnTy::extract(getData(start)));
        FnTy::reset(getData(start));
      }
      net.send(x, syncRecv, b);
    }
    //Will force all messages to be processed before continuing
    net.flush();
    galois::Runtime::getHostBarrier().wait();

  }

          */
#if 0 // TODO Finsih implementation
         template<typename GraphType>
         void prepSend(GraphType & g){
            CL_Kernel pushKernel;
            auto * ctx = galois::OpenCL::getCLContext();
            auto & net = galois::Runtime::getSystemNetworkInterface();
            pushKernel.init("buffer_wrapper.cl", "syncPush");
            pushKernel.set_arg(0, g);
            pushKernel.set_arg(1, gpu_impl.buffer_ptr);
            pushKernel.set_work_size(num_ghosts);
            pushKernel();
            const unsigned selfID=net.ID;
            for(int i=0; i<net.Num; ++i){
               if(i==selfID)continue;
               galois::Runtime::SendBuffer buffer;
               const uint32_t counter =  0; // get counters from buffers.
               gSerialize(buffer, selfID, uint32_t(counter));
               cl_command_queue queue = ctx->get_default_device()->command_queue();
               int err = clEnqueueReadBuffer(queue, gpuMessageBuffer[i], CL_TRUE, 0, sizeof(MessageType)*(num_ghosts), &messageBuffers[i], 0, NULL, NULL);
               for(int m=0; m<counter; ++m){
                  gSerialize(buffer, messageBuffers[i][m].first, messageBuffers[i][m].second);
               }
               net.send(i, prepRecv, buffer);
               //send to recipient.
            }

         }//End prepSend
#endif // Finish implementation TODO
         /*
            static void syncRecv(galois::Runtime::RecvBuffer& buf) {
    uint32_t oid;
    void (hGraph::*fn)(galois::Runtime::RecvBuffer&);
    galois::Runtime::gDeserialize(buf, oid, fn);
    hGraph* obj = reinterpret_cast<hGraph*>(ptrForObj(oid));
    (obj->*fn)(buf);
  }

  template<typename FnTy>
  void syncRecvApply(galois::Runtime::RecvBuffer& buf) {
    uint32_t num;
    galois::Runtime::gDeserialize(buf, num);
    for(; num ; --num) {
      uint64_t gid;
      typename FnTy::ValTy val;
      galois::Runtime::gDeserialize(buf, gid, val);
      assert(isOwned(gid));
      FnTy::reduce(getData(gid - globalOffset), val);
    }
  }

          */
/*
         template<typename GraphType>
         void prepRecv(GraphType & g, galois::Runtime::RecvBuffer& buffer){
            CL_Kernel applyKernel;
            auto * ctx = galois::OpenCL::getCLContext();
            auto & net = galois::Runtime::getSystemNetworkInterface();
            cl_command_queue queue = ctx->get_default_device()->command_queue();
            unsigned sender=0;
            const uint32_t counter =  0; // get counters from buffers.
            gDeserialize(buffer, sender, counter);
            for(int m=0; m<counter; ++m){
               gDeserialize(buffer, messageBuffers[sender][m].first, messageBuffers[sender][m].second);
            }
            int err = clEnqueueWriteBuffer(queue, gpuMessageBuffer[sender], CL_TRUE, 0, sizeof(MessageType)*(num_ghosts), &messageBuffers[sender], 0, NULL, NULL);
               applyKernel.init("buffer_wrapper.cl", "syncApply");
               applyKernel.set_arg(0, g);
               applyKernel.set_arg(1, gpuMessageBuffer[sender]);
               applyKernel.set_work_size(counter);
               applyKernel();

         }//End prepSend
*/
      };
      /*####################################################################################################################################################*/
               enum GRAPH_FIELD_FLAGS {
            NODE_DATA=0x1,EDGE_DATA=0x10,OUT_INDEX=0x100, NEIGHBORS=0x1000,ALL=0x1111, ALL_DATA=0x0011, STRUCTURE=0x1100,
         };
         /*
          std::string vendor_name;
          cl_platform_id m_platform_id;
          cl_device_id m_device_id;
          cl_context m_context;
          cl_command_queue m_command_queue;
          cl_program m_program;
          */
         struct _CL_LC_Graph_GPU {
            cl_mem outgoing_index;
            cl_mem node_data;
            cl_mem neighbors;
            cl_mem edge_data;
            cl_uint num_nodes;
            cl_uint num_edges;
            cl_uint num_owned;
            cl_uint local_offset;
            cl_mem ghostMap;
         };

         static const char * cl_wrapper_str_CL_LC_Graph =
         "\
         typedef struct _ND{int dist_current;} NodeData;\ntypedef unsigned int EdgeData; \n\
      typedef struct _GraphType { \n\
   uint _num_nodes;\n\
   uint _num_edges;\n\
    uint _node_data_size;\n\
    uint _edge_data_size;\n\
    uint _num_owned;\n\
    uint _global_offset;\n\
    __global NodeData *_node_data;\n\
    __global uint *_out_index;\n\
    __global uint *_out_neighbors;\n\
    __global EdgeData *_out_edge_data;\n\
    }GraphType;\
      ";
         static const char * init_kernel_str_CL_LC_Graph =
         "\
      __kernel void initialize_graph_struct(__global uint * res, __global uint * g_meta, __global NodeData *g_node_data, __global uint * g_out_index, __global uint * g_nbr, __global EdgeData * edge_data){ \n \
      __global GraphType * g = (__global GraphType *) res;\n\
      g->_num_nodes = g_meta[0];\n\
      g->_num_edges = g_meta[1];\n\
      g->_node_data_size = g_meta[2];\n\
      g->_edge_data_size= g_meta[3];\n\
      g->_num_owned= g_meta[4];\n\
      g->_global_offset= g_meta[6];\n\
      g->_node_data = g_node_data;\n\
      g->_out_index= g_out_index;\n\
      g->_out_neighbors = g_nbr;\n\
      g->_out_edge_data = edge_data;\n\
      }\n\
      ";


         template<typename NodeDataTy, typename EdgeDataTy>
         struct CL_LC_Graph {
            typedef NodeDataTy NodeDataType;
            typedef EdgeDataTy EdgeDataType;
            typedef boost::counting_iterator<uint64_t> NodeIterator;
            typedef NodeIterator GraphNode;
            typedef boost::counting_iterator<uint64_t> EdgeIterator;
            typedef unsigned int NodeIDType;
            typedef unsigned int EdgeIDType;
            typedef CL_LC_Graph<NodeDataType,EdgeDataType> SelfType;

            /*
             * A wrapper class to return when a request for getData is made.
             * The wrapper ensures that any pending writes are made before the
             * kernel is called on the graph instance. This is done either auto-
             * matically via the destructor or through a "sync" method.
             * */
            class NodeDataWrapper : public NodeDataType{
            public:
               SelfType * const parent;
               NodeIterator id;
//               NodeDataType cached_data;
               bool isDirty;
               size_t idx_in_vec;
               NodeDataWrapper( SelfType * const p, NodeIterator & _id, bool isD): parent(p), id(_id),isDirty(isD){
                  parent->read_node_data_impl(id, *this);
                  if(isDirty){
                     parent->dirtyData.push_back(this);
                     idx_in_vec =parent->dirtyData.size();
                  }
               }
               NodeDataWrapper( const SelfType * const p, NodeIterator & _id, bool isD): parent(const_cast<SelfType *>(p)), id(_id),isDirty(isD){
                  parent->read_node_data_impl(id, *this);
                  if(isDirty){
                     parent->dirtyData.push_back(this);
                     idx_in_vec =parent->dirtyData.size();
                  }
               }
               NodeDataWrapper(const NodeDataWrapper & other):parent(other.parent), id(other.id), isDirty(other.isDirty), idx_in_vec(other.idx_in_vec){

               }
               ~NodeDataWrapper(){
                  //Write upon cleanup for automatic scope cleaning.
                  if(isDirty){
//                     fprintf(stderr, "Destructor - Writing to device %d\n", *(id));
                     parent->write_node_data_impl(id, *this);
                     parent->dirtyData[idx_in_vec] = nullptr;
                  }
               }
            };//End NodeDataWrapper

            typedef NodeDataWrapper UserNodeDataType;
         protected:
            galois::OpenCL::CLContext * ctx = getCLContext();
            //CPU Data
            size_t _num_nodes;
            size_t _num_edges;
            size_t _num_owned;
            size_t _global_offset;
            size_t _max_degree;
            const size_t SizeEdgeData;
            const size_t SizeNodeData;
            unsigned int * outgoing_index;
            unsigned int * neighbors;
            NodeDataType * node_data;
            EdgeDataType * edge_data;
            //BufferWrapperImpl<unsigned int> bufferWrapper;

            std::vector<UserNodeDataType * > dirtyData;
            //GPU Data
            cl_mem gpu_struct_ptr;
            _CL_LC_Graph_GPU gpu_wrapper;
            cl_mem gpu_meta;

            //Read a single node-data value from the device.
            //Blocking call.
            void read_node_data_impl(NodeIterator & it, NodeDataType & data){
               int err;
               auto id = *it;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
               err = clEnqueueReadBuffer(queue, gpu_wrapper.node_data, CL_TRUE, sizeof(NodeDataType)*(id), sizeof(NodeDataType), &data, 0, NULL, NULL);
               fprintf(stderr,  "Data read[%d], offset=%d :: val=%d \n", id, sizeof(NodeDataType) * id, data.dist_current);
               CHECK_CL_ERROR(err, "Error reading node-data 0\n");

            }
            //Write a single node-data value to the device.
            //Blocking call.
            void write_node_data_impl(NodeIterator & it, NodeDataType & data){
               int err;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
               err = clEnqueueWriteBuffer(queue, gpu_wrapper.node_data, CL_TRUE, sizeof(NodeDataType)*(*it), sizeof(NodeDataType), &data, 0, NULL, NULL);
               fprintf(stderr,  "Data read[%d], offset=%d :: val=%d \n", *it, sizeof(NodeDataType) * (*it), data.dist_current);
               CHECK_CL_ERROR(err, "Error writing node-data 0\n");
            }
         public:
            /*
             * Go over all the wrapper instances created and write them to device.
             * Also, cleanup any automatically destroyed wrapper instances.
             * */
            void sync_outstanding_data(){
               fprintf(stderr, "Writing to device %u\n", dirtyData.size());
               std::vector<UserNodeDataType *> purged;
               for(auto i : dirtyData){
                  if(i && i->isDirty){
                     fprintf(stderr, "Writing to device %d\n", *(i->id));
                     write_node_data_impl(i->id, i->cached_data);
                     purged.push_back(i);
                  }
               }
               dirtyData.clear();
               dirtyData.insert(dirtyData.begin(), purged.begin(), purged.end());
            }
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
            CL_LC_Graph() :
            SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)), SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
//      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge data.", (int) SizeNodeData, (int) SizeEdgeData);
               fprintf(stderr, "Loading devicegraph with copy-optimization.\n");
               _max_degree = _num_nodes = _num_edges = 0;
               _num_owned = _global_offset = 0;
               outgoing_index=neighbors=nullptr;
               node_data =nullptr;
               edge_data = nullptr;
               gpu_struct_ptr = gpu_meta= nullptr;

            }
            template<typename HGraph>
            CL_LC_Graph(std::string filename, unsigned myid, unsigned numHost) :
                        SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)), SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
            //      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge data.", (int) SizeNodeData, (int) SizeEdgeData);
                           fprintf(stderr, "Loading device-graph [%s] with copy-optimization.\n", filename.c_str());
                           _max_degree = _num_nodes = _num_edges = 0;
                           _num_owned = _global_offset = 0;
                           outgoing_index=neighbors=nullptr;
                           node_data =nullptr;
                           edge_data = nullptr;
                           gpu_struct_ptr = gpu_meta= nullptr;
//                           hGraph<NodeDataType, EdgeDataType> g(filename, myid, numHost);
                           HGraph g(filename, myid, numHost);
                           fprintf(stderr, "Loading from hgraph\n");
                           load_from_galois(g);
                        }
//                        CL_LC_Graph(std::string filename, unsigned myid, unsigned numHost) :
//                                    SizeEdgeData(sizeof(EdgeDataType) / sizeof(unsigned int)), SizeNodeData(sizeof(NodeDataType) / sizeof(unsigned int)) {
//                        //      fprintf(stderr, "Created LC_LinearArray_Graph with %d node %d edge data.", (int) SizeNodeData, (int) SizeEdgeData);
//                                       fprintf(stderr, "Loading device-graph [%s] with copy-optimization.\n", filename.c_str());
//                                       _max_degree = _num_nodes = _num_edges = 0;
//                                       _num_owned = _global_offset = 0;
//                                       outgoing_index=neighbors=nullptr;
//                                       node_data =nullptr;
//                                       edge_data = nullptr;
//                                       gpu_struct_ptr = gpu_meta= nullptr;
//                                       hGraph<NodeDataType, EdgeDataType> g(filename, myid, numHost);
////                                       HGraph g(filename, myid, numHost);
//                                       fprintf(stderr, "Loading from hgraph\n");
//                                       load_from_galois(g);
//                                    }
            template<typename HGraph>
            void load_from_hgraph(HGraph/*<NodeDataType,EdgeDataType>*/ & hg) {
                           fprintf(stderr, "Loading device-graph from hGraph with copy-optimization.\n");
                           _max_degree = _num_nodes = _num_edges = 0;
                           _num_owned = _global_offset = 0;
                           outgoing_index=neighbors=nullptr;
                           node_data =nullptr;
                           gpu_struct_ptr = gpu_meta= nullptr;
                           fprintf(stderr, "Loading from hgraph\n");
                           load_from_galois(hg);
                        }
            template<typename GaloisGraph>
            void load_from_galois(GaloisGraph & ggraph) {
               typedef typename GaloisGraph::GraphNode GNode;
               const size_t gg_num_nodes = ggraph.size();
               const size_t gg_num_edges = ggraph.sizeEdges();
               _num_owned = ggraph.getNumOwned();
               _global_offset = ggraph.getGlobalOffset();
               init(gg_num_nodes, gg_num_edges);
               int edge_counter = 0;
               int node_counter = 0;
               outgoing_index[0] = 0;
               int min_node=0, max_node=0;
               for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
                  int src_node = *n;
                  memcpy(&node_data[src_node], &ggraph.getData(*n), sizeof(NodeDataType));
                  //TODO - RK - node_data[src_node] = ggraph.getData(*n);
                  outgoing_index[src_node] = edge_counter;
                  assert(src_node >=0 && src_node< gg_num_nodes);
                  min_node = std::min(src_node, min_node);
                  max_node = std::max(src_node, max_node);
                  for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
                     GNode dst = ggraph.getEdgeDst(*nbr);
                     min_node = std::min((int)dst, min_node);
                     max_node = std::max((int)dst, max_node);
                     neighbors[edge_counter] = dst;
                     edge_data[edge_counter] = ggraph.getEdgeData(*nbr);
                     assert(dst>=0 && dst < gg_num_nodes);
                     edge_counter++;
                  }
               }
               while(node_counter!=gg_num_nodes)
                  outgoing_index[node_counter++] = edge_counter;
               outgoing_index[gg_num_nodes] = edge_counter;
               fprintf(stderr, "Min :: %d, Max :: %d \n", min_node, max_node);
               fprintf(stderr, "Nodes :: %d, Owned,  %zu, Counter %d \n", gg_num_nodes, _num_owned, node_counter);
               assert(edge_counter == gg_num_edges && "Failed to add all edges.");
               init_graph_struct();
               initialize_sync_buffers(ggraph);
               fprintf(stderr, "Loaded from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu, Owned=%lu, Offset=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType), _num_owned, _global_offset);
            }
            template<typename HGraph>
            void initialize_sync_buffers(HGraph & hgraph){
               //bufferWrapper.init(galois::Runtime::NetworkInterface::Num, hgraph.masterNodes, hgraph.slaveNodes);
            }
            /*
             * */
            template<typename GaloisGraph>
            void writeback_from_galois(GaloisGraph & ggraph) {
               typedef typename GaloisGraph::GraphNode GNode;
               const size_t gg_num_nodes = ggraph.size();
               const size_t gg_num_edges = ggraph.sizeEdges();
               int edge_counter = 0;
               int node_counter = 0;
               outgoing_index[0] = 0;
               for (auto n = ggraph.begin(); n != ggraph.end(); n++, node_counter++) {
                  int src_node = *n;
//               std::cout<<*n<<", "<<ggraph.getData(*n)<<", "<< getData()[src_node]<<"\n";
                  ggraph.getData(*n) = getData()[src_node];

               }
               fprintf(stderr, "Writeback from GaloisGraph [V=%zu,E=%zu,ND=%lu,ED=%lu, Owned=%lu, Offset=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType), _num_owned, _global_offset);
            }

            //TODO RK : fix - might not work with changes in interface.
            template<typename GaloisGraph>
            void load_from_galois(GaloisGraph & ggraph, int gg_num_nodes, int gg_num_edges, int num_ghosts) {
               typedef typename GaloisGraph::GraphNode GNode;
               init(gg_num_nodes + num_ghosts, gg_num_edges);
               fprintf(stderr, "Loading from GaloisGraph [%d,%d,%d].\n", (int) gg_num_nodes, (int) gg_num_edges, num_ghosts);
               int edge_counter = 0;
               int node_counter = 0;
               for (auto n = ggraph.begin(); n != ggraph.begin() + gg_num_nodes; n++, node_counter++) {
                  int src_node = *n;
                  getData(src_node) = ggraph.getData(*n);
                  outgoing_index[src_node] = edge_counter;
                  for (auto nbr = ggraph.edge_begin(*n); nbr != ggraph.edge_end(*n); ++nbr) {
                     GNode dst = ggraph.getEdgeDst(*nbr);
                     neighbors[edge_counter] = dst;
                     edge_data[edge_counter] = ggraph.getEdgeData(*nbr);
                     std::cout<<src_node<<" "<<dst<<" "<<edge_data[edge_counter]<<"\n";
                     edge_counter++;
                  }
               }
               for (; node_counter < gg_num_nodes + num_ghosts; node_counter++) {
                  outgoing_index[node_counter] = edge_counter;
               }
               outgoing_index[gg_num_nodes] = edge_counter;
               if (node_counter != gg_num_nodes)
               fprintf(stderr, "FAILED EDGE-COMPACTION :: %d, %d, %d\n", node_counter, gg_num_nodes, num_ghosts);
               assert(edge_counter == gg_num_edges && "Failed to add all edges.");
               init_graph_struct();
               fprintf(stderr, "Loaded from GaloisGraph [V=%d,E=%d,ND=%lu,ED=%lu].\n", gg_num_nodes, gg_num_edges, sizeof(NodeDataType), sizeof(EdgeDataType));
            }
            ~CL_LC_Graph() {
               deallocate();
            }
            uint32_t getNumOwned()const{
               return this->_num_owned;
            }
            uint64_t getGlobalOffset() const{
               return this->_global_offset;
            }
            const cl_mem &device_ptr(){
               return gpu_struct_ptr;
            }
            void read(const char * filename) {
               readFromGR(*this, filename);
            }
            const NodeDataType & getData(NodeIterator nid) const{
               return node_data[*nid];
            }
            NodeDataType & getData(NodeIterator nid) {
               return node_data[*nid];
            }
            const NodeDataWrapper getDataR(NodeIterator id)const {
               return NodeDataWrapper(this, id, false);
            }
            NodeDataWrapper getDataW(NodeIterator id){
//               fprintf(stderr, "GettingDataW for node :: %d \n", *id);
               return NodeDataWrapper(this, id, true);
            }
            unsigned int edge_begin(NodeIterator nid) {
               return outgoing_index[*nid];
            }
            unsigned int edge_end(NodeIterator nid) {
               return outgoing_index[*nid + 1];
            }
            unsigned int num_neighbors(NodeIterator node) {
               return outgoing_index[*node + 1] - outgoing_index[*node];
            }
            unsigned int & getEdgeDst(unsigned int eid) {
               return neighbors[eid];
            }
            EdgeDataType & getEdgeData(unsigned int eid) {
               return edge_data[eid];
            }
            NodeIterator begin(){
               return NodeIterator(0);
            }
            NodeIterator end(){
               return NodeIterator(_num_nodes);
            }
            size_t size() {
               return _num_nodes;
            }
            size_t sizeEdges() {
               return _num_edges;
            }
            size_t max_degree() {
               return _max_degree;
            }
            void init(size_t n_n, size_t n_e) {
               _num_nodes = n_n;
               _num_edges = n_e;
               fprintf(stderr, "Allocating NN: :%d,  , NE %d :\n", (int) _num_nodes, (int) _num_edges);
               node_data = new NodeDataType[_num_nodes];
               edge_data = new EdgeDataType[_num_edges];
               outgoing_index= new unsigned int [_num_nodes+1];
               neighbors = new unsigned int[_num_edges];
               allocate_on_gpu();
            }
            void allocate_on_gpu() {
               fprintf(stderr, "Buffer sizes : %d , %d \n", _num_nodes, _num_edges);
               int err;
               cl_mem_flags flags = 0; //CL_MEM_READ_WRITE  ;
               cl_mem_flags flags_read = 0;//CL_MEM_READ_ONLY ;

               gpu_wrapper.outgoing_index = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(cl_uint) *( _num_nodes + 1), nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 0\n");
               gpu_wrapper.node_data = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(NodeDataType) * _num_nodes, nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 1\n");
               gpu_wrapper.neighbors = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(cl_uint) * _num_edges, nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 2\n");
               gpu_wrapper.edge_data = clCreateBuffer(ctx->get_default_device()->context(), flags_read, sizeof(EdgeDataType) * _num_edges, nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 3\n");
               gpu_struct_ptr = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(cl_uint) * 32, nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 4\n");
               gpu_wrapper.num_nodes = _num_nodes;
               gpu_wrapper.num_edges= _num_edges;


               const int meta_buffer_size = 16;
               int  cpu_meta[meta_buffer_size];
               gpu_meta = clCreateBuffer(ctx->get_default_device()->context(), flags, sizeof(int) * meta_buffer_size, nullptr, &err);
               CHECK_CL_ERROR(err, "Error: clCreateBuffer of SVM - 5\n");
               cpu_meta[0] = _num_nodes;
               cpu_meta[1] =_num_edges;
               cpu_meta[2] =SizeNodeData;
               cpu_meta[3] =SizeEdgeData;
               cpu_meta[4] = _num_owned;
               cpu_meta[5] = _global_offset;
               err= clEnqueueWriteBuffer(ctx->get_default_device()->command_queue(), gpu_meta, CL_TRUE,0, sizeof(int)*meta_buffer_size, cpu_meta,NULL,0,NULL);
               CHECK_CL_ERROR(err, "Error: Writing META to GPU failed- 6\n");
               fprintf(stderr, "NUMOWNED :: %d \n", _num_owned);
//               err = clReleaseMemObject(gpu_meta);
//               CHECK_CL_ERROR(err, "Error: Releasing meta buffer.- 7\n");
               init_graph_struct();
            }
            void init_graph_struct() {
#if !PRE_INIT_STRUCT_ON_DEVICE
               this->copy_to_device();
               CL_Kernel init_kernel(getCLContext()->get_default_device());
               //TODO RK - cleanup this place. We are generating the code, so we do not need to 'build' kernels here.
//               size_t kernel_len = strlen(cl_wrapper_str_CL_LC_Graph) + strlen(init_kernel_str_CL_LC_Graph) + 1;
//               char * kernel_src = new char[kernel_len];
//               sprintf(kernel_src, "%s\n%s", cl_wrapper_str_CL_LC_Graph, init_kernel_str_CL_LC_Graph);
//               init_kernel.init_string(kernel_src, "initialize_graph_struct");
               init_kernel.init("app_header.h", "initialize_graph_struct");
//               init_kernel.set_arg_list(gpu_struct_ptr, gpu_meta, gpu_wrapper.node_data, gpu_wrapper.outgoing_index, gpu_wrapper.neighbors, gpu_wrapper.edge_data);
               init_kernel.set_arg_list_raw(gpu_struct_ptr, gpu_meta, gpu_wrapper.node_data, gpu_wrapper.outgoing_index, gpu_wrapper.neighbors, gpu_wrapper.edge_data);
               init_kernel.run_task();
#endif
            }
            ////////////##############################################################///////////
            ////////////##############################################################///////////
            //TODO RK - complete these.
            template<typename FnTy>
             void sync_push() {
               //bufferWrapper.sync_outgoing(*this);

#if 0
               char * synKernelString=" ";
               typedef std::pair<unsigned, typename FnTy::ValTy> MsgType;
               std::vector<MsgType> data;
               std::vector<size_t> peerHosts;
               size_t selfID;
               for(int peerID : peerHosts){
                  if(peerID != selfID){
                     std::pair<unsigned int, unsigned int> peerRange; // getPeerRange(peerID);
                     for(int i = peerRange.first; i<peerRange.second; ++i){
                        data.push_back(MsgType(i, getData(i)));
                     }
                  }
               }

#endif
            }
            template<typename FnTy>
             void sync_pull() {

            }
            ////////////##############################################################///////////
            ////////////##############################################################///////////
            void deallocate(void) {
               delete [] node_data;
               delete [] edge_data;
               delete [] outgoing_index;
               delete [] neighbors;

               cl_int err = clReleaseMemObject(gpu_wrapper.outgoing_index);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 0\n");
               err = clReleaseMemObject(gpu_wrapper.node_data );
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 1\n");
               err= clReleaseMemObject(gpu_wrapper.neighbors);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 2\n");
               err= clReleaseMemObject( gpu_wrapper.edge_data );
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 3\n");
               err= clReleaseMemObject(gpu_struct_ptr);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 4\n");
               err= clReleaseMemObject(gpu_meta);
               CHECK_CL_ERROR(err, "Error: clReleaseMemObject of SVM - 5\n");
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            //Done
            void copy_to_host(GRAPH_FIELD_FLAGS flags = GRAPH_FIELD_FLAGS::ALL) {
               int err;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
               if(flags & GRAPH_FIELD_FLAGS::OUT_INDEX){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.outgoing_index, CL_TRUE, 0, sizeof(cl_uint) * (_num_nodes + 1), outgoing_index, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to host 0\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::NODE_DATA){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.node_data, CL_TRUE, 0, sizeof(NodeDataType) * _num_nodes, node_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to host 1\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::NEIGHBORS){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.neighbors, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, neighbors, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to host 2\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::EDGE_DATA){
                  err = clEnqueueReadBuffer(queue, gpu_wrapper.edge_data, CL_TRUE, 0, sizeof(EdgeDataType) * _num_edges, edge_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to host 3\n");
               }
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            //Done
            void copy_to_device(GRAPH_FIELD_FLAGS flags = GRAPH_FIELD_FLAGS::ALL) {
               int err;
               cl_command_queue queue = ctx->get_default_device()->command_queue();
//               fprintf(stderr, "Buffer sizes : %d , %d \n", _num_nodes, _num_edges);
               if(flags & GRAPH_FIELD_FLAGS::OUT_INDEX) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.outgoing_index, CL_TRUE, 0, sizeof(cl_uint) * (_num_nodes + 1), outgoing_index, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to device 0\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::NODE_DATA) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.node_data, CL_TRUE, 0, sizeof(NodeDataType) * _num_nodes, node_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to device 1\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::NEIGHBORS) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.neighbors, CL_TRUE, 0, sizeof(cl_uint) * _num_edges, neighbors, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to device 2\n");
               }
               if(flags & GRAPH_FIELD_FLAGS::EDGE_DATA) {
                  err = clEnqueueWriteBuffer(queue, gpu_wrapper.edge_data, CL_TRUE, 0, sizeof(EdgeDataType) * _num_edges, edge_data, 0, NULL, NULL);
                  CHECK_CL_ERROR(err, "Error copying to device 3\n");
               }
               ////////////Initialize kernel here.
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_graph(void) {
               std::cout << "\n====Printing graph (" << _num_nodes << " , " << _num_edges << ")=====\n";
               for (size_t i = 0; i < _num_nodes; ++i) {
                  print_node(i);
                  std::cout << "\n";
               }
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_node(unsigned int idx) {
               if (idx < _num_nodes) {
                  std::cout << "N-" << idx << "(" << node_data[idx] << ")" << " :: [";
                  for (size_t i = outgoing_index[idx]; i < outgoing_index[idx + 1]; ++i) {
                     std::cout << " " << neighbors[i] << "(" << edge_data[i] << "), ";
                  }
                  std::cout << "]";
               }
               return;
            }
            /////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////
            void print_compact(void) {
               std::cout << "\nOut-index [";
               for (size_t i = 0; i < _num_nodes + 1; ++i) {
                  std::cout << " " << outgoing_index[i] << ",";
               }
               std::cout << "]\nNeigh[";
               for (size_t i = 0; i < _num_edges; ++i) {
                  std::cout << " " << neighbors[i] << ",";
               }
               std::cout << "]\nEData [";
               for (size_t i = 0; i < _num_edges; ++i) {
                  std::cout << " " << edge_data[i] << ",";
               }
               std::cout << "]";
            }
            //TODO
                       void reset_num_iter(uint32_t runNum){

                          }
                       size_t getGID(const NodeIterator &nt){
                          return 0;
                       }

         };//End CL_LC_Graph
      }//Namespace Graph
   }//Namespace OpenCL
}//Namespace Galois

#endif /* _GDIST_CL_LC_Graph_H_ */
