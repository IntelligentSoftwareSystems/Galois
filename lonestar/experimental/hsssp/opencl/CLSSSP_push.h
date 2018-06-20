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

/*
 * OpenCLPrBackend.h
 *
 *  Created on: Jul 9, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_
#define GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_

#include "opencl/CLWrapper.h"
/************************************************************************************
 * OpenCL PageRank operator implementation.
 * Uses two kernels for the updates and one kernel for initialization.
 * In order to support BSP semantics, an aux_array is created which will
 * be used to buffer the writes in 'kernel'. These updates will be
 * written to the node-data in the 'writeback' kernel.
 *************************************************************************************/

template <typename GraphType>
struct OPENCL_Context {
  typedef typename GraphType::NodeDataType NodeDataType;
  GraphType m_graph;
  galois::opencl::CL_Kernel kernel;
  galois::opencl::CL_Kernel wb_kernel;
  galois::opencl::Array<int>* meta_array;
  OPENCL_Context() : meta_array(nullptr) {}

  GraphType& get_graph() { return m_graph; }
  template <typename GaloisGraph>
  void loadGraphNonCPU(GaloisGraph& g, size_t numOwned, size_t numEdges,
                       size_t numReplicas) {
    m_graph.load_from_galois(g, numOwned, numEdges, numReplicas);
  }
  template <typename GaloisGraph>
  void loadGraphNonCPU(GaloisGraph& g) {
    m_graph.load_from_galois(g.g, g.numOwned, g.numEdges,
                             g.numNodes - g.numOwned);
  }
  void init(int num_items, int num_inits) {
    galois::opencl::CL_Kernel init_nodes;
    init_nodes.init("sssp_kernel.cl", "initialize_nodes");
    meta_array = new galois::opencl::Array<int>(16);
    kernel.init("sssp_kernel.cl", "sssp");
    wb_kernel.init("sssp_kernel.cl", "writeback");
    m_graph.copy_to_device();

    wb_kernel.set_work_size(num_items);
    kernel.set_work_size(num_items);
    init_nodes.set_work_size(num_items);

    kernel.set_arg_list(&m_graph, meta_array);
    wb_kernel.set_arg_list(&m_graph);
    init_nodes.set_arg_list(&m_graph);
    int num_nodes = m_graph.num_nodes();

    wb_kernel.set_arg(1, sizeof(cl_int), &num_items);
    kernel.set_arg(2, sizeof(cl_int), &num_items);
    init_nodes.set_arg(1, sizeof(cl_int), &num_items);
    init_nodes();

    m_graph.copy_to_host();
    //      meta_array->host_ptr()[0] = hasChanged;
    meta_array->copy_to_host();
    //      hasChanged = meta_array->host_ptr()[0];
  }
  void operator()(int num_items, bool& hasChanged) {
    meta_array->host_ptr()[0] = hasChanged;
    meta_array->copy_to_device();
    //      fprintf(stderr, "CL-Backend:: Src distance :%d %d /",
    //      graph.getData(0).dist[0],graph.getData(0).dist[1]);
    m_graph.copy_to_device();
    kernel();
    wb_kernel();
    m_graph.copy_to_host();
    meta_array->copy_to_host();
    hasChanged = meta_array->host_ptr()[0];
    //      fprintf(stderr, "[Changed=%d, %d, %d ]\n", hasChanged,
    //      graph.getData(0).dist[0],graph.getData(0).dist[1]);
  }
  NodeDataType& getData(unsigned ID) { return m_graph.getData(ID); }
};
#endif /* GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_ */
