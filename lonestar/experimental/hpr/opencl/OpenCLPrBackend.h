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

#include <opencl/CLWrapper.h>

template <typename GraphType>
struct OPENCL_Context {
  typedef typename GraphType::NodeDataType NodeDataType;
  GraphType m_graph;
  galois::opencl::CL_Kernel kernel;
  galois::opencl::CL_Kernel wb_kernel;
  OPENCL_Context() {}

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
  std::string self_directory() {
    char buffer[1024];
    size_t len = readlink("/proc/self/exe", buffer, 1024);
    if (len != -1) {
      char* filename = dirname(buffer);
      return std::string(filename);
    }
    assert(false);
    return std::string(buffer);
  }
  void init(int num_items, int num_inits) {
    galois::opencl::CL_Kernel init_all, init_nout;
    kernel.init("pagerank_kernel.cl", "pagerank");
    wb_kernel.init("pagerank_kernel.cl", "writeback");
    init_nout.init("pagerank_kernel.cl", "initialize_nout");
    init_all.init("pagerank_kernel.cl", "initialize_all");
    m_graph.copy_to_device();

    init_all.set_work_size(m_graph.num_nodes());
    init_nout.set_work_size(m_graph.num_nodes());
    wb_kernel.set_work_size(num_items);
    kernel.set_work_size(num_items);

    init_nout.set_arg_list(&m_graph);
    init_all.set_arg_list(&m_graph);
    kernel.set_arg_list(&m_graph);
    wb_kernel.set_arg_list(&m_graph);
    int num_nodes = m_graph.num_nodes();

    init_nout.set_arg(1, sizeof(cl_int), &num_items);
    wb_kernel.set_arg(1, sizeof(cl_int), &num_items);
    kernel.set_arg(1, sizeof(cl_int), &num_items);

    init_all();
    init_nout();
    m_graph.copy_to_host();
  }
  void operator()(int num_items) {
    m_graph.copy_to_device();
    kernel();
    wb_kernel();
    m_graph.copy_to_host();
  }
  NodeDataType& getData(unsigned ID) { return m_graph.getData(ID); }
};
#endif /* GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_ */
