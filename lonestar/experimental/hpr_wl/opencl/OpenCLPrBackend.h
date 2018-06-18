/*
 * OpenCLPrBackend.h
 *
 *  Created on: Jul 9, 2015
 *      Author: rashid
 */

#ifndef GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_
#define GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_

#include "CLWrapper.h"
template <typename GraphType>

struct OPENCL_Context {
  typedef typename GraphType::NodeDataType NodeDataType;
  GraphType m_graph;
  galois::opencl::CL_Kernel kernel;
  galois::opencl::CL_Kernel wb_kernel;
  galois::opencl::Array<float>* aux_array;

  // this one will have a different scheme than the c++ and the cuda.
  // since here we have an array. Instead of swapping lists I'll have an array
  // with the same size as the number of nodes. Then all nodes that haven't
  // converged yet, are marked as one. Basically, 1 marks the nodes that still
  // have to be processed and work size will count the amount of work left
  galois::opencl::Array<int>* wl;
  // galois::opencl::Array<int> *work_size;
  // a simple int wasn't working, so i changed it to a vector of size 1
  // int size_work; //stores the amount of work at each iteration
  // galois::opencl::Array<int> *wl2;
  OPENCL_Context() : aux_array(nullptr) {
    // not sure what this does, should wl be added here?
  }

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
    galois::opencl::CL_Kernel init_all, init_nout;

    aux_array = new galois::opencl::Array<float>(m_graph.num_nodes());
    // initialize work list and work size
    wl = new galois::opencl::Array<int>(m_graph.num_nodes());

    // work_size = new galois::opencl::Array<int>(1);
    // not quite the  best solution but at least definying a array of ints with
    // a single position seems to work

    kernel.init("pagerank_kernel.cl", "pagerank");
    wb_kernel.init("pagerank_kernel.cl", "writeback");
    init_nout.init("pagerank_kernel.cl", "initialize_nout");
    init_all.init("pagerank_kernel.cl", "initialize_all");
    m_graph.copy_to_device();

    init_all.set_work_size(m_graph.num_nodes());
    init_nout.set_work_size(m_graph.num_nodes());
    wb_kernel.set_work_size(num_items);
    kernel.set_work_size(num_items);

    init_nout.set_arg_list(&m_graph, aux_array);
    init_all.set_arg_list(&m_graph, aux_array, wl); //, size_work);
    kernel.set_arg_list(&m_graph, aux_array, wl);   //, size_work);
    wb_kernel.set_arg_list(&m_graph, aux_array);
    int num_nodes = m_graph.num_nodes();

    init_nout.set_arg(2, sizeof(cl_int), &num_items);
    init_all.set_arg(2, sizeof(cl_int) * m_graph.num_nodes(),
                     wl); // added this line
    wb_kernel.set_arg(2, sizeof(cl_int), &num_items);
    kernel.set_arg(2, sizeof(cl_int) * m_graph.num_nodes(),
                   wl);                            // added this one
    kernel.set_arg(3, sizeof(cl_int), &num_items); // changed this to a 3

    init_all();
    init_nout();
    m_graph.copy_to_host();
  }

  int operator()(int num_items) {
    m_graph.copy_to_device();
    kernel();
    wb_kernel();
    std::cout << "Amount of work: " << m_graph.getWork() << "\n";
    m_graph.copy_to_host();
    // return work_size[0];
    return getWorkLeft();
  }

  NodeDataType& getData(unsigned ID) { return m_graph.getData(ID); }

  int getWorkLeft() { return m_graph.getWork(); }
};

#endif /* GDIST_EXP_APPS_HPR_OPENCL_OPENCLPRBACKEND_H_ */
