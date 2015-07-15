#if _HETERO_DEBUG_
   std::cout << g.id << " graph loaded\n";
#endif
   //OPENCL dPageRank dOp;

   //local initialization
   if (personality == CPU) {
//      InitializeGraph::go(g.g, g.numOwned);
      InitializeGraph::go(g.g, g.numOwned);
   } else if (personality == GPU_CUDA) {
      initialize_node_cuda(pr_cuda_ctx);
    } else if (personality == GPU_OPENCL) {
       pr_cl_ctx.init(g.numOwned, g.numNodes);
    }

#if _HETERO_DEBUG_
   std::cout << g.id << " initialized\n";
#endif
   barrier.wait();

   // send out partial contributions for nout from local -> ghost 
   send_nout_shared(net, g);
   barrier.wait();

   // send final nout values to remote replicas
#if _HETERO_DEBUG_
   std::cout << "["<<my_host_id<< "]:ask for ghost cell attrs\n";
#endif
   send_nout(net, g);
   barrier.wait();

   for (int i = 0; i < maxIterations; ++i) {
#if _HETERO_DEBUG_
      std::cout << "Starting " << i << "\n";
#endif
      //communicate ghost cells
      send_pr(net, g);
      barrier.wait();
#if _HETERO_DEBUG_
      std::cout << "Starting PR\n";
#endif
      //Do pagerank
      switch (personality) {
      case CPU:
         PageRank::go(g.g, g.numOwned);
         WriteBack::go(g.g, g.numOwned);
         break;
       case GPU_OPENCL:
          pr_cl_ctx(g.numOwned);
          break;
      case GPU_CUDA:
         pr_cuda(pr_cuda_ctx);
         break;
      default:
         break;
      }
      barrier.wait();
   }

   //Final synchronization to ensure that all the nodes are updated.
   send_pr(net, g);
   barrier.wait();

   if (verify) {
      std::stringstream ss;
      ss << personality_str(personality) << "_" << my_host_id << "_of_" << Galois::Runtime::NetworkInterface::Num << "_page_ranks.csv";
      std::ofstream out_file(ss.str());
      switch (personality) {
      case CPU: {
         for (auto n = g.g.begin(); n != g.g.begin() + g.numOwned; ++n) {
            out_file << *n + g.g_offset << ", " << g.g.getData(*n).pr << ", " << g.g.getData(*n).nout << "\n";
         }
         break;
      }
       case GPU_OPENCL: {
          for (int n = 0; n < g.numOwned; ++n) {
             out_file << n + g.g_offset << ", " << pr_cl_ctx.getData(n).pr << ", " << pr_cl_ctx.getData(n).nout << "\n";
          }
          break;
       }
      case GPU_CUDA:
         for (int i = 0; i < g.numOwned; i++) {
            out_file << i + g.g_offset << ", " << get_PRNode_pr_CUDA(pr_cuda_ctx, i) << ", " << get_PRNode_nout_CUDA(pr_cuda_ctx, i) << "\n";
         }
         break;
      }
      out_file.close();
   }
   std::cout.flush();
   return 0;
