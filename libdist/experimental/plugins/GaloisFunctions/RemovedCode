
Located in getSyncer
#if 0
    s << "\tstruct Syncer_" << struct_type << counter << " {\n";
    s << "\t\tstatic " << i.VAL_TYPE <<" extract(uint32_t node_id, const " << i.NODE_TYPE << " node) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) return " << "get_node_" << i.FIELD_NAME <<  "_cuda(cuda_ctx, node_id);\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn " << "node." << i.FIELD_NAME <<  ";\n"
      << "\t\t}\n";
    s << "\t\tstatic bool extract_reset_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, " << i.VAL_TYPE << " *y, size_t *s, DataCommMode *data_mode) {\n";
    if (!i.RESET_VAL_EXPR.empty()) {
      s << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
        << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_reset_node_" << i.FIELD_NAME 
        <<  "_cuda(cuda_ctx, from_id, b, o, y, s, data_mode, " << i.RESET_VAL_EXPR << "); return true; }\n"
        << "\t\t\tassert (personality == CPU);\n"
        << "\t\t#endif\n"
        << "\t\t\treturn false;\n";
    } else {
      s << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
        << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_slave_node_" << i.FIELD_NAME 
        <<  "_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }\n"
        << "\t\t\tassert (personality == CPU);\n"
        << "\t\t#endif\n"
        << "\t\t\treturn false;\n";
    }
    s << "\t\t}\n";
    s << "\t\tstatic bool extract_reset_batch(unsigned from_id, " << i.VAL_TYPE << " *y) {\n";
    if (!i.RESET_VAL_EXPR.empty()) {
      s << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
        << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_reset_node_" << i.FIELD_NAME 
        <<  "_cuda(cuda_ctx, from_id, y, " << i.RESET_VAL_EXPR << "); return true; }\n"
        << "\t\t\tassert (personality == CPU);\n"
        << "\t\t#endif\n"
        << "\t\t\treturn false;\n";
    } else {
      s << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
        << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_slave_node_" << i.FIELD_NAME 
        <<  "_cuda(cuda_ctx, from_id, y); return true; }\n"
        << "\t\t\tassert (personality == CPU);\n"
        << "\t\t#endif\n"
        << "\t\t\treturn false;\n";
    }
    s << "\t\t}\n";
    s << "\t\tstatic void reduce (uint32_t node_id, " << i.NODE_TYPE << " node, " << i.VAL_TYPE << " y) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) " << i.REDUCE_OP_EXPR << "_node_" << i.FIELD_NAME <<  "_cuda(cuda_ctx, node_id, y);\n"
      << "\t\t\telse if (personality == CPU)\n"
      << "\t\t#endif\n"
      << "\t\t\t\t{ galois::" << i.REDUCE_OP_EXPR << "(node." << i.FIELD_NAME  << ", y); }\n"
      << "\t\t}\n";
    s << "\t\tstatic bool reduce_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, " << i.VAL_TYPE << " *y, size_t s, DataCommMode data_mode) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) { " << "batch_" << i.REDUCE_OP_EXPR << "_node_" << i.FIELD_NAME 
      <<  "_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn false;\n"
      << "\t\t}\n";
    s << "\t\tstatic void reset (uint32_t node_id, " << i.NODE_TYPE << " node ) {\n";
    if (!i.RESET_VAL_EXPR.empty()) {
      s << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
        << "\t\t\tif (personality == GPU_CUDA) " << "set_node_" << i.FIELD_NAME <<  "_cuda(cuda_ctx, node_id, " << i.RESET_VAL_EXPR << ");\n"
        << "\t\t\telse if (personality == CPU)\n"
        << "\t\t#endif\n"
        << "\t\t\t\t{ node." << i.FIELD_NAME << " = " << i.RESET_VAL_EXPR << "; }\n";
    }
    s << "\t\t}\n";
    s << "\t\ttypedef " << i.VAL_TYPE << " ValTy;\n"
      << "\t};\n";
#endif


Located in getSyncerPull
#if 0
    s << "\tstruct SyncerPull_" << struct_type << counter << " {\n";
    s << "\t\tstatic " << i.VAL_TYPE <<" extract(uint32_t node_id, const " << i.NODE_TYPE << " node) {\n"
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) return " << "get_node_" << i.FIELD_NAME <<  "_cuda(cuda_ctx, node_id);\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn " << "node." << i.FIELD_NAME <<  ";\n"
      << "\t\t}\n";
    s << "\t\tstatic bool extract_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, " << i.VAL_TYPE << " *y, size_t *s, DataCommMode *data_mode) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_node_" << i.FIELD_NAME 
      <<  "_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn false;\n"
      << "\t\t}\n";
    s << "\t\tstatic bool extract_batch(unsigned from_id, " << i.VAL_TYPE << " *y) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) { " << "batch_get_node_" << i.FIELD_NAME 
      <<  "_cuda(cuda_ctx, from_id, y); return true; }\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn false;\n"
      << "\t\t}\n";
    s << "\t\tstatic void setVal (uint32_t node_id, " << i.NODE_TYPE << " node, " << i.VAL_TYPE << " y) " << "{\n"
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) " << "set_node_" << i.FIELD_NAME <<  "_cuda(cuda_ctx, node_id, y);\n"
      << "\t\t\telse if (personality == CPU)\n"
      << "\t\t#endif\n"
      << "\t\t\t\tnode." << i.FIELD_NAME << " = y;\n"
      << "\t\t}\n";
    s << "\t\tstatic bool setVal_batch(unsigned from_id, unsigned long long int *b, unsigned int *o, " << i.VAL_TYPE << " *y, size_t s, DataCommMode data_mode) {\n" 
      << "\t\t#ifdef __GALOIS_HET_CUDA__\n"
      << "\t\t\tif (personality == GPU_CUDA) { " << "batch_set_node_" << i.FIELD_NAME 
      <<  "_cuda(cuda_ctx, from_id, b, o, y, s, data_mode); return true; }\n"
      << "\t\t\tassert (personality == CPU);\n"
      << "\t\t#endif\n"
      << "\t\t\treturn false;\n"
      << "\t\t}\n";
    s << "\t\ttypedef " << i.VAL_TYPE << " ValTy;\n"
      << "\t};\n";
#endif

Located in ForEach handler
#if 0
            stringstream SSAfter;
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_push"){
                SSAfter <<"\n\t" << "_graph.sync_forward<Syncer_" << i << ", SyncerPull_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                //SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_pull"){
                SSAfter <<"\n\t" << "_graph.sync_forward<Syncer_vertexCut_" << i << ", SyncerPull_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                //SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
            }
#endif


#if 0
            for (unsigned i = 0; i < write_set_vec_PUSH.size(); i++) {
              SSAfter <<"\n" << "\t\t_graph.sync_push<Syncer_" << i << ">" <<"(\"" << OperatorStructName << "\");\n";
              rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              SSAfter.str(string());
              SSAfter.clear();
            }

            //For sync Pull and push for vertex cut
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_pull"){
                SSAfter << "\n" << "if(_graph.is_vertex_cut()) {";
                SSAfter <<"\n\t" << "_graph.sync_push<Syncer_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
            }

            //For sync Pull
            for (unsigned i = 0; i < write_set_vec_PULL.size(); i++) {
              SSAfter <<"\n" << "\t\t_graph.sync_pull<SyncerPull_" << i << ">" <<"(\"" << OperatorStructName << "\");\n";
              rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              SSAfter.str(string());
              SSAfter.clear();
            }

            //For sync Pull and push for vertex cut
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_push"){
                SSAfter << "\n" << "if(_graph.is_vertex_cut()) {";
                SSAfter <<"\n\t" << "_graph.sync_pull<SyncerPull_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
            }
#endif


Located in FunctionCallHandler
#if 0
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_push"){
                //SSAfter <<"\n\t" << "_graph.sync_forward<Syncer_" << i << ", SyncerPull_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                SSAfter <<"\n\t" << "_graph.sync_forward<Syncer_" << i << ", SyncerPull_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");\n";
                //SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_pull"){
                SSAfter <<"\n\t" << "_graph.sync_forward<Syncer_vertexCut_" << i << ", SyncerPull_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                //SSAfter << "\n}\n";
                rewriter.InsertText(ST_main, SSAfter.str(), true, true);
              }
              SSAfter.str(string());
              SSAfter.clear();
            }
#endif


#if 0
            for (unsigned i = 0; i < write_set_vec_PUSH.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              //SSAfter <<"\n" <<write_set_vec_PUSH[i].GRAPH_NAME<< ".sync_push<Syncer_" << i << ">" <<"();\n";
              SSAfter <<"\n" << "_graph.sync_push<Syncer_" << i << ">" <<"(\"" << OperatorStructName << "\");\n";
              rewriter.InsertText(ST, SSAfter.str(), true, true);
            }


            //For sync Pull and push for vertex cut
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_pull"){
                SSAfter << "\n" << "if(_graph.is_vertex_cut()) {";
                SSAfter <<"\n\t" << "_graph.sync_push<Syncer_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                SSAfter << "\n}\n";
                rewriter.InsertText(ST, SSAfter.str(), true, true);
              }
            }

            //For sync Pull
            for (unsigned i = 0; i < write_set_vec_PULL.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              SSAfter <<"\n" << "_graph.sync_pull<SyncerPull_" << i << ">" <<"(\"" << OperatorStructName << "\");\n";
              rewriter.InsertText(ST, SSAfter.str(), true, true);
            }

            //For sync Pull and push for vertex cut
            for (unsigned i = 0; i < write_set_vec_PUSH_PULL.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              if(write_set_vec_PUSH_PULL[i].SYNC_TYPE == "sync_push"){
                SSAfter << "\n" << "if(_graph.is_vertex_cut()) {";
                SSAfter <<"\n\t" << "_graph.sync_pull<SyncerPull_vertexCut_" << i << ">" <<"(\"" << OperatorStructName << "\");";
                SSAfter << "\n}\n";
                rewriter.InsertText(ST, SSAfter.str(), true, true);
              }
            }
#endif


Located in handlers (unused though)
          // Vector to store read and write set.
          //vector<pair<string, string>>read_set_vec;
          //vector<pair<string,string>>write_set_vec;
          //vector<string>write_setGNode_vec;
          //string GraphNode;

