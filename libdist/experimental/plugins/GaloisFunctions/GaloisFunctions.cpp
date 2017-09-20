/** clang analysis plugin for Galois programs -*- C++ -*-
 *
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * TODO
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */
#include <sstream>
#include <climits>
#include <vector>
#include <algorithm>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

/**
 *  Matchers
 */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/**
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

#define __GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__ "local_"
#define __GALOIS_ACCUMULATOR_TYPE__ "galois::DGAccumulator"

namespace {
  class GaloisFunctionsVisitor :
    public RecursiveASTVisitor<GaloisFunctionsVisitor> {
  private:
    ASTContext* astContext;
  public:
    explicit GaloisFunctionsVisitor(CompilerInstance *CI) :
      astContext(&(CI->getASTContext())) {}

    virtual ~GaloisFunctionsVisitor() {}

    virtual bool VisitCXXRecordDecl(CXXRecordDecl* Dec) {
      Dec->dump();
      return true;
    }

    virtual bool VisitFunctionDecl(FunctionDecl* func) {
      //std::string funcName = func->getNameInfo().getName().getAsString();
      std::string funcName = func->getNameAsString();

      // foo finding test
      if (funcName == "foo"){
        llvm::errs() << "Found" << "\n";
      }

      return true;
    }
  };

  class NameSpaceHandler : public MatchFinder::MatchCallback {
    //CompilerInstance &Instance;
    //clang::LangOptions& langOptions; //Instance.getLangOpts();
  public:
    NameSpaceHandler(clang::LangOptions &langOptions) {}
    virtual void run(const MatchFinder::MatchResult &results) {
      llvm::errs() << "It is coming here\n"
                   << results.Nodes.getNodeAs<clang::NestedNameSpecifier>(
                        "galoisLoop"
                      )->getAsNamespace()->getIdentifier()->getName();
      const NestedNameSpecifier* NSDecl = 
        results.Nodes.getNodeAs<clang::NestedNameSpecifier>("galoisLoop");

      if (NSDecl != NULL) {
        llvm::errs() << "Found Galois Loop\n";
        //NSDecl->dump(langOptions);
      }
    }
  };

  class ForStmtHandler : public MatchFinder::MatchCallback {
    public:
      virtual void run(const MatchFinder::MatchResult &results) {
        if (const ForStmt* FS = 
              results.Nodes.getNodeAs<clang::ForStmt>("forLoop")) {
          llvm::errs() << "for loop found\n";
          FS->dump();
        }
      }
  };

  struct Write_set {
    string GRAPH_NAME;
    string NODE_TYPE;
    string FIELD_TYPE;
    string FIELD_NAME;
    string REDUCE_OP_EXPR;
    string VAL_TYPE;
    string RESET_VAL_EXPR;
    string SYNC_TYPE;
    string READ_FLAG;
    string WRITE_FLAG;
  };

  struct Write_set_PULL {
    string GRAPH_NAME;
    string NODE_TYPE;
    string FIELD_TYPE;
    string FIELD_NAME;
    string VAL_TYPE;
  };

  /**
   * Given an input, upper-cases + returns
   */
  std::string stringToUpper(std::string input){
    std::transform(input.begin(), input.end(), input.begin(), ::toupper);
    return input;
  }

  /**
   * Creates the reduce sync structure macro given a write set containing the 
   * variable to create the sync structure for
   */
  std::string getSyncer(const Write_set& i) {
    std::stringstream s;
    s << "GALOIS_SYNC_STRUCTURE_REDUCE_" << stringToUpper(i.REDUCE_OP_EXPR) << 
         "(" << i.FIELD_NAME << " , " << i.VAL_TYPE << ");\n";
    return s.str();
  }


  /**
   * Create reduce sync structure macro given write set with variable 
   * information + saves names of flags structure and reduce structure to a map.
   */
  typedef std::map<std::string, std::vector<std::string>> SyncCall_map;
          //std::map<std::string, std::vector<std::string>> syncCall_map;
  std::string getSyncer(const Write_set& i, SyncCall_map& syncCall_map) {
    std::stringstream s;
    s << "GALOIS_SYNC_STRUCTURE_REDUCE_" << stringToUpper(i.REDUCE_OP_EXPR) << 
         "(" << i.FIELD_NAME << " , " << i.VAL_TYPE << ");\n";

    std::string str_readFlags = " Flags_" + i.FIELD_NAME + " , " + i.READ_FLAG;
    std::string str_reduce_call = " Reduce_" + i.REDUCE_OP_EXPR + "_" + 
                                  i.FIELD_NAME + " ";
    syncCall_map[i.FIELD_NAME].push_back(str_readFlags);
    syncCall_map[i.FIELD_NAME].push_back(str_reduce_call);

    return s.str();
  }

  // TODO separate out bitset macro from syncer pull call

  /**
   * Construct the broadcast and bitset macros given writeset with field info
   * TODO getsyncerpull is misleading name since bitset is included
   */
  std::string getSyncerPull(const Write_set& i) {
    std::stringstream s;
    s << "GALOIS_SYNC_STRUCTURE_BROADCAST" << "(" << i.FIELD_NAME 
      << " , " << i.VAL_TYPE << ");\n";
    s << "GALOIS_SYNC_STRUCTURE_BITSET" << "(" << i.FIELD_NAME << ");\n";

    return s.str();
  }


  /**
   * Construct the broadcast and bitset macros given writeset with field info +
   * save the structure names into map
   * TODO getsyncerpull is misleading name since bitset is included
   */
  std::string getSyncerPull(const Write_set& i, SyncCall_map& syncCall_map) {
    std::stringstream s;
    s << "GALOIS_SYNC_STRUCTURE_BROADCAST" << "(" << i.FIELD_NAME << " , " 
      << i.VAL_TYPE << ");\n";
    s << "GALOIS_SYNC_STRUCTURE_BITSET" << "(" << i.FIELD_NAME << ");\n";

    std::string str_broadcast_call = " Broadcast_" + i.FIELD_NAME + " ";
    std::string str_bitset = " Bitset_" + i.FIELD_NAME + " ";
    syncCall_map[i.FIELD_NAME].push_back(str_broadcast_call);
    syncCall_map[i.FIELD_NAME].push_back(str_bitset);

    return s.str();
  }

  /**
   * Handles sync additions/other things for galois::for_each calls
   */
  // TODO(Loc) check this for correctness/sanity
  class FunctionForEachHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;

    public:
      FunctionForEachHandler(Rewriter &rewriter) : rewriter(rewriter) {}

      virtual void run(const MatchFinder::MatchResult &results) {
        llvm::errs() << "for_each found\n";
        const CallExpr* callFS = 
          results.Nodes.getNodeAs<clang::CallExpr>("galoisLoop_forEach");
        auto callerFS = results.Nodes.getNodeAs<clang::CXXRecordDecl>("class");

        //callerFS->dump();
        if (callFS) {
          std::string OperatorStructName = callerFS->getNameAsString();
          llvm::errs() << " NAME OF THE STRUCT : " << 
                          OperatorStructName << "\n";
          llvm::errs() << "It is coming here for_Each \n" << 
                          callFS->getNumArgs() << "\n";



          vector<Write_set> write_set_vec_PUSH;
          vector<Write_set> write_set_vec_PULL;

          // TODO document this part

          // c++ pretty print policy
          clang::LangOptions LangOpts;
          LangOpts.CPlusPlus = true;
          clang::PrintingPolicy Policy(LangOpts);

          // deal with begin/end to determine if loop goes over 1 source
          assert(callFS->getNumArgs() > 1);
          string str_begin_arg;
          llvm::raw_string_ostream s_begin(str_begin_arg);
          callFS->getArg(0)->printPretty(s_begin, 0, Policy);
          string begin = s_begin.str();
          string str_end_arg;
          llvm::raw_string_ostream s_end(str_end_arg);
          callFS->getArg(1)->printPretty(s_end, 0, Policy);
          string end = s_end.str();
          string single_source;

          // TODO: use type comparison instead
          if (end.find(".end()") == string::npos) {
            single_source = begin;
            begin = "_graph.getLID(" + single_source + ")";
            end = begin + "+1";
            // use begin and end only if _graph_isOwned(single_source)
          } else {
            assert(begin.compare("_graph.begin()") == 0);
            assert(end.compare("_graph.end()") == 0);
          }

          // Get the arguments
          for (int i = 0, j = callFS->getNumArgs(); i < j; ++i) {
            string str_arg;
            llvm::raw_string_ostream s(str_arg);

            callFS->getArg(i)->printPretty(s, 0, Policy);

            //callFS->getArg(i)->IgnoreParenImpCasts()->dump();

            // see if argument is a call expression
            const CallExpr* callExpr = dyn_cast<CallExpr>(
                                        callFS->getArg(i)->IgnoreParenImpCasts()
                                       );
            if (callExpr) {
              const FunctionDecl* func = callExpr->getDirectCallee();
              //func->dump();
              if (func) {
                if (func->getNameInfo().getName().getAsString() == "write_set") {
                  llvm::errs() << "Inside arg write_set: " << i <<  s.str() 
                               << "\n\n";

                  // Take out first argument to see if it is push or pull
                  string str_sub_arg;
                  llvm::raw_string_ostream s_sub(str_sub_arg);
                  callExpr->getArg(0)->printPretty(s_sub, 0, Policy);

                  string str_sync_type = s_sub.str();

                  // remove " from beginning and end if exists
                  if(str_sync_type.front() == '"'){
                    str_sync_type.erase(0,1);
                    str_sync_type.erase(str_sync_type.size()-1);
                  }

                  // SYNC_PUSH
                  // if(callExpr->getNumArgs() == 8) 
                  if (str_sync_type == "sync_push") {
                    vector<string> Temp_vec_PUSH;

                    // get all arguments into a vector
                    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();
                      // trim unnecessary chars
                      if (temp_str.front() == '"') {
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if (temp_str.front() == '&') {
                        temp_str.erase(0,1);
                      }

                      llvm::errs() << " ------- > " << temp_str << "\n";
                      Temp_vec_PUSH.push_back(temp_str);
                    }

                    // save this field's writeset 
                    Write_set WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PUSH[1];
                    WR_entry.NODE_TYPE = Temp_vec_PUSH[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PUSH[3];
                    WR_entry.FIELD_NAME = Temp_vec_PUSH[4];
                    WR_entry.VAL_TYPE = Temp_vec_PUSH[5];
                    WR_entry.REDUCE_OP_EXPR = Temp_vec_PUSH[6];
                    WR_entry.RESET_VAL_EXPR = Temp_vec_PUSH[7];
                    WR_entry.SYNC_TYPE = "sync_push";

                    write_set_vec_PUSH.push_back(WR_entry);
                  // SYNC_PULL
                  } else if (str_sync_type == "sync_pull") {
                    vector<string> Temp_vec_PULL;

                    // get all arguments into a vector
                    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();
                      // trim unnecessary chars
                      if(temp_str.front() == '"'){
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if(temp_str.front() == '&'){
                        temp_str.erase(0,1);
                      }

                      Temp_vec_PULL.push_back(temp_str);
                    }

                    // save this field's writeset; same as push except
                    // type is sync_pull
                    Write_set WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PULL[1];
                    WR_entry.NODE_TYPE = Temp_vec_PULL[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PULL[3];
                    WR_entry.FIELD_NAME = Temp_vec_PULL[4];
                    WR_entry.VAL_TYPE = Temp_vec_PULL[5];
                    WR_entry.REDUCE_OP_EXPR = Temp_vec_PULL[6];
                    WR_entry.RESET_VAL_EXPR = Temp_vec_PULL[7];
                    WR_entry.SYNC_TYPE = "sync_pull";

                    write_set_vec_PULL.push_back(WR_entry);
                  }
                }
              } else if (func->getNameInfo().getName().getAsString() == 
                           "read_set") {
                llvm::errs() << "Inside arg read_set: " << i << s.str() 
                             << "\n\n";
              }
            } // if (callExpr) endif
          } // for loop over arguments end

          /************************************
           * Find unique one from push and pull
           * sets to generate for vertex cut
           ************************************/
          vector<Write_set> write_set_vec_PUSH_PULL;

          for (auto& i : write_set_vec_PUSH) {
            bool found = false;

            for (auto& j : write_set_vec_PULL) {
              if (i.GRAPH_NAME == j.GRAPH_NAME && 
                  i.NODE_TYPE == j.NODE_TYPE && 
                  i.FIELD_TYPE == j.FIELD_TYPE && 
                  i.FIELD_NAME == j.FIELD_NAME) {
                found = true;
              }
            }

            if (!found) {
              write_set_vec_PUSH_PULL.push_back(i);
            }
          }

          for (auto& i : write_set_vec_PULL) {
            bool found = false;

            for (auto& j : write_set_vec_PUSH) {
              if (i.GRAPH_NAME == j.GRAPH_NAME && 
                  i.NODE_TYPE == j.NODE_TYPE && 
                  i.FIELD_TYPE == j.FIELD_TYPE && 
                  i.FIELD_NAME == j.FIELD_NAME) {
                found = true;
              }
            }

            if (!found) {
              write_set_vec_PUSH_PULL.push_back(i);
            }
          }


          SourceLocation ST_main = callerFS->getSourceRange().getBegin();

          // TODO this seems completely unnecessary in our new code
          stringstream SSHelperStructFunctions;
          SSHelperStructFunctions 
            << "template <typename GraphTy>\n"
            << "struct Get_info_functor : public galois::op_tag {\n"
            << "\tGraphTy &graph;\n";
          rewriter.InsertText(ST_main, SSHelperStructFunctions.str(), true, true);
          SSHelperStructFunctions.str(string());
          SSHelperStructFunctions.clear();

          // Addding macros for push sync structures
          stringstream SSSyncer;
          std::map<std::string, std::vector<std::string>> syncCall_map;

          for (auto& i : write_set_vec_PUSH) {
            SSSyncer << getSyncer(i, syncCall_map);
            rewriter.InsertText(ST_main, SSSyncer.str(), true, true);
            SSSyncer.str(string());
            SSSyncer.clear();
          }

          // Addding macros for pull sync structures (includes bitset)
          stringstream SSSyncer_pull;
          for (auto& i : write_set_vec_PULL) {
            SSSyncer_pull << getSyncerPull(i, syncCall_map);
            rewriter.InsertText(ST_main, SSSyncer_pull.str(), true, true);
            SSSyncer_pull.str(string());
            SSSyncer_pull.clear();
          }

          // Addding additional structs for vertex cut
          stringstream SSSyncer_vertexCut;
          for (auto& i : write_set_vec_PUSH_PULL) {
            if (i.SYNC_TYPE == "sync_push") {
              SSSyncer_vertexCut << getSyncerPull(i, syncCall_map);
              rewriter.InsertText(ST_main, SSSyncer_vertexCut.str(), true, true);
              SSSyncer_vertexCut.str(string());
              SSSyncer_vertexCut.clear();
            } else if(i.SYNC_TYPE == "sync_pull"){
              SSSyncer_vertexCut << getSyncer(i, syncCall_map);
              rewriter.InsertText(ST_main, SSSyncer_vertexCut.str(), true, true);
              SSSyncer_vertexCut.str(string());
              SSSyncer_vertexCut.clear();
            }
          }

          // TODO seems completely unnecessary in new version of code
          // Adding helper function
          SSHelperStructFunctions <<"\tGet_info_functor(GraphTy& _g): graph(_g){}\n"
                                  <<"\tunsigned operator()(GNode n) const {\n"
                                  <<"\t\treturn graph.getHostID(n);\n\t}\n"
                                  <<"\tGNode getGNode(uint32_t local_id) const {\n"
                                  <<"\t\treturn GNode(graph.getGID(local_id));\n\t}\n"
                                  <<"\tuint32_t getLocalID(GNode n) const {\n"
                                  <<"\t\treturn graph.getLID(n);\n\t}\n"
                                  <<"\tvoid sync_graph(){\n"
                                  <<"\t\tsync_graph_static(graph);\n\t}\n"
                                  <<"\tstd::string get_run_identifier() const {\n"
                                  <<"\t\treturn graph.get_run_identifier();\n\t}\n"
                                  <<"\tvoid static sync_graph_static(Graph& _graph) {\n";

          rewriter.InsertText(ST_main, SSHelperStructFunctions.str(), true, true);
          SSHelperStructFunctions.str(string());
          SSHelperStructFunctions.clear();

          // Adding sync calls for write set
          stringstream SSAfter;
          for(auto& i : syncCall_map){
            SSAfter <<"\n\t" << "_graph.sync<";
            uint32_t c = 0;
            for(auto& j : i.second){
              if((c + 1) < i.second.size())
                SSAfter << j << ",";
              else
                SSAfter << j;
              ++c;
            }

            // TODO
            // originally a \( instead of (...
            SSAfter << ">(\"" << OperatorStructName << "\");\n";
            rewriter.InsertText(ST_main, SSAfter.str(), true, true);
            SSAfter.str(string());
            SSAfter.clear();
          }

          // TODO again completely unnecessary in new code
          // closing helpFunction struct
          SSHelperStructFunctions << "\t}\n};\n\n";
          rewriter.InsertText(ST_main, SSHelperStructFunctions.str(), true, true);
          SSHelperStructFunctions.str(string());
          SSHelperStructFunctions.clear();

          // TODO the below stuff seems like CUDA stuff: revise as this seems 
          // outdated?
          auto recordDecl = results.Nodes.getNodeAs<clang::CXXRecordDecl>("class");
          std::string className = recordDecl->getNameAsString();
          
          std::string accumulator;
          for (auto& decl : recordDecl->decls()) {
            auto var = dyn_cast<VarDecl>(decl);
            if (var && var->isStaticDataMember()) {
              std::string type = var->getType().getAsString();
              if (type.find(__GALOIS_ACCUMULATOR_TYPE__) == 0) {
                assert(accumulator.empty());
                accumulator = var->getNameAsString();
              }
            }
          }
          stringstream kernelBefore;
          kernelBefore << "#ifdef __GALOIS_HET_CUDA__\n";
          kernelBefore << "\tif (personality == GPU_CUDA) {\n";
          kernelBefore << "\t\tauto& __sync_functor = Get_info_functor<Graph>(_graph);\n";
          kernelBefore << "\t\ttypedef galois::DGBag<GNode, Get_info_functor<Graph> > DBag;\n";
          kernelBefore << "\t\tDBag dbag(__sync_functor, \"" << className << "\");\n";
          kernelBefore << "\t\tauto& &local_wl = DBag::get();\n";
          kernelBefore << "\t\tstd::string impl_str(\"CUDA_FOR_EACH_IMPL_" 
            << className << "_\" + (_graph.get_run_identifier()));\n";
          kernelBefore << "\t\tgalois::StatTimer StatTimer_cuda(impl_str.c_str());\n";
          kernelBefore << "\t\tunsigned long _num_work_items;\n";
          kernelBefore << "\t\tStatTimer_cuda.start();\n";
          if (!single_source.empty()) {
            kernelBefore << "\t\tif (_graph.isOwned(" << single_source << ")) {\n";
            kernelBefore << "\t\t\tcuda_wl.num_in_items = 1;\n";
            kernelBefore << "\t\t\tcuda_wl.in_items[0] = " << begin << ";\n";
            kernelBefore << "\t\t} else\n";
            kernelBefore << "\t\t\tcuda_wl.num_in_items = 0;\n";
          } else {
            kernelBefore << "\t\tcuda_wl.num_in_items = (*(" << end << ")-*(" << begin << "));\n";
            kernelBefore << "\t\tfor (int __i = *(" << begin << "); __i < *(" << end << "); ++__i) cuda_wl.in_items[__i] = __i;\n";
          }
          stringstream cudaKernelCall;
          if (!accumulator.empty()) {
            cudaKernelCall << "\t\tint __retval = 0;\n";
          }
          cudaKernelCall << "\t\tcuda_wl.num_out_items = 0;\n";
          cudaKernelCall << "\t\t_num_work_items += cuda_wl.num_in_items;\n";
          cudaKernelCall << "\t\tif (cuda_wl.num_in_items > 0)\n";
          cudaKernelCall << "\t\t\t" << className << "_cuda(";
          if (!accumulator.empty()) {
            cudaKernelCall << "__retval, ";
          }
          for (auto field : recordDecl->fields()) {
            std::string name = field->getNameAsString();
            if (name.find(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__) == 0) {
              cudaKernelCall << name.substr(std::strlen(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__)) << ", ";
            }
          }
          cudaKernelCall << "cuda_ctx);\n";
          if (!accumulator.empty()) {
            cudaKernelCall << "\t\t" << accumulator << " += __retval;\n";
          }
          cudaKernelCall << "\t\tStatTimer_cuda.stop();\n";
          cudaKernelCall << "\t\t__sync_functor.sync_graph();\n";
          cudaKernelCall << "\t\tdbag.set_local(cuda_wl.out_items, cuda_wl.num_out_items);\n";
          cudaKernelCall << "\t\t#ifdef __GALOIS_DEBUG_WORKLIST__\n";
          cudaKernelCall << "\t\tstd::cout << \"[\" << galois::runtime::getSystemNetworkInterface().ID << \"] worklist size : \" << cuda_wl.num_out_items << \" duplication factor : \" << (double)cuda_wl.num_out_items/_graph.size() << \"\\n\";\n";
          cudaKernelCall << "\t\t#endif\n";
          cudaKernelCall << "\t\tdbag.sync();\n";

          kernelBefore << cudaKernelCall.str();
          kernelBefore << "\t\tunsigned _num_iterations = 1;\n";
          kernelBefore << "\t\twhile (!dbag.canTerminate()) {\n";
          kernelBefore << "\t\t_graph.set_num_iter(_num_iterations);\n";
          kernelBefore << "\t\tStatTimer_cuda.start();\n";
          kernelBefore << "\t\tcuda_wl.num_in_items = local_wl.size();\n";
          kernelBefore << "\t\tif (cuda_wl.num_in_items > cuda_wl.max_size) {\n";
          kernelBefore << "\t\t\tstd::cout << \"[\" << galois::runtime::getSystemNetworkInterface().ID << \"] ERROR - worklist size insufficient; size : \" << cuda_wl.max_size << \" , expected : \" << cuda_wl.num_in_items << \"\\n\";\n";
          kernelBefore << "\t\t\texit(1);\n";
          kernelBefore << "\t\t}\n";
          kernelBefore << "\t\t//std::cout << \"[\" << galois::runtime::getSystemNetworkInterface().ID << \"] Iter : \" << num_iter ";
          kernelBefore << "<< \" Total items to work on : \" << cuda_wl.num_in_items << \"\\n\";\n";
          kernelBefore << "\t\tstd::copy(local_wl.begin(), local_wl.end(), cuda_wl.in_items);\n";
          kernelBefore << cudaKernelCall.str();
          kernelBefore << "\t\t++_num_iterations;\n";
          kernelBefore << "\t\t}\n";
          kernelBefore << "\t\tgalois::runtime::reportStat(\"(NULL)\", \"NUM_ITERATIONS_\" + (_graph.get_run_identifier()), (unsigned long)_num_iterations, 0);\n";
          kernelBefore << "\t\tgalois::runtime::reportStat(\"(NULL)\", \"NUM_WORK_ITEMS_\" + (_graph.get_run_identifier()), _num_work_items, 0);\n";
          kernelBefore << "\t} else if (personality == CPU)\n";
          kernelBefore << "#endif\n";

          // TODO check for sanity
          // here begins non-cuda code adding
          // this is for first-itr loops
          if (!single_source.empty()) {
            kernelBefore << "\t{\n";
            kernelBefore << "\t\tunsigned int __begin, __end;\n";
            kernelBefore << "\t\tif (_graph.isOwned(" << single_source << ")) {\n";
            kernelBefore << "\t\t\t__begin = " << begin << ";\n";
            kernelBefore << "\t\t\t__end = __begin + 1;\n";
            kernelBefore << "\t\t} else {\n";
            kernelBefore << "\t\t\t__begin = 0;\n";
            kernelBefore << "\t\t\t__end = 0;\n";
            kernelBefore << "\t\t}\n";
          }
          SourceLocation ST_before = callFS->getSourceRange().getBegin();
          rewriter.InsertText(ST_before, kernelBefore.str(), true, true);

          if (!single_source.empty()) {
            SourceLocation ST_forEach_start = callFS->getSourceRange().getBegin().getLocWithOffset(0);
            string galois_foreach = "galois::for_each(";
            string iterator_range = "boost::make_counting_iterator(__begin), boost::make_counting_iterator(__end)";
            rewriter.ReplaceText(ST_forEach_start, galois_foreach.length() + single_source.length(), galois_foreach + iterator_range);
          }

          //insert helperFunc in for_each call
          SourceLocation ST_forEach_end = callFS->getSourceRange().getEnd().getLocWithOffset(0);
          //SSHelperStructFunctions << ", Get_info_functor<Graph>(_graph), galois::wl<dChunk>()";

          //Assumption:: User will give worklist galois::wl in for_each.
          SSHelperStructFunctions << ", Get_info_functor<Graph>(_graph)";
          rewriter.InsertText(ST_forEach_end, SSHelperStructFunctions.str(), true, true);

          if (!single_source.empty()) {
            SourceLocation ST_forEach_end2 = callFS->getSourceRange().getEnd().getLocWithOffset(2);
            string end_single_source = "\t}\n";
            rewriter.InsertText(ST_forEach_end2, end_single_source, true, true);
          }
        }
      }
  };

  class FunctionCallHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
    public:
      FunctionCallHandler(Rewriter &rewriter) : rewriter(rewriter) {}
      virtual void run(const MatchFinder::MatchResult &results) {
        const CallExpr* callFS = results.Nodes.getNodeAs<clang::CallExpr>(
                                   "galoisLoop");
        llvm::errs() << "Compiler in FunctionCallHandler with this many args:\n" << 
                        callFS->getNumArgs() << "\n";

        auto callerStruct = results.Nodes.getNodeAs<clang::CXXRecordDecl>(
                              "class");

        if (callFS) {
          std::string OperatorStructName = callerStruct->getNameAsString();
          llvm::errs() << " NAME OF THE STRUCT : " << OperatorStructName << "\n";

          // location of string where we can insert text; this is the beginning
          // of the structure
          SourceLocation ST_main = callFS->getSourceRange().getBegin();

          llvm::errs() << "galois::do_all loop found\n";


          /*
          string str_first;
          llvm::raw_string_ostream s_first(str_first);
          callFS->getArg(0)->printPretty(s_first, 0, Policy);
          //llvm::errs() << " //////// > " << s_first.str() << "\n";
          callFS->getArg(0)->dump();
          */

          vector<Write_set> write_set_vec_PUSH;
          vector<Write_set> write_set_vec_PULL;

          // pretty print options
          clang::LangOptions LangOpts;
          LangOpts.CPlusPlus = true;
          clang::PrintingPolicy Policy(LangOpts);

          // grab 1st and 2nd argument of do all call; regularly
          // it is begin/end, but if not, then assume this is a single
          // source do all
          assert(callFS->getNumArgs() > 1);
          string str_begin_arg;
          llvm::raw_string_ostream s_begin(str_begin_arg);
          callFS->getArg(0)->printPretty(s_begin, 0, Policy);
          string begin = s_begin.str();
          string str_end_arg;
          llvm::raw_string_ostream s_end(str_end_arg);
          callFS->getArg(1)->printPretty(s_end, 0, Policy);
          string end = s_end.str();

          string single_source;

          // TODO: revise to work with new range calls too.... (allNodes, etc.)
          // TODO: use type comparison instead
          if (end.find(".end()") == string::npos) {
            // second arg wasn't end; assume single source
            single_source = begin;
            begin = "_graph.getLID(" + single_source + ")";
            end = begin + "+1";
            // use begin and end only if _graph_isOwned(single_source)
          } else {
            // make sure arguments are begin and end
            assert(begin.compare("_graph.begin()") == 0);
            assert(end.compare("_graph.end()") == 0);
          }

          // Get the arguments
          for (int i = 0, j = callFS->getNumArgs(); i < j; ++i) {
            string str_arg;
            llvm::raw_string_ostream s(str_arg);
            callFS->getArg(i)->printPretty(s, 0, Policy);

            //callFS->getArg(i)->IgnoreParenImpCasts()->dump();
            const CallExpr* callExpr = dyn_cast<CallExpr>(
                callFS->getArg(i)->IgnoreParenImpCasts());

            //const CallExpr* callExpr1 = callFS->getArg(i);;
            //callExpr->dump();
            if (callExpr) {
              const FunctionDecl* func = callExpr->getDirectCallee();
              //func->dump();
              if (func) {
                if (func->getNameInfo().getName().getAsString() == "write_set") {
                  llvm::errs() << "Inside arg write_set: " << i << s.str() << "\n\n";

                  // Take out first argument to see if it is push or pull
                  string str_sub_arg;
                  llvm::raw_string_ostream s_sub(str_sub_arg);
                  callExpr->getArg(0)->printPretty(s_sub, 0, Policy);

                  string str_sync_type = s_sub.str();
                  // clear " from begin and end
                  if(str_sync_type.front() == '"'){
                    str_sync_type.erase(0,1);
                    str_sync_type.erase(str_sync_type.size()-1);
                  }

                  // SYNC_PUSH
                  // if(callExpr->getNumArgs() == 8) 
                  if (str_sync_type == "sync_push") {
                    vector<string> Temp_vec_PUSH;

                    // grab write set arguments
                    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();
                      // clear unnecessary things
                      if (temp_str.front() == '"') {
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if (temp_str.front() == '&') {
                        temp_str.erase(0,1);
                      }

                      llvm::errs() << " ------- > " << temp_str << " Size : " 
                                   << Temp_vec_PUSH.size() <<  "\n";
                      Temp_vec_PUSH.push_back(temp_str);
                    }

                    // save write set arguments
                    Write_set WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PUSH[1];
                    WR_entry.NODE_TYPE = Temp_vec_PUSH[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PUSH[3];
                    WR_entry.FIELD_NAME = Temp_vec_PUSH[4];
                    WR_entry.VAL_TYPE = Temp_vec_PUSH[5];
                    WR_entry.REDUCE_OP_EXPR = Temp_vec_PUSH[6];
                    WR_entry.RESET_VAL_EXPR = Temp_vec_PUSH[7];
                    WR_entry.SYNC_TYPE = "sync_push";
                    WR_entry.READ_FLAG = Temp_vec_PUSH[8];
                    WR_entry.WRITE_FLAG = Temp_vec_PUSH[9];

                    write_set_vec_PUSH.push_back(WR_entry);
                  // SYNC_PULL
                  } else if (str_sync_type == "sync_pull") {
                    vector<string> Temp_vec_PULL;

                    // get write set args
                    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();

                      // clear unnecessary things
                      if (temp_str.front() == '"') {
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if (temp_str.front() == '&') {
                        temp_str.erase(0,1);
                      }

                      Temp_vec_PULL.push_back(temp_str);

                    }

                    // save write set args
                    Write_set WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PULL[1];
                    WR_entry.NODE_TYPE = Temp_vec_PULL[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PULL[3];
                    WR_entry.FIELD_NAME = Temp_vec_PULL[4];
                    WR_entry.VAL_TYPE = Temp_vec_PULL[5];
                    WR_entry.REDUCE_OP_EXPR = Temp_vec_PULL[6];
                    WR_entry.RESET_VAL_EXPR = Temp_vec_PULL[7];
                    WR_entry.SYNC_TYPE = "sync_pull";
                    WR_entry.READ_FLAG = Temp_vec_PULL[8];
                    WR_entry.WRITE_FLAG = Temp_vec_PULL[9];

                    write_set_vec_PULL.push_back(WR_entry);
                  }
                }
                if (func->getNameInfo().getName().getAsString() == "read_set"){
                  llvm::errs() << "Inside arg read_set: " << i <<  s.str() << "\n\n";
                }

              }
            }
          }

          /********************************************************************
           * Find unique one from push and pull sets to generate for vertex cut
           ********************************************************************/
          vector<Write_set> write_set_vec_PUSH_PULL;

          for (auto& i : write_set_vec_PUSH) {
            bool found = false;

            for (auto& j : write_set_vec_PULL) {
              if (i.GRAPH_NAME == j.GRAPH_NAME && 
                 i.NODE_TYPE == j.NODE_TYPE && 
                 i.FIELD_TYPE == j.FIELD_TYPE && 
                 i.FIELD_NAME == j.FIELD_NAME) {
                found = true;
              }
            }

            if (!found) write_set_vec_PUSH_PULL.push_back(i);
          }

          for (auto& i : write_set_vec_PULL) {
            bool found = false;

            for (auto& j : write_set_vec_PUSH) {
              if (i.GRAPH_NAME == j.GRAPH_NAME && 
                  i.NODE_TYPE == j.NODE_TYPE && 
                  i.FIELD_TYPE == j.FIELD_TYPE && 
                  i.FIELD_NAME == j.FIELD_NAME) {
                found = true;
              }
            }

            if(!found) write_set_vec_PUSH_PULL.push_back(i);
          }

          // deal with single source case: set up do all call vars
          if (!single_source.empty()) {
            stringstream kernelBefore;
            kernelBefore << "\t\tunsigned int __begin, __end;\n";
            kernelBefore << "\t\tif (_graph.isOwned(" << single_source << ")) {\n";
            kernelBefore << "\t\t\t__begin = " << begin << ";\n";
            kernelBefore << "\t\t\t__end = __begin + 1;\n";
            kernelBefore << "\t\t} else {\n";
            kernelBefore << "\t\t\t__begin = 0;\n";
            kernelBefore << "\t\t\t__end = 0;\n";
            kernelBefore << "\t\t}\n";
            rewriter.InsertText(ST_main, kernelBefore.str(), true, true);
          }

          // IMP: Ordering matter : All getSyncer calls should be before 
          // getSyncerPull
          // TODO reason about why this is the case
          stringstream SSSyncer;

          // map from FIELD_NAME -> reduce, broadcast, bitset;
          std::map<std::string, std::vector<std::string>> syncCall_map;

          for (auto& i : write_set_vec_PUSH) {
            SSSyncer << getSyncer(i, syncCall_map);
            rewriter.InsertText(ST_main, SSSyncer.str(), true, true);
            SSSyncer.str(string());
            SSSyncer.clear();
          }

          // Addding additional structs for vertex cut
          stringstream SSSyncer_vertexCut;
          for (auto& i : write_set_vec_PUSH_PULL) {
            if (i.SYNC_TYPE == "sync_pull") {
              SSSyncer_vertexCut << getSyncer(i, syncCall_map);
              rewriter.InsertText(ST_main, SSSyncer_vertexCut.str(), true, true);
              SSSyncer_vertexCut.str(string());
              SSSyncer_vertexCut.clear();
            }
          }

          // Addding structure for pull sync
          stringstream SSSyncer_pull;
          for(auto& i : write_set_vec_PULL) {
            SSSyncer_pull << getSyncerPull(i, syncCall_map);
            rewriter.InsertText(ST_main, SSSyncer_pull.str(), true, true);
            SSSyncer_pull.str(string());
            SSSyncer_pull.clear();
          }


          // Addding additional structs for vertex cut
          for (auto& i : write_set_vec_PUSH_PULL) {
            if(i.SYNC_TYPE == "sync_push"){
              SSSyncer_vertexCut << getSyncerPull(i, syncCall_map);
              rewriter.InsertText(ST_main, SSSyncer_vertexCut.str(), true, true);
              SSSyncer_vertexCut.str(string());
              SSSyncer_vertexCut.clear();
            }
          }

          for (auto& i : syncCall_map) {
            llvm::outs() << "__________________> : "
                         << i.first << " ---- > " <<  i.second[0] << " , "
                         << i.second[1] << "\n";
          }

          auto recordDecl = results.Nodes.getNodeAs<clang::CXXRecordDecl>("class");
          std::string className = recordDecl->getNameAsString();

          // TODO pass over this
          // CUDA stuff
          stringstream kernelBefore;
          kernelBefore << "#ifdef __GALOIS_HET_CUDA__\n";
          kernelBefore << "\tif (personality == GPU_CUDA) {\n";
          kernelBefore << "\t\tstd::string impl_str(\"CUDA_DO_ALL_IMPL_" 
            << className << "_\" + (_graph.get_run_identifier()));\n";
          kernelBefore << "\t\tgalois::StatTimer StatTimer_cuda(impl_str.c_str());\n";
          kernelBefore << "\t\tStatTimer_cuda.start();\n";
          std::string accumulator;
          for (auto& decl : recordDecl->decls()) {
            auto var = dyn_cast<VarDecl>(decl);
            if (var && var->isStaticDataMember()) {
              std::string type = var->getType().getAsString();
              if (type.find(__GALOIS_ACCUMULATOR_TYPE__) == 0) {
                assert(accumulator.empty());
                accumulator = var->getNameAsString();
              }
            }
          }
          if (!accumulator.empty()) {
            kernelBefore << "\t\tint __retval = 0;\n";
          }
          if (!single_source.empty()) {
            kernelBefore << "\t\t" << className << "_cuda(";
            kernelBefore << "__begin, __end, ";
          } else {
            kernelBefore << "\t\t" << className << "_all_cuda(";
          }
          if (!accumulator.empty()) {
            kernelBefore << "__retval, ";
          }
          for (auto field : recordDecl->fields()) {
            std::string name = field->getNameAsString();
            if (name.find(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__) == 0) {
              kernelBefore << name.substr(
               std::strlen(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__)) << ", ";
            }
          }
          kernelBefore << "cuda_ctx);\n";
          if (!accumulator.empty()) {
            kernelBefore << "\t\t" << accumulator << " += __retval;\n";
          }
          kernelBefore << "\t\tStatTimer_cuda.stop();\n";
          kernelBefore << "\t} else if (personality == CPU)\n";
          kernelBefore << "#endif\n";
          rewriter.InsertText(ST_main, kernelBefore.str(), true, true);


          // TODO make sanity pass
          // replace do all call with single source call if necessary
          if (!single_source.empty()) {
            string galois_doall = "galois::do_all(";
            string iterator_range = "boost::make_counting_iterator(__begin), "
                                    "boost::make_counting_iterator(__end)";
            rewriter.ReplaceText(ST_main,
                                 galois_doall.length() + single_source.length(),
                                 galois_doall + iterator_range);
          }


          // Adding sync calls for write set
          stringstream SSAfter;
          for (auto& i : syncCall_map) {
            SSAfter << "\n\t" << "_graph.sync_on_demand<";

            uint32_t c = 0;

            // get the structures and add them to the sync on demand call
            // template args; they should be 
            // (read location, redue, broadcast, bitset)
            // TODO verify read location is being added
            // TODO remove flags from this as they aren't a template argument
            // anymore
            for (auto& j : i.second) {
              if((c + 1) < i.second.size())
                SSAfter << j << ",";
              else
                SSAfter << j;
              ++c;
            }

            // TODO flags now an argument; add them
            // TODO originally \( instead of (
            SSAfter << ">(\"" << OperatorStructName << "\");\n\n";
            rewriter.InsertText(ST_main, SSAfter.str(), true, true);
            SSAfter.str(string());
            SSAfter.clear();
          }

          SourceLocation ST = callFS->
                              getSourceRange().getEnd().getLocWithOffset(2);

          // Inserting write flags after do_all

          // TODO why is it only looping over push and pull and not
          // others?
          for (auto& i : write_set_vec_PUSH_PULL) {
            if (i.SYNC_TYPE == "sync_push" || i.SYNC_TYPE == "sync_pull") {
              if (i.WRITE_FLAG == "writeDestination") {
                SSAfter << "\nFlags_" << i.FIELD_NAME << "::set_write_dst();\n";
              } else if (i.WRITE_FLAG == "writeSource") {
                SSAfter << "\nFlags_" << i.FIELD_NAME << "::set_write_src();\n";
              } else if (i.WRITE_FLAG == "writeAny") {
                SSAfter << "\nFlags_" << i.FIELD_NAME << "::set_write_both();\n";
              } else {
                SSAfter << "\nFlags_" << i.FIELD_NAME << "::set_write_none();\n";
              }

              rewriter.InsertText(ST, SSAfter.str(), true, true);
              SSAfter.str(string());
              SSAfter.clear();
            }
          }
        }
      }
  };

  class GaloisFunctionsConsumer : public ASTConsumer {
    private:
      //CompilerInstance &Instance;
      std::set<std::string> ParsedTemplates;
      GaloisFunctionsVisitor* Visitor;
      MatchFinder Matchers;
      ForStmtHandler forStmtHandler;
      //NameSpaceHandler namespaceHandler;
      FunctionCallHandler functionCallHandler;
      FunctionForEachHandler forEachHandler;
    public:
      GaloisFunctionsConsumer(CompilerInstance &Instance, 
                              std::set<std::string> ParsedTemplates, 
                              Rewriter &R) : 
            ParsedTemplates(ParsedTemplates), 
            Visitor(new GaloisFunctionsVisitor(&Instance)), 
            functionCallHandler(R), 
            forEachHandler(R) 
      {
        // do all loop matcher
        // TODO this matcher assumes do all syntax is range.begin, range.end;
        // do_all_local does not necessarily follow this syntax......
        Matchers.addMatcher(
          callExpr(
            isExpansionInMainFile(), 
            callee(
              functionDecl(
                anyOf(
                  hasName("galois::do_all"),
                  hasName("galois::do_all_local")
                )
              )
            ), 
            hasAncestor(recordDecl().bind("class"))
          ).bind("galoisLoop"), &functionCallHandler
        );

        // TODO for each needed?
        /** for galois::for_each. Needs different treatment. **/
        Matchers.addMatcher(
          callExpr(
            isExpansionInMainFile(), 
            callee(functionDecl(hasName("galois::for_each"))), 
            hasDescendant(
              declRefExpr(to(functionDecl(hasName("workList_version"))))
            ), 
            hasAncestor(recordDecl().bind("class"))
          ).bind("galoisLoop_forEach"), &forEachHandler
        );
      }

      virtual void HandleTranslationUnit(ASTContext &Context) {
        Matchers.matchAST(Context);
      }
  };

  class GaloisFunctionsAction : public PluginASTAction {
    private:
      std::set<std::string> ParsedTemplates;
      Rewriter TheRewriter;
    protected:
      /**
       * Write changes to file at end
       */
      void EndSourceFileAction() override {
        TheRewriter.getEditBuffer(
          TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs()
        );
        if (!TheRewriter.overwriteChangedFiles()) {
          llvm::errs() << "Successfully saved changes\n";
        }
      }

      /**
       * Register the AST matcher
       */
      std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
                                                     llvm::StringRef) override {
        TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        return llvm::make_unique<GaloisFunctionsConsumer>(CI, ParsedTemplates,
                                                          TheRewriter);
      }

      bool ParseArgs(const CompilerInstance &CI, 
                     const std::vector<std::string> &args) override {
        return true;
      }
  };
}

static FrontendPluginRegistry::Add<GaloisFunctionsAction> X("galois-fns", 
                                              "Add synchronization constructs");
