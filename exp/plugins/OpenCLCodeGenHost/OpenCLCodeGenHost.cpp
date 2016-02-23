/*
 * Author : Rashid Kaleem (rashid.kaleem@gmail.com)
 */

#include <sstream>
#include <climits>
#include <vector>
#include <map>
#include <regex>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/SourceManager.h"

/*
 *Matchers
 */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

#include "Analysis.h"
#include "ClangUtils.h"
#include "GaloisAST.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

static const TypeMatcher AnyType = anything();
ASTUtility ast_utility;

namespace {

/********************************************************************************************************
 *
 *
 ********************************************************************************************************/

class OpenCLCodeGenConsumer: public ASTConsumer {
private:
   CompilerInstance &Instance;
   std::set<std::string> ParsedTemplates;
   MatchFinder Matchers;
   Rewriter & R;
   Galois::GAST::GaloisApp app_data;
   DoAllHandler do_all_handler;
   GraphDeclHandler graphDeclHandler;

public:
   OpenCLCodeGenConsumer(CompilerInstance &Instance, std::set<std::string> &ParsedTemplates, Rewriter &_R) :
         Instance(Instance), ParsedTemplates(ParsedTemplates), R(_R), app_data(R), do_all_handler(R, app_data), graphDeclHandler(R) {
      llvm::outs() << "=============================Created OpenCLCodeGenConsumer===========================\n";

   }
   /********************************************************************************************************
    * 1) Find all the do_all calls in the main-file.
    * 2) For each do_all call, go over the class that defines the operator
    * 3) Add call to the generated operator implementation in OpenCL
    *
    ********************************************************************************************************/

   virtual void HandleTranslationUnit(ASTContext &Context) {
      ast_utility.init(&R);
      llvm::outs() << "=============================InHandleTranslationUnit===========================\n";
      { // Get the graph declarations
         MatchFinder graphDecls;
         Galois::GAST::GraphTypeMapper gtm(R);
         graphDecls.addMatcher(
                     varDecl(hasType(recordDecl(hasName("Galois::Graph::LC_CSR_Graph")).bind("GraphType"))).bind("graphDecl")
               , &gtm);
         graphDecls.matchAST(Context);
         app_data.register_type(*gtm.graph_types.begin(), "CLGraph");

      }
      { // Replace typenames for graph data-types.
      MatchFinder mf;
      Galois::GAST::GraphTypeReplacer typeReplacer(R);
      mf.addMatcher(varDecl( isExpansionInMainFile(), hasInitializer( callExpr( callee(functionDecl( anyOf(
                                             hasName("getData"),
                                             hasName("edge_begin"),
                                             hasName("edge_end"),
                                             hasName("getEdgeData"),
                                             hasName("getEdgeDst"),
                                             hasName("edges")  )).bind("callExpr") )  ) ) ).bind("varDecl"),
            &typeReplacer
            );

      mf.addMatcher(varDecl( isExpansionInMainFile() ,hasInitializer( constructExpr( hasAnyArgument( callExpr( callee (functionDecl( anyOf(
                                             hasName("getData"),
                                             hasName("edge_begin"),
                                             hasName("edge_end"),
                                             hasName("getEdgeData"),
                                             hasName("getEdgeDst")  )).bind("callExpr") ) ) ) ) ) ).bind("varDecl"), &typeReplacer);
      mf.matchAST(Context);
      }
      {//#################Find kernels/operators via do_all calls
            Matchers.addMatcher(callExpr(
                                 callee(functionDecl(hasName("Galois::do_all")))
                                        ,hasArgument(2,hasType(recordDecl().bind("kernelType")))
                                 ).bind("galoisLoop"), &do_all_handler);
            Matchers.matchAST(Context);
            //#################End find kernels/operators via do_all calls
      }
   }//End HandleTranslationUnit
 };
/********************************************************************************************************
 * The plugin AST action.
 *
 ********************************************************************************************************/
class OpenCLCodeGenAction: public PluginASTAction {
private:
   std::set<std::string> ParsedTemplates;
   Rewriter TheRewriter;
protected:

   void EndSourceFileAction() override {
      llvm::outs() << "EndSourceFileAction() :: Entered\n" << "ParseTemplates:: " << ParsedTemplates.size() << "\n";
      for (auto s : ParsedTemplates) {
         llvm::outs() << "PARSED-TEMPLATE:: " << s << "\n";
      }
      //Write the transformed AST to llvm::outs()
      if (!TheRewriter.overwriteChangedFiles()) {
         llvm::outs() << "Successfully saved changes\n";
      }
      llvm::outs() << "EndSourceFileAction() :: Returning\n";
   }
   std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
      llvm::outs() << "CreateASTConsumer() :: Entered\n";
      TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
      llvm::outs() << "CreateASTConsumer() :: Returning\n";
      return llvm::make_unique<OpenCLCodeGenConsumer>(CI, ParsedTemplates, TheRewriter);
   }

   bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
      return true;
   }

};

} // End anonymous namespace
/********************************************************************************************************
 * Register the plugin. The plugin can be invoked by
 * ./clang -cc1 -load ../lib/OpenCLCodeGen.so -plugin opencl-analysis <FILENAME>
 *
 ********************************************************************************************************/
static FrontendPluginRegistry::Add<OpenCLCodeGenAction> X("opencl-analysis", "Galois OpenCL codegen");
