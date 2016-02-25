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
//#include "clang/Sema/TreeTransform.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

//#include "Generation.h"
#include "ClangUtils.h"
//#include "GaloisCLGen.h"
//#include "GaloisAST.h"
//#include "AOSToSOA.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

static const TypeMatcher AnyType = anything();
ASTUtility ast_utility;

#include "Analysis.h"

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

public:
   OpenCLCodeGenConsumer(CompilerInstance &Instance, std::set<std::string> &ParsedTemplates, Rewriter &_R) :
         Instance(Instance), ParsedTemplates(ParsedTemplates), R(_R){
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
      OpenCLDeviceHandler clDevHandler(R);
         MatchFinder opFinder;
         opFinder.addMatcher(methodDecl(isExpansionInMainFile(), hasName("operator()")).bind("kernel"), &clDevHandler);
         opFinder.matchAST(Context);

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
//      llvm::outs() << "EndSourceFileAction() :: Entered\n" << "ParseTemplates:: " << ParsedTemplates.size() << "\n";
//      std::string kernel_code;
//      raw_string_ostream kernel_stream(kernel_code);
//      TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(kernel_stream);
//      kernel_code.append("\n=======================================================\n");
//      TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID())->write(kernel_stream);
//      std::ofstream kernel_file ("kernel.cl");
//      kernel_file << kernel_code;
//      kernel_file.close();
//      llvm::outs() << kernel_code << " \n";
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
