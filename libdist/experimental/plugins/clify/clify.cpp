// A tool to generate  OpenCL Host side code from Galois shared memory code.
//@author Rashid Kaleem <rashid.kaleem@gmail.com>
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#include "utils.h"

ASTUtility ast_utility;

#include "cl_transforms.h"
#include "kernel_gen.h"
#include "syncHandler.h"

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("distifier options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

/********************************************************************************************************
 *
 *
 ********************************************************************************************************/

class OpenCLHostGenConsumer : public ASTConsumer {
private:
  Rewriter& R;
  galois::GAST::GaloisApp app_data;

public:
  OpenCLHostGenConsumer(Rewriter& _R) : R(_R), app_data(R) {
    llvm::outs() << "=============================Created "
                    "OpenCLHostGenConsumer===========================\n";
  }
  /********************************************************************************************************
   * 1) Find all the do_all calls in the main-file.
   * 2) For each do_all call, go over the class that defines the operator
   * 3) Add call to the generated operator implementation in OpenCL
   *
   ********************************************************************************************************/
  //#define GEN_KERNEL
  virtual void HandleTranslationUnit(ASTContext& Context) {
    ast_utility.init(&R);
    llvm::outs() << "=============================InHandleTranslationUnit======"
                    "=====================\n";
#ifdef GEN_KERNEL
    {
      MatchFinder ndFinder;
      NodeDataGen ndGen(R, Context, app_data);
      //         ndFinder.addMatcher( varDecl(isExpansionInMainFile(),
      //         hasType(allOf( recordDecl(hasMethod(hasName("getData")) ) ,
      //         classTemplateSpecializationDecl( hasTemplateArgument(0,
      //         templateArgument().bind("NData") )  )) ) ), &ndGen);
      ndFinder.addMatcher(
          varDecl(
              isExpansionInMainFile(),
              allOf(hasType(recordDecl(hasMethod(hasName("getData")))
                                .bind("GraphType")),
                    hasType(classTemplateSpecializationDecl(hasTemplateArgument(
                        0, templateArgument().bind("NData")))))),
          &ndGen);
      ndFinder.addMatcher(
          varDecl(
              isExpansionInMainFile(),
              allOf(hasType(recordDecl(hasMethod(hasName("getData")))
                                .bind("GraphType")),
                    hasType(classTemplateSpecializationDecl(hasTemplateArgument(
                        1, templateArgument().bind("EData")))))),
          &ndGen);
      ndFinder.matchAST(Context);
    }
    { // Perform code generation.
      OpenCLDeviceHandler clDevHandler(R, app_data);
      MatchFinder opFinder;
      opFinder.addMatcher(
          methodDecl(isExpansionInMainFile(), hasName("operator()"))
              .bind("kernel"),
          &clDevHandler);
      opFinder.matchAST(Context);
    }
#else
    if (false) {
      // Sync. handler code.
      // TODO - RK - finsih this as it will greatly improve the sync. efficiency
      // (reduce time) of the system
      SyncHandler syncHandler(R, Context, app_data);
      MatchFinder syncFinder;
      syncFinder.addMatcher(
          memberCallExpr(isExpansionInMainFile(),
                         callee(functionDecl(hasName("sync_pull"),
                                             isTemplateInstantiation())
                                    .bind("methodDecl")))
              .bind("callSite"),
          &syncHandler);
      syncFinder.addMatcher(
          memberCallExpr(isExpansionInMainFile(),
                         callee(functionDecl(hasName("sync_push"),
                                             isTemplateInstantiation())
                                    .bind("methodDecl")))
              .bind("callSite"),
          &syncHandler);
      syncFinder.matchAST(Context);
      return;
    }
    {
      // 1) Replace DistGraph with CLGraph in typedefs
      MatchFinder graphTypedef;
      GraphTypedefRewriter gtr(R, Context);
      graphTypedef.addMatcher(
          typedefDecl(isExpansionInMainFile(), hasName("Graph"))
              .bind("GraphTypeDef"),
          &gtr);
      graphTypedef.matchAST(Context);
    }
    llvm::outs() << " Done phase 0\n";
    { // 2) Get the graph declarations
      MatchFinder graphDecls;
      GraphTypeMapper gtm(R);
      graphDecls.addMatcher(
          varDecl(
              isExpansionInMainFile(),
              hasType(
                  recordDecl(hasMethod(hasName("getData"))).bind("GraphType")))
              .bind("graphDecl"),
          &gtm);
      graphDecls.matchAST(Context);
      llvm ::outs() << "Found graph types :: " << gtm.graph_types.size()
                    << "\n";
      assert(gtm.graph_types.size() > 0);
      app_data.register_type(*gtm.graph_types.begin(), "CLGraph");
    }
    llvm::outs() << " Done phase 1\n";
    { // 4) Find kernels/operators via do_all calls
      DoAllHandler do_all_handler(R, app_data);
      MatchFinder doAllMatcher;
      doAllMatcher.addMatcher(
          callExpr(callee(functionDecl(hasName("galois::do_all"))),
                   hasArgument(2, hasType(recordDecl().bind("kernelType"))))
              .bind("galoisLoop"),
          &do_all_handler);
      doAllMatcher.matchAST(Context);
      //#################End find kernels/operators via do_all calls
    }
    {
      GetDataHandler gdh(R, app_data);
      MatchFinder getDataFinder;
      getDataFinder.addMatcher(
          memberCallExpr(isExpansionInMainFile(),
                         unless(hasAncestor(methodDecl(isExpansionInMainFile(),
                                                       hasName("operator()")))),
                         callee(functionDecl(hasName("getData"))))
              .bind("callsite"),
          &gdh);
      //         getDataFinder.addMatcher(memberCallExpr(isExpansionInMainFile()
      //         , callee( functionDecl(hasName("getData") ) )
      //         ).bind("callsite"), &gdh);
      getDataFinder.matchAST(Context);
    }
    llvm::outs() << " Done phase 2\n";
#endif
  } // End HandleTranslationUnit
};

class OpenCLHostGenAction : public ASTFrontendAction {
public:
  void EndSourceFileAction() override {
    std::ofstream out_file;
    const char* dir =
        TheRewriter.getSourceMgr()
            .getFileEntryForID(TheRewriter.getSourceMgr().getMainFileID())
            ->getDir()
            ->getName();
    char fname[1024];
    sprintf(fname, "%s/cl_host.cpp", dir);
    out_file.open(fname);
    if (!out_file.is_open()) {
      llvm::outs() << "Unable to open file\n";
    }
    out_file << " //Generated @ " << ASTUtility::get_timestamp() << "\n";
    string s;
    raw_string_ostream sstream(s);
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
        .write(sstream);
    //     llvm::outs() << sstream.str();
    out_file << sstream.str();
    out_file.close();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
                                                 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<OpenCLHostGenConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

/*
 * Entry point.*/
int main(int argc, const char** argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  return Tool.run(newFrontendActionFactory<OpenCLHostGenAction>().get());
}
