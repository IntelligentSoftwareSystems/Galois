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

#include "ClangUtils.h"

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

class OpenCLCodeGenConsumer : public ASTConsumer {
private:
  CompilerInstance& Instance;
  std::set<std::string> ParsedTemplates;
  MatchFinder Matchers;
  Rewriter& R;

public:
  OpenCLCodeGenConsumer(CompilerInstance& Instance,
                        std::set<std::string>& ParsedTemplates, Rewriter& _R)
      : Instance(Instance), ParsedTemplates(ParsedTemplates), R(_R) {
    llvm::outs() << "=============================Created "
                    "OpenCLCodeGenConsumer===========================\n";
  }
  /********************************************************************************************************
   * 1) Find all the do_all calls in the main-file.
   * 2) For each do_all call, go over the class that defines the operator
   * 3) Add call to the generated operator implementation in OpenCL
   *
   ********************************************************************************************************/

  virtual void HandleTranslationUnit(ASTContext& Context) {
    ast_utility.init(&R);
    llvm::outs() << "=============================InHandleTranslationUnit======"
                    "=====================\n";
    CXXRecordDecl* graphClass = nullptr;

    //      std::map<const Type * , string> type_map;
    {
      MatchFinder ndFinder;
      NodeDataGen ndGen(R, Context);
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
    {
        // Find graph instance to create table for type replacement
        /*GraphCollector gc(R);
        MatchFinder gFinder;
        gFinder.addMatcher(varDecl(isExpansionInMainFile(),
        hasType(recordDecl(hasMethod(hasName("getData")) ).bind("graphClass"))),
        &gc); gFinder.matchAST(Context); graphClass=gc.graphClass;*/
    } { // Perform code generation.
      OpenCLDeviceHandler clDevHandler(R, graphClass);
      MatchFinder opFinder;
      opFinder.addMatcher(
          methodDecl(isExpansionInMainFile(), hasName("operator()"))
              .bind("kernel"),
          &clDevHandler);
      opFinder.matchAST(Context);
    }

  } // End HandleTranslationUnit
};
/********************************************************************************************************
 * The plugin AST action.
 *
 ********************************************************************************************************/
class OpenCLCodeGenAction : public PluginASTAction {
private:
  std::set<std::string> ParsedTemplates;
  Rewriter TheRewriter;

protected:
  void EndSourceFileAction() override {
    llvm::outs() << "EndSourceFileAction() :: Returning\n";
  }
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
                                                 llvm::StringRef) override {
    llvm::outs() << "CreateASTConsumer() :: Entered\n";
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    llvm::outs() << "CreateASTConsumer() :: Returning\n";
    return llvm::make_unique<OpenCLCodeGenConsumer>(CI, ParsedTemplates,
                                                    TheRewriter);
  }

  bool ParseArgs(const CompilerInstance& CI,
                 const std::vector<std::string>& args) override {
    return true;
  }
};

} // End anonymous namespace
/********************************************************************************************************
 * Register the plugin. The plugin can be invoked by
 * ./clang -cc1 -load ../lib/OpenCLCodeGen.so -plugin opencl-analysis <FILENAME>
 *
 ********************************************************************************************************/
static FrontendPluginRegistry::Add<OpenCLCodeGenAction>
    X("opencl-device-codegen", "Galois OpenCL codegen for device");
