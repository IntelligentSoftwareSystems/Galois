#include <sstream>
#include <climits>
#include <vector>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

/*
 *  * Matchers
 *   */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;

namespace {

class IrGLFunctionsVisitor : public RecursiveASTVisitor<IrGLFunctionsVisitor> {
private:
  ASTContext* astContext;

public:
  explicit IrGLFunctionsVisitor(CompilerInstance &CI) : astContext(&(CI.getASTContext())){}

  virtual bool VisitFunctionDecl(FunctionDecl* func){
    std::string funcName = func->getNameAsString();
    llvm::errs() << "Function Name: " << funcName << "\n";
    return true;
  }
};

class FunctionDeclHandler : public MatchFinder::MatchCallback {
private:
  CompilerInstance &Instance;
  Rewriter &rewriter;
  IrGLFunctionsVisitor visitor;
public:
  FunctionDeclHandler(CompilerInstance &CI, Rewriter &R): Instance(CI), rewriter(R), visitor(CI) {}
  virtual void run(const MatchFinder::MatchResult &Results) {
    const CXXMethodDecl* decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("graphOperator");
    if (decl)
      llvm::errs() << "Graph operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
    else {
      decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("nodeOperator");
      if (decl)
        llvm::errs() << "Node operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
    }
    visitor.TraverseDecl(const_cast<CXXMethodDecl *>(decl));
    //decl->dump();
  }
};

class IrGLFunctionsConsumer : public ASTConsumer {
private:
  CompilerInstance &Instance;
  Rewriter &rewriter;
  MatchFinder Matchers;
  FunctionDeclHandler functionDeclHandler;
public:
  IrGLFunctionsConsumer(CompilerInstance &CI, Rewriter &R): Instance(CI), rewriter(R), functionDeclHandler(CI, R) {
    Matchers.addMatcher(functionDecl(hasOverloadedOperatorName("()"),hasAnyParameter(hasName("graph"))).bind("graphOperator"), &functionDeclHandler);
    Matchers.addMatcher(functionDecl(hasOverloadedOperatorName("()"),hasAnyParameter(hasName("src"))).bind("nodeOperator"), &functionDeclHandler);
  }

  virtual void HandleTranslationUnit(ASTContext &Context){
    Matchers.matchAST(Context);
  }
};

class IrGLFunctionsAction : public PluginASTAction {
private:
  Rewriter rewriter;
protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
    rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<IrGLFunctionsConsumer>(CI, rewriter);
  }

  bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
    return true;
  }
};

}

static FrontendPluginRegistry::Add<IrGLFunctionsAction> 
X("irgl", "translate LLVM AST to IrGL AST");
