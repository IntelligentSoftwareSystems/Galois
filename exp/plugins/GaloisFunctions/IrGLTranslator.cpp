#include <sstream>
#include <climits>
#include <vector>
#include <unordered_set>
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
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  std::unordered_set<const Stmt *> skipStmts;
  std::unordered_set<const Decl *> skipDecls;

public:
  explicit IrGLFunctionsVisitor(CompilerInstance &CI, Rewriter &R) : 
    astContext(&(CI.getASTContext())),
    rewriter(R)
  {}

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLFunctionsVisitor>::TraverseDecl(D);
    if (traverse && D && isa<CXXMethodDecl>(D)) {
      llvm::errs() << "]),\n"; // end ForAll
      llvm::errs() << "])\n"; // end kernel
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLFunctionsVisitor>::TraverseStmt(S);
    if (traverse && S && isa<CXXForRangeStmt>(S)) {
      llvm::errs() << "])\n"; // end ForAll
    }
    return traverse;
  }

  virtual bool VisitFunctionDecl(FunctionDecl* func) {
    std::string funcName = func->getNameAsString();
    llvm::errs() << "\nIrGL Function:\n";
    llvm::errs() << "Kernel(\"" << funcName << "\", [G.param()";
    for (auto param : func->params()) {
      // FIXME: support for multiple arguments
      // FIXME: fix type to node's datatype
      llvm::errs() << ", ('" << param->getOriginalType().getAsString() << " *', 'graph_data')";
    }
    llvm::errs() << "],\n";
    llvm::errs() << "[ForAll(\"vertex\", G.nodes(),\n[\n";
    return true;
  }

  virtual bool VisitCXXForRangeStmt(CXXForRangeStmt* forStmt) {
    skipDecls.insert(forStmt->getLoopVariable());
    DeclStmt *rangeStmt = forStmt->getRangeStmt();
    for (auto decl : rangeStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
      }
    }
    llvm::errs() << "ForAll(\"" << forStmt->getLoopVariable()->getNameAsString() 
      << "\", G.edges(\"node\"),\n[\n";
    return true;
  }

  virtual bool VisitDeclStmt(DeclStmt *declStmt) {
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
        if (skipDecls.find(varDecl) == skipDecls.end()) {
          llvm::errs() << "CDecl([(\"" << varDecl->getType().getAsString() 
            << "\", \"" << varDecl->getNameAsString() << "\", \"\")]),\n";

          if (expr != NULL) {
            SourceRange range(expr->getLocStart(), expr->getLocEnd());
            llvm::errs() << "CBlock([\"" << varDecl->getNameAsString() 
              << " = " << rewriter.getRewrittenText(range) << "\"]),\n";
          }
        }
      }
    }
    return true;
  }

  virtual bool VisitCompoundAssignOperator(CompoundAssignOperator *assignOp) {
    skipStmts.insert(assignOp->getRHS());
    SourceRange range(assignOp->getLocStart(), assignOp->getLocEnd());
    llvm::errs() << "CBlock([\"" << rewriter.getRewrittenText(range) << "\"]),\n";
    return true;
  }

  virtual bool VisitCXXMemberCallExpr(CXXMemberCallExpr *callExpr) {
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      SourceRange range(callExpr->getLocStart(), callExpr->getLocEnd());
      llvm::errs() << "CBlock([\"" << rewriter.getRewrittenText(range) << "\"]),\n";
    }
    return true;
  }

  virtual bool VisitBinaryOperator(BinaryOperator *binaryOp) {
    skipStmts.insert(binaryOp->getLHS());
    skipStmts.insert(binaryOp->getRHS());
    return true;
  }

  virtual bool VisitCastExpr(CastExpr *castExpr) {
    if (skipStmts.find(castExpr) != skipStmts.end()) {
      skipStmts.insert(castExpr->getSubExpr());
    }
    return true;
  }

  virtual bool VisitMaterializeTemporaryExpr(MaterializeTemporaryExpr *tempExpr) {
    if (skipStmts.find(tempExpr) != skipStmts.end()) {
      skipStmts.insert(tempExpr->GetTemporaryExpr());
    }
    return true;
  }
};

class FunctionDeclHandler : public MatchFinder::MatchCallback {
private:
  CompilerInstance &Instance;
  Rewriter &rewriter;
  IrGLFunctionsVisitor visitor;
public:
  FunctionDeclHandler(CompilerInstance &CI, Rewriter &R): Instance(CI), rewriter(R), visitor(CI, R) {}
  virtual void run(const MatchFinder::MatchResult &Results) {
    const CXXMethodDecl* decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("graphOperator");
    if (decl)
      llvm::errs() << "Graph operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
    else {
      decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("nodeOperator");
      if (decl)
        llvm::errs() << "Node operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
      //decl->dump();
      visitor.TraverseDecl(const_cast<CXXMethodDecl *>(decl));
    }
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
    // FIXME: Design a standard to identify graph and node operators
    Matchers.addMatcher(functionDecl(
          hasOverloadedOperatorName("()"),
          hasAnyParameter(hasType(references(recordDecl(matchesName("Galois::Graph")))))).
          bind("graphOperator"), &functionDeclHandler);
    Matchers.addMatcher(functionDecl(
          hasOverloadedOperatorName("()"),
          hasAnyParameter(hasName("vertex"))).
          bind("nodeOperator"), &functionDeclHandler);
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
