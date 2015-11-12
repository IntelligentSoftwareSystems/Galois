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

#define VARIABLE_NAME_CHARACTERS "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
namespace {

class IrGLFunctionsVisitor : public RecursiveASTVisitor<IrGLFunctionsVisitor> {
private:
  ASTContext* astContext;
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  std::unordered_set<const Stmt *> skipStmts;
  std::unordered_set<const Decl *> skipDecls;
  std::map<std::string, std::string> nodeMap;
  bool entryOnce;
  std::stringstream declString, bodyString;
  std::map<std::string, std::string> parameterToTypeMap;

public:
  explicit IrGLFunctionsVisitor(CompilerInstance &CI, Rewriter &R) : 
    astContext(&(CI.getASTContext())),
    rewriter(R),
    entryOnce(false)
  {}
  
  void WriteCBlock(std::string text) {
    assert(text.find("graph.getData") == std::string::npos);

    // replace node variables: array of structures to structure of arrays
    // FIXME: does not handle scoped variables (nesting with same variable name)
    for (auto& nodeVar : nodeMap) {
      std::size_t pos = text.find(nodeVar.first);
      if (pos != std::string::npos) {
        std::size_t end = text.find(".", pos);
        text.erase(pos, end - pos + 1);
        end = text.find_first_not_of(VARIABLE_NAME_CHARACTERS, pos); 
        text.insert(end, "[" + nodeVar.second + "]");
        text.insert(pos, "p_");
      }
    }

    bodyString << "CBlock([\"" << text << "\"]),\n";
  }

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLFunctionsVisitor>::TraverseDecl(D);
    if (traverse && D && isa<FunctionDecl>(D)) {
      bodyString << "]),\n"; // end ForAll
      bodyString << "]),\n"; // end kernel
      for (auto& param : parameterToTypeMap) {
        declString << ", ('" << param.second << " *', 'p_" << param.first << "')";
      }
      declString << "],\n";
      llvm::errs() << declString.str();
      llvm::errs() << bodyString.str();
      entryOnce = false;
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLFunctionsVisitor>::TraverseStmt(S);
    if (traverse && S && isa<CXXForRangeStmt>(S)) {
      bodyString << "]),\n"; // end ForAll
    }
    return traverse;
  }

  virtual bool VisitFunctionDecl(FunctionDecl* func) {
    assert(!entryOnce);
    entryOnce = true;
    skipStmts.clear();
    skipDecls.clear();
    nodeMap.clear();
    declString.str("");
    bodyString.str("");
    std::string funcName = func->getNameAsString();
    llvm::errs() << "\nIrGL Function:\n";
    declString << "Kernel(\"" << funcName << "\", [G.param()";
    bodyString << "[ForAll(\"vertex\", G.nodes(),\n[\n";
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
    bodyString << "ForAll(\"" << forStmt->getLoopVariable()->getNameAsString() 
      << "\", G.edges(\"vertex\"),\n[\n";
    return true;
  }

  virtual bool VisitDeclStmt(DeclStmt *declStmt) {
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
        if (skipDecls.find(varDecl) == skipDecls.end()) {
          std::size_t pos = std::string::npos;
          std::string initText;
          if (expr != NULL) {
            SourceRange range(expr->getLocStart(), expr->getLocEnd());
            initText = rewriter.getRewrittenText(range);
            pos = initText.find("graph.getData");
          }
          if (pos == std::string::npos) {
            bodyString << "CDecl([(\"" << varDecl->getType().getAsString() 
              << "\", \"" << varDecl->getNameAsString() << "\", \"\")]),\n";

            if (!initText.empty()) {
              WriteCBlock(varDecl->getNameAsString() + " = " + initText);
            }
          }
          else {
            // get the first argument to graph.getData()
            std::size_t begin = initText.find("(", pos);
            std::size_t end = initText.find(",", pos);
            std::string index = initText.substr(begin+1, end - begin - 1);
            nodeMap[varDecl->getNameAsString()] = index;
          }
        }
      }
    }
    return true;
  }

  virtual bool VisitBinaryOperator(BinaryOperator *binaryOp) {
    skipStmts.insert(binaryOp->getLHS());
    skipStmts.insert(binaryOp->getRHS());
    if (skipStmts.find(binaryOp) == skipStmts.end()) {
      assert(binaryOp.isAssignmentOp());
      SourceRange range(binaryOp->getLocStart(), binaryOp->getLocEnd());
      WriteCBlock(rewriter.getRewrittenText(range));
    }
    return true;
  }

  virtual bool VisitCXXMemberCallExpr(CXXMemberCallExpr *callExpr) {
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      SourceRange range(callExpr->getLocStart(), callExpr->getLocEnd());
      WriteCBlock(rewriter.getRewrittenText(range));
    }
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

  virtual bool VisitParenExpr(ParenExpr *parenExpr) {
    if (skipStmts.find(parenExpr) != skipStmts.end()) {
      skipStmts.insert(parenExpr->getSubExpr());
    }
    return true;
  }

  virtual bool VisitMemberExpr(MemberExpr *memberExpr) {
    SourceRange range(memberExpr->getBase()->getLocStart(), memberExpr->getBase()->getLocEnd());
    std::string objectName = rewriter.getRewrittenText(range);
    if (nodeMap.find(objectName) != nodeMap.end()) {
      parameterToTypeMap[memberExpr->getMemberNameInfo().getAsString()] = memberExpr->getMemberDecl()->getType().getAsString();
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
