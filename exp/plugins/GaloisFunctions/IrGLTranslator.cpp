#include <sstream>
#include <fstream>
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

// FIXME: reinitialize after every main/orchestration code
std::map<std::string, std::string> SharedVariablesToTypeMap;
std::map<std::string, std::vector<std::string> > KernelToArgumentsMap;
std::ofstream Output;

void findAndReplace(std::string &str, 
    const std::string &toReplace, 
    const std::string &replaceWith) 
{
  size_t found = str.find(toReplace);
  if (found != std::string::npos) {
    str = str.replace(found, toReplace.length(), replaceWith);
  }
}

class IrGLOperatorVisitor : public RecursiveASTVisitor<IrGLOperatorVisitor> {
private:
  ASTContext* astContext;
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  std::unordered_set<const Stmt *> skipStmts;
  std::unordered_set<const Decl *> skipDecls;
  std::map<std::string, std::string> nodeMap;
  bool entryOnce;
  std::string funcName;
  std::stringstream declString, bodyString;
  std::map<std::string, std::string> parameterToTypeMap;

public:
  explicit IrGLOperatorVisitor(CompilerInstance &CI, Rewriter &R) : 
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
      while (pos != std::string::npos) {
        std::size_t end = text.find(".", pos);
        text.erase(pos, end - pos + 1);
        end = text.find_first_not_of(VARIABLE_NAME_CHARACTERS, pos); 
        if (end == std::string::npos)
          text.append("[" + nodeVar.second + "]");
        else
          text.insert(end, "[" + nodeVar.second + "]");
        text.insert(pos, "p_");
        pos = text.find(nodeVar.first);
      }
    }

    // FIXME: assumes only one occurrence
    findAndReplace(text, "getEdgeDst", "getAbsDestination");
    findAndReplace(text, "getEdgeData", "getAbsWeight");
    findAndReplace(text, "std::fabs", "fabs");
    // FIXME: should it be only withint quoted strings?
    findAndReplace(text, "\\", "\\\\");

    size_t found = text.find("printf");
    if (found != std::string::npos) {
      bodyString << "CBlock(['" << text << "']),\n";
    } else {
      bodyString << "CBlock([\"" << text << "\"]),\n";
    }
  }

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLOperatorVisitor>::TraverseDecl(D);
    if (traverse && D && isa<CXXMethodDecl>(D)) {
      bodyString << "]),\n"; // end ForAll
      bodyString << "]),\n"; // end kernel
      std::vector<std::string> arguments;
      for (auto& param : parameterToTypeMap) {
        declString << ", ('" << param.second << " *', 'p_" << param.first << "')";
        arguments.push_back(param.first);
        // FIXME: assert type matches if it exists
        SharedVariablesToTypeMap[param.first] = param.second;
      }
      KernelToArgumentsMap[funcName] = arguments;
      declString << "],\n[\n";
      Output << declString.str();
      Output << bodyString.str();
      entryOnce = false;
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLOperatorVisitor>::TraverseStmt(S);
    if (traverse && S && isa<CXXForRangeStmt>(S)) {
      bodyString << "]),\n"; // end ForAll
    }
    return traverse;
  }

  virtual bool VisitCXXMethodDecl(CXXMethodDecl* func) {
    assert(!entryOnce);
    entryOnce = true;
    skipStmts.clear();
    skipDecls.clear();
    nodeMap.clear();
    declString.str("");
    bodyString.str("");
    funcName = func->getParent()->getNameAsString();
    declString << "Kernel(\"" << funcName << "\", [G.param()";
    bodyString << "ForAll(\"vertex\", G.nodes(),\n[\n";
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
            std::string decl = varDecl->getType().getAsString();
            findAndReplace(decl, "GNode", "index_type");
            bodyString << "CDecl([(\"" << decl 
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
      assert(binaryOp->isAssignmentOp());
      SourceRange range(binaryOp->getLocStart(), binaryOp->getLocEnd());
      WriteCBlock(rewriter.getRewrittenText(range));
    }
    return true;
  }

  virtual bool VisitCXXMemberCallExpr(CXXMemberCallExpr *callExpr) {
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      CXXRecordDecl *record = callExpr->getRecordDecl();
      std::string recordName = record->getNameAsString();
      if (recordName.compare("GReducible") == 0) {
        Expr *implicit = callExpr->getImplicitObjectArgument();
        if (CastExpr *cast = dyn_cast<CastExpr>(implicit)) {
          implicit = cast->getSubExpr();
        }
        SourceLocation start;
        if (MemberExpr *member = dyn_cast<MemberExpr>(implicit)) {
          start = member->getMemberLoc();
        } else {
          start = implicit->getLocStart();
        }
        SourceRange range(start, implicit->getLocEnd());
        std::string reduceVar = rewriter.getRewrittenText(range);

        assert(callExpr->getNumArgs() == 1);
        Expr *argument = callExpr->getArg(0);
        SourceRange range2(argument->getLocStart(), argument->getLocEnd());
        std::string var = rewriter.getRewrittenText(range2);

        WriteCBlock("p_" + reduceVar + "[vertex] = " + var);
        parameterToTypeMap[reduceVar] = argument->getType().getUnqualifiedType().getAsString();
      } else {
        SourceRange range(callExpr->getLocStart(), callExpr->getLocEnd());
        WriteCBlock(rewriter.getRewrittenText(range));
      }
    }
    return true;
  }

  virtual bool VisitCallExpr(CallExpr *callExpr) {
    if (skipStmts.find(callExpr) != skipStmts.end()) {
      for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
        skipStmts.insert(callExpr->getArg(i));
      }
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
      parameterToTypeMap[memberExpr->getMemberNameInfo().getAsString()] = 
        memberExpr->getMemberDecl()->getType().getUnqualifiedType().getAsString();
    }
    return true;
  }
};

class IrGLOrchestratorVisitor : public RecursiveASTVisitor<IrGLOrchestratorVisitor> {
private:
  ASTContext* astContext;
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  std::unordered_set<const Stmt *> skipStmts;
  std::unordered_set<const Decl *> skipDecls;
  bool entryOnce;

public:
  explicit IrGLOrchestratorVisitor(CompilerInstance &CI, Rewriter &R) : 
    astContext(&(CI.getASTContext())),
    rewriter(R),
    entryOnce(false)
  {}
  
  void WriteCBlock(std::string text) {
    // FIXME: assumes only one occurrence
    // FIXME: should it be only withint quoted strings?
    findAndReplace(text, "\\", "\\\\");

    size_t found = text.find("printf");
    if (found != std::string::npos) {
      Output << "CBlock(['" << text << "']),\n";
    } else {
      found = text.find("mgpu::Reduce");
      if (found != std::string::npos) {
        Output << "CBlock([\"" << text << "\"], parse=False),\n";
      } else {
        Output << "CBlock([\"" << text << "\"]),\n";
      }
    }
  }

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLOrchestratorVisitor>::TraverseDecl(D);
    if (traverse && D && isa<CXXMethodDecl>(D)) {
      Output << "]),\n"; // end kernel
      entryOnce = false;
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLOrchestratorVisitor>::TraverseStmt(S);
    if (traverse && S && isa<WhileStmt>(S)) {
      Output << "]),\n"; // end DoWhile
    }
    return traverse;
  }

  virtual bool VisitCXXMethodDecl(CXXMethodDecl* func) {
    assert(!entryOnce);
    entryOnce = true;
    skipStmts.clear();
    skipDecls.clear();
    Output << "Kernel(\"" << "gg_main" << "\", ";
    Output << "[GraphParam('hg', True), GraphParam('gg', True)],\n[\n";
    for (auto& var : SharedVariablesToTypeMap) {
      Output << "CDecl((\"Shared<" << var.second << ">\", \"p_" << var.first << "\"";
      Output << ", \"= Shared<" << var.second << "> (hg.nnodes)\")),\n";
    }
    return true;
  }

  virtual bool VisitWhileStmt(WhileStmt* whileStmt) {
    Expr *cond = whileStmt->getCond();
    skipStmts.insert(cond);
    SourceRange range(cond->getLocStart(), cond->getLocEnd());
    std::string condText = rewriter.getRewrittenText(range);
    if (condText.empty()) condText = "true";
    Output << "DoWhile(\"" << condText << "\",\n[\n";
    return true;
  }

  virtual bool VisitDeclStmt(DeclStmt *declStmt) {
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
        if (skipDecls.find(varDecl) == skipDecls.end()) {
          Output << "CDecl([(\"" << varDecl->getType().getAsString() 
            << "\", \"" << varDecl->getNameAsString() << "\", \"\")]),\n";
          if (expr != NULL) {
            SourceRange range(expr->getLocStart(), expr->getLocEnd());
            std::string initText = rewriter.getRewrittenText(range);
            WriteCBlock(varDecl->getNameAsString() + " = " + initText);
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
      bool skip = false;
      Expr *rhs = binaryOp->getRHS();
      if (CastExpr *cast = dyn_cast<CastExpr>(rhs)) {
        rhs = cast->getSubExpr();
      }
      if (CXXMemberCallExpr *callExpr = dyn_cast<CXXMemberCallExpr>(rhs)) {
        CXXRecordDecl *record = callExpr->getRecordDecl();
        std::string recordName = record->getNameAsString();
        if (recordName.compare("GReducible") == 0) {
          skip = true;
          assert(!record->getMethodDecl()->getNameAsString().compare("reduce"));

          Expr *implicit = callExpr->getImplicitObjectArgument();
          if (CastExpr *cast = dyn_cast<CastExpr>(implicit)) {
            implicit = cast->getSubExpr();
          }
          SourceLocation start;
          if (MemberExpr *member = dyn_cast<MemberExpr>(implicit)) {
            start = member->getMemberLoc();
          } else {
            start = implicit->getLocStart();
          }
          SourceRange range(start, implicit->getLocEnd());
          std::string reduceVar = rewriter.getRewrittenText(range);

          SourceRange range2(binaryOp->getLHS()->getLocStart(), binaryOp->getLHS()->getLocEnd());
          std::string var = rewriter.getRewrittenText(range2);

          std::ostringstream reduceCall;
          // FIXME: choose matching reduction operation
          reduceCall << "mgpu::Reduce(p_" << reduceVar << ".gpu_rd_ptr(), hg.nnodes";
          reduceCall << ", (" << SharedVariablesToTypeMap[reduceVar] << ")0";
          reduceCall << ", mgpu::maximum<" << SharedVariablesToTypeMap[reduceVar] << ">()";
          reduceCall << ", (" << SharedVariablesToTypeMap[reduceVar] << "*)0";
          reduceCall << ", &" << var << ", *mgc)";

          // FIXME: Add "parse=False"
          WriteCBlock(reduceCall.str());
        }
      }
      if (!skip) {
        SourceRange range(binaryOp->getLocStart(), binaryOp->getLocEnd());
        WriteCBlock(rewriter.getRewrittenText(range));
      }
    }
    return true;
  }

  virtual bool VisitCallExpr(CallExpr *callExpr) {
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      SourceRange range(callExpr->getLocStart(), callExpr->getLocEnd());
      std::string text = rewriter.getRewrittenText(range);
      std::size_t begin = text.find("do_all");
      if (begin == std::string::npos) begin = text.find("for_each");
      if (begin == std::string::npos) {
        bool skip = false;
        Expr *expr = callExpr;
        if (CastExpr *cast = dyn_cast<CastExpr>(expr)) {
          expr = cast->getSubExpr();
        }
        if (CXXMemberCallExpr *memberCallExpr = dyn_cast<CXXMemberCallExpr>(expr)) {
          CXXRecordDecl *record = memberCallExpr->getRecordDecl();
          std::string recordName = record->getNameAsString();
          if (recordName.compare("GReducible") == 0) {
            skip = true;
          }
        }
        if (!skip) WriteCBlock(text);
      } else {
        begin = text.find(",", begin);
        ++begin;
        std::size_t end = text.find("(", begin);
        --end;
        // remove whitespace
        while (text[begin] == ' ') ++begin;
        while (text[end] == ' ') --end; 
        std::string kernelName = text.substr(begin, end-begin+1);
        Output << "Invoke(\"" << kernelName << "\", ";
        Output << "(\"gg\"";
        auto& arguments = KernelToArgumentsMap[kernelName];
        for (auto& argument : arguments) {
          Output << ", \"p_" << argument << ".gpu_wr_ptr()\"";
        }
        Output << ")),\n";
      }
    } else {
      for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
        skipStmts.insert(callExpr->getArg(i));
      }
    }
    return true;
  }

  virtual bool VisitCXXMemberCallExpr(CXXMemberCallExpr *callExpr) {
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      CXXRecordDecl *record = callExpr->getRecordDecl();
      std::string recordName = record->getNameAsString();
      if (recordName.compare("GReducible") == 0) {
        skipStmts.insert(callExpr);
      }
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
};
  
class FunctionDeclHandler : public MatchFinder::MatchCallback {
private:
  CompilerInstance &Instance;
  Rewriter &rewriter;
  IrGLOperatorVisitor operatorVisitor;
  IrGLOrchestratorVisitor orchestratorVisitor;
public:
  FunctionDeclHandler(CompilerInstance &CI, Rewriter &R): 
    Instance(CI), 
    rewriter(R), 
    operatorVisitor(CI, R),
    orchestratorVisitor(CI, R) 
  {}
  virtual void run(const MatchFinder::MatchResult &Results) {
    const CXXMethodDecl* decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("graphOperator");
    if (decl) {
      //llvm::errs() << "\nGraph Operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
      //decl->dump();
      //llvm::errs() << "\nIrGL Orchestrator Kernel:\n";
      orchestratorVisitor.TraverseDecl(const_cast<CXXMethodDecl *>(decl));
    } else {
      decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("vertexOperator");
      if (decl) {
        //llvm::errs() << "\nVertex Operator:\n" << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
        //decl->dump();
        //llvm::errs() << "\nIrGL Kernel for Vertex Operators:\n";
        operatorVisitor.TraverseDecl(const_cast<CXXMethodDecl *>(decl));
      }
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
          hasAnyParameter(hasName("graph_main"))).
          bind("graphOperator"), &functionDeclHandler);
    Matchers.addMatcher(functionDecl(
          hasOverloadedOperatorName("()"),
          hasAnyParameter(hasName("vertex"))).
          bind("vertexOperator"), &functionDeclHandler);
  }

  virtual void HandleTranslationUnit(ASTContext &Context){
    // FIXME: Build file name from source file
    Output.open("irgl.py");
    Output << "from gg.ast import *\n";
    Output << "from gg.lib.graph import Graph\n";
    Output << "from gg.ast.params import GraphParam\n";
    Output << "import cgen\n";
    Output << "G = Graph(\"graph\")\n";
    Output << "ast = Module([\n";
    Matchers.matchAST(Context);
    Output << "])\n";
    Output.close();
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
