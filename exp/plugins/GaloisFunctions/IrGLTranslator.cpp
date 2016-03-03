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
#define COMPUTE_FILENAME "._compute_kernels"
#define ORCHESTRATOR_FILENAME "._orchestrator_kernel"

namespace {

// finds and replaces all occurrences
// ensures progress even if replaceWith contains toReplace
void findAndReplace(std::string &str, 
    const std::string &toReplace, 
    const std::string &replaceWith) 
{
  std::size_t pos = 0;
  while (1) {
    std::size_t found = str.find(toReplace, pos);
    if (found != std::string::npos) {
      str = str.replace(found, toReplace.length(), replaceWith);
      pos = found + replaceWith.length();
    } else {
      break;
    }
  }
}

class IrGLOperatorVisitor : public RecursiveASTVisitor<IrGLOperatorVisitor> {
private:
  //ASTContext* astContext;
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  bool isTopological;
  std::unordered_set<const Stmt *> skipStmts;
  std::unordered_set<const Decl *> skipDecls;
  std::map<std::string, std::string> nodeMap;
  std::string funcName;
  std::stringstream declString, bodyString;
  std::map<std::string, std::string> parameterToTypeMap;
  std::unordered_set<std::string> symbolTable; // FIXME: does not handle scoped variables (nesting with same variable name)
  std::ofstream Output;

  std::map<std::string, std::string> &GlobalVariablesToTypeMap;
  std::map<std::string, std::string> &SharedVariablesToTypeMap;
  std::map<std::string, std::vector<std::string> > &KernelToArgumentsMap;

public:
  explicit IrGLOperatorVisitor(ASTContext *context, Rewriter &R, 
      bool isTopo,
      std::map<std::string, std::string> &globalVariables,
      std::map<std::string, std::string> &sharedVariables,
      std::map<std::string, std::vector<std::string> > &kernels) : 
    //astContext(context),
    rewriter(R),
    isTopological(isTopo),
    GlobalVariablesToTypeMap(globalVariables),
    SharedVariablesToTypeMap(sharedVariables),
    KernelToArgumentsMap(kernels)
  {}
  
  virtual ~IrGLOperatorVisitor() {}

  std::string FormatCBlock(std::string text) {
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
    // FIXME: use & (address-of) operator for atomicu variables in atomicAdd() etc

    while (1) {
      std::size_t found = text.find("nout(");
      if (found != std::string::npos) {
        std::size_t begin = text.find(",", found);
        std::size_t end = text.find(",", begin+1);
        std::string var = text.substr(begin+1, end - begin - 1);
        std::string replace = "graph.getOutDegree(" + var + ")";
        end = text.find(")", end);
        text = text.replace(found, end - found + 1, replace);
      } else {
        break;
      }
    } 

    findAndReplace(text, "getEdgeDst", "getAbsDestination");
    findAndReplace(text, "getEdgeData", "getAbsWeight");
    findAndReplace(text, "std::fabs", "fabs");
    findAndReplace(text, "ctx.push", "WL.push"); // FIXME: insert arguments within quotes
    findAndReplace(text, "\\", "\\\\"); // FIXME: should it be only within quoted strings?

    return text;
  }

  void WriteCBlock(std::string text) {
    text = FormatCBlock(text);
    std::size_t found = text.find("printf");
    if (found != std::string::npos) {
      bodyString << "CBlock(['" << text << "']),\n";
    } else {
      found = text.find("WL.push");
      if (found != std::string::npos) {
        bodyString << text << ",\n";
      } else {
        bodyString << "CBlock([\"" << text << "\"]),\n";
      }
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
      Output.open(COMPUTE_FILENAME, std::ios::app);
      Output << declString.str();
      Output << bodyString.str();
      Output.close();
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLOperatorVisitor>::TraverseStmt(S);
    if (traverse && S) {
      if (isa<CXXForRangeStmt>(S)) {
        bodyString << "]),\n"; // end ForAll
      } else if (isa<IfStmt>(S)) {
        bodyString << "]),\n"; // end If
      }
    }
    return traverse;
  }

  virtual bool VisitCXXMethodDecl(CXXMethodDecl* func) {
    assert(declString.rdbuf()->in_avail() == 0);
    declString.str("");
    bodyString.str("");
    funcName = func->getParent()->getNameAsString();
    declString << "Kernel(\"" << funcName << "\", [G.param()";
    // FIXME: assuming first parameter is GNode
    std::string vertexName = func->getParamDecl(0)->getNameAsString();
    if (isTopological) {
      bodyString << "ForAll(\"" << vertexName << "\", G.nodes(),\n[\n";
    } else {
      bodyString << "ForAll(\"wlvertex\", WL.items(),\n[\n";
      bodyString << "CDecl([(\"int\", \"" << vertexName << "\", \"\")]),\n";
      bodyString << "CDecl([(\"bool\", \"pop\", \"\")]),\n";
      bodyString << "WL.pop(\"pop\", \"wlvertex\", \"" << vertexName << "\"),\n";
    }
    symbolTable.insert(vertexName);
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
    symbolTable.insert(forStmt->getLoopVariable()->getNameAsString());
    const Expr *expr = forStmt->getRangeInit();
    SourceRange range(expr->getLocStart(), expr->getLocEnd());
    std::string vertexName = rewriter.getRewrittenText(range);
    std::size_t found = vertexName.find("graph.edges");
    assert(found != std::string::npos);
    std::size_t begin = vertexName.find("(", found);
    std::size_t end = vertexName.find(",", begin);
    vertexName = vertexName.substr(begin+1, end - begin - 1);
    bodyString << "ForAll(\"" << forStmt->getLoopVariable()->getNameAsString() 
      << "\", G.edges(\"" << vertexName << "\"),\n[\n";
    return true;
  }

  virtual bool VisitDeclStmt(DeclStmt *declStmt) {
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
        if (skipDecls.find(varDecl) == skipDecls.end()) {
          symbolTable.insert(varDecl->getNameAsString());
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
            if (end == std::string::npos) {
              end = initText.find(")", pos);
            }
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

  virtual bool VisitCallExpr(CallExpr *callExpr) {
    symbolTable.insert(callExpr->getCalleeDecl()->getAsFunction()->getNameAsString());
    if (skipStmts.find(callExpr) == skipStmts.end()) {
      CXXMemberCallExpr *cxxCallExpr = dyn_cast<CXXMemberCallExpr>(callExpr);
      CXXRecordDecl *record = NULL;
      if (cxxCallExpr) {
        record = cxxCallExpr->getRecordDecl();
      }
      if (record && 
          (record->getNameAsString().compare("GReducible") == 0)) {
        Expr *implicit = cxxCallExpr->getImplicitObjectArgument();
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
    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
      skipStmts.insert(callExpr->getArg(i));
    }
    return true;
  }

  virtual bool VisitIfStmt(IfStmt *ifStmt) {
    const Expr *expr = ifStmt->getCond();
    SourceRange range(expr->getLocStart(), expr->getLocEnd());
    std::string text = FormatCBlock(rewriter.getRewrittenText(range));
    bodyString << "If(\"" << text << "\",\n[\n";
    skipStmts.insert(expr);
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
      std::string type = 
        memberExpr->getMemberDecl()->getType().getUnqualifiedType().getAsString();
      if (type.compare(0, 11, "std::atomic") == 0) {
        std::size_t begin = type.find("<");
        std::size_t end = type.find_last_of(">");
        type = type.substr(begin+1, end - begin - 1);
      }
      parameterToTypeMap[memberExpr->getMemberNameInfo().getAsString()] = type;
    }
    return true;
  }

  virtual bool VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    const ValueDecl *decl = declRefExpr->getDecl();
    const std::string &varName = decl->getNameAsString();
    if (symbolTable.find(varName) == symbolTable.end()) {
      if (GlobalVariablesToTypeMap.find(varName) == GlobalVariablesToTypeMap.end()) {
        // FIXME: get initialization value if any
        GlobalVariablesToTypeMap[varName] = decl->getType().getAsString();
      }
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
  std::unordered_set<std::string> symbolTable; // FIXME: does not handle scoped variables (nesting with same variable name)
  std::ofstream Output;

  std::map<std::string, std::string> GlobalVariablesToTypeMap;
  std::map<std::string, std::string> SharedVariablesToTypeMap;
  std::map<std::string, std::vector<std::string> > KernelToArgumentsMap;

public:
  explicit IrGLOrchestratorVisitor(ASTContext *context, Rewriter &R) : 
    astContext(context),
    rewriter(R)
  {}
  
  virtual ~IrGLOrchestratorVisitor() {}

  void GenerateIrGLASTToFile(std::string filename) {
    std::ofstream IrGLAST;
    IrGLAST.open(filename);
    IrGLAST << "from gg.ast import *\n";
    IrGLAST << "from gg.lib.graph import Graph\n";
    IrGLAST << "from gg.lib.wl import Worklist\n";
    IrGLAST << "from gg.ast.params import GraphParam\n";
    IrGLAST << "import cgen\n";
    IrGLAST << "G = Graph(\"graph\")\n";
    IrGLAST << "WL = Worklist()\n";
    IrGLAST << "ast = Module([\n";
    IrGLAST << "CBlock([cgen.Include(\"kernels/reduce.cuh\", "
      << "system = False)], parse = False),\n";
    for (auto& var : SharedVariablesToTypeMap) {
      std::string upper = var.first;
      std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
      IrGLAST << "CDeclGlobal([(\"" << var.second 
        << " *\", \"P_" << upper << "\", \"\")]),\n";
    }
    for (auto& var : GlobalVariablesToTypeMap) {
      size_t found = var.second.find("Galois::");
      if (found == std::string::npos) {
        found = var.second.find("cll::opt");
        std::string type;
        if (found == std::string::npos) {
          type = var.second;
        } else {
          size_t start = var.second.find("<", found);
          // FIXME: nested < < > >
          size_t end = var.second.find(">", found);
          type = var.second.substr(start+1, end-start-1);
        }
        // FIXME: initialization value?
        IrGLAST << "CDeclGlobal([(\"extern " << type 
          << "\", \"" << var.first << "\", \"\")]),\n";
      }
    }

    std::ifstream compute(COMPUTE_FILENAME);
    IrGLAST << compute.rdbuf(); 
    compute.close();
    remove(COMPUTE_FILENAME);

    IrGLAST << "Kernel(\"" << "gg_main" << "\", ";
    IrGLAST << "[GraphParam('hg', True), GraphParam('gg', True)],\n[\n";
    // should wait for all operators to be generated before reading shared variables
    for (auto& var : SharedVariablesToTypeMap) {
      IrGLAST << "CDecl((\"Shared<" << var.second << ">\", \"p_" << var.first << "\"";
      IrGLAST << ", \"= Shared<" << var.second << "> (hg.nnodes)\")),\n";
    }
    // FIXME: initialize shared variables to zero if they are atomic
    std::ifstream orchestrator(ORCHESTRATOR_FILENAME);
    IrGLAST << orchestrator.rdbuf();
    orchestrator.close();
    for (auto& var : SharedVariablesToTypeMap) {
      std::string upper = var.first;
      std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
      IrGLAST << "CBlock([\"" << "P_" << upper << " = p_" 
        << var.first << ".cpu_rd_ptr()" << "\"]),\n";
    }
    IrGLAST << "]),\n"; // end kernel
    remove(ORCHESTRATOR_FILENAME);

    IrGLAST << "])\n"; // end Module
    IrGLAST.close();
  }

  void WriteCBlock(std::string text) {
    // FIXME: should it be only within quoted strings?
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
      Output.close();

      CXXMethodDecl *method = dyn_cast<CXXMethodDecl>(D);
      // FIXME: Build file name from source file
      GenerateIrGLASTToFile(method->getParent()->getNameAsString() + ".py");
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
    assert(symbolTable.empty());
    Output.open(ORCHESTRATOR_FILENAME);
    skipStmts.clear();
    skipDecls.clear();
    symbolTable.clear();
    symbolTable.insert("_graph");
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
          symbolTable.insert(varDecl->getNameAsString());
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
      assert(binaryOp->isAssignmentOp());
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
          //assert(!record->getMethodDecl()->getNameAsString().compare("reduce"));

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
    symbolTable.insert(callExpr->getCalleeDecl()->getAsFunction()->getNameAsString());
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
        // generate kernel for operator
        bool isTopological = true;
        assert(callExpr->getNumArgs() > 1);
        Expr *op = callExpr->getArg(1);
        RecordDecl *record = op->getType().getTypePtr()->getAsStructureType()->getDecl();
        CXXRecordDecl *cxxrecord = dyn_cast<CXXRecordDecl>(record);
        for (auto method : cxxrecord->methods()) {
          // FIXME: handle overloading/polymorphism
          if (method->getNameAsString().compare("operator()") == 0) {
            // FIXME: be precise in determining data-driven algorithms
            if (callExpr->getNumArgs() > 2) {
              std::string type = callExpr->getArg(2)->getType().getAsString();
              if (type.find("Galois::WorkList") != std::string::npos) {
                isTopological = false;
              }
            }
            //llvm::errs() << "\nVertex Operator:\n" << 
            //  rewriter.getRewrittenText(method->getSourceRange()) << "\n";
            //method->dump();
            // FIXME: handle arguments to object of CXXRecordDecl?
            IrGLOperatorVisitor operatorVisitor(astContext, rewriter,
                isTopological,
                GlobalVariablesToTypeMap,
                SharedVariablesToTypeMap,
                KernelToArgumentsMap);
            operatorVisitor.TraverseDecl(method);
            break;
          }
        }

        if (!isTopological) { // data driven
          // FIXME: generate precise initial worklist
          std::ofstream compute;
          compute.open(COMPUTE_FILENAME, std::ios::app);
          compute << "Kernel(\"__init_worklist__\", [G.param()],\n[\n";
          compute << "ForAll(\"vertex\", G.nodes(),\n[\n";
          compute << "WL.push(\"vertex\"),\n";
          compute << "]),\n";
          compute << "]),\n";
          compute.close();

          Output << "Pipe([\n";
          Output << "Invoke(\"__init_worklist__\", (\"gg\", )),\n";
          Output << "Pipe([\n";
        }

        // generate call to kernel
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

        if (!isTopological) { // data driven
          Output << "], ),\n";
          Output << "], once=True, wlinit=WLInit(\"hg.nedges\", [])),\n";
        }
      }
    }
    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
      skipStmts.insert(callExpr->getArg(i));
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

  virtual bool VisitDeclRefExpr(DeclRefExpr *declRefExpr) {
    const ValueDecl *decl = declRefExpr->getDecl();
    const std::string &varName = decl->getNameAsString();
    if (symbolTable.find(varName) == symbolTable.end()) {
      if (GlobalVariablesToTypeMap.find(varName) == GlobalVariablesToTypeMap.end()) {
        GlobalVariablesToTypeMap[varName] = decl->getType().getAsString();
      }
    }
    return true;
  }
};
  
class FunctionDeclHandler : public MatchFinder::MatchCallback {
private:
  ASTContext *astContext;
  Rewriter &rewriter;
public:
  FunctionDeclHandler(ASTContext *context, Rewriter &R): 
    astContext(context),
    rewriter(R)
  {}
  virtual void run(const MatchFinder::MatchResult &Results) {
    const CXXMethodDecl* decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("graphAlgorithm");
    if (decl) {
      //llvm::errs() << "\nGraph Operator:\n" << 
      //  rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
      //decl->dump();
      IrGLOrchestratorVisitor orchestratorVisitor(astContext, rewriter);
      orchestratorVisitor.TraverseDecl(const_cast<CXXMethodDecl *>(decl));
    }
  }
};

class IrGLFunctionsConsumer : public ASTConsumer {
private:
  //Rewriter &rewriter;
  MatchFinder Matchers;
  FunctionDeclHandler functionDeclHandler;
public:
  IrGLFunctionsConsumer(CompilerInstance &CI, Rewriter &R): 
    //rewriter(R), 
    functionDeclHandler(&(CI.getASTContext()), R) 
  {
    Matchers.addMatcher(functionDecl(
          hasOverloadedOperatorName("()"),
          hasAnyParameter(hasName("_graph"))).
          bind("graphAlgorithm"), &functionDeclHandler);
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
