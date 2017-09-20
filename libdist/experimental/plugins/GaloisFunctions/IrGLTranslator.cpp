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

#define __GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__ "local_"
#define __GALOIS_ACCUMULATOR_TYPE__ "galois::DGAccumulator"

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
  std::string accumulatorName;
  std::stringstream declString, varDeclString, bodyString;
  std::map<std::string, std::string> parameterToTypeMap;
  std::map<std::string, std::string> globalVariableToTypeMap;
  std::unordered_set<std::string> symbolTable; // FIXME: does not handle scoped variables (nesting with same variable name)
  std::ofstream Output;

  std::map<std::string, std::string> &SharedVariablesToTypeMap;
  std::map<std::string, std::pair<std::vector<std::string>, std::vector<std::string> > > &KernelToArgumentsMap;
  std::set<std::string> &KernelsHavingReturnValue;

  unsigned conditional; // if (generated) code is enclosed within an if-condition
  bool conditional_pop; // if (generated) code is enclosed within an (generated) if-condition for pop

public:
  explicit IrGLOperatorVisitor(ASTContext *context, Rewriter &R, 
      bool isTopo,
      std::string accumName,
      std::map<std::string, std::string> &sharedVariables,
      std::map<std::string, std::pair<std::vector<std::string>, std::vector<std::string> > > &kernels,
      std::set<std::string> &returnKernels) : 
    //astContext(context),
    rewriter(R),
    isTopological(isTopo),
    accumulatorName(accumName),
    SharedVariablesToTypeMap(sharedVariables),
    KernelToArgumentsMap(kernels),
    KernelsHavingReturnValue(returnKernels),
    conditional(0),
    conditional_pop(false)
  {}
  
  virtual ~IrGLOperatorVisitor() {}

  std::string FormatCBlock(std::string text) {
    assert(text.find("graph->getData") == std::string::npos);

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

    /*while (1) {
      std::size_t found = text.find("nout(");
      if (found != std::string::npos) {
        std::size_t begin = text.find(",", found);
        std::size_t end = text.find(",", begin+1);
        std::string var = text.substr(begin+1, end - begin - 1);
        std::string replace = "graph->getOutDegree(" + var + ")";
        end = text.find(")", end);
        text = text.replace(found, end - found + 1, replace);
      } else {
        break;
      }
    }*/

    // replace std::distance(graph->edge_begin(NODE), graph->edge_end(NODE)) with getOutDegree(NODE)
    while (1) {
      std::string distance = "std::distance(graph->edge_begin(";
      std::size_t found = text.find(distance);
      if (found != std::string::npos) {
        std::size_t begin = found+distance.length();
        std::size_t end = text.find(")", begin);
        std::string var = text.substr(begin, end - begin);
        std::string replace = "graph.getOutDegree(" + var + ")";
        assert(text.find("graph->edge_end(" + var + ")", end) != std::string::npos);
        end = text.find(")", end+1); // graph->edge_end() 
        end = text.find(")", end+1); // std::distance()
        text = text.replace(found, end - found + 1, replace);
      } else {
        break;
      }
    } 

    // replace ATOMIC_VAR[NODE].exchange(VALUE) with atomicExch(&ATOMIC_VAR[NODE], VALUE)
    while (1) {
      std::string exchange = ".exchange(";
      std::size_t found = text.find(exchange);
      if (found != std::string::npos) {
        std::size_t begin = found+exchange.length();
        std::size_t end = text.find(")", begin);
        std::string value = text.substr(begin, end - begin);
        begin = text.rfind("p_", found-1); 
        std::string var = text.substr(begin, found - begin);
        std::string replace = "atomicExch(&" + var + ", " + value + ")";
        text = text.replace(begin, end - begin + 1, replace);
      } else {
        break;
      }
    }

    // use & (address-of) operator for atomic variables in atomicAdd() etc
    {
      std::size_t start = 0;
      while (1) {
        std::size_t found = text.find("galois::atomic", start);
        if (found != std::string::npos) {
          std::size_t replace = text.find("(", start);
          text.insert(replace+1, "&");
          start = replace;
        } else {
          break;
        }
      }
      findAndReplace(text, "galois::atomic", "atomic");
    }
    {
      std::size_t start = 0;
      while (1) {
        std::size_t found = text.find("galois::min", start);
        if (found != std::string::npos) {
          std::size_t replace = text.find("(", start);
          text.insert(replace+1, "&");
          start = replace;
        } else {
          break;
        }
      }
      findAndReplace(text, "galois::min", "atomicMin");
    }
    {
      std::size_t start = 0;
      while (1) {
        std::size_t found = text.find("galois::add", start);
        if (found != std::string::npos) {
          std::size_t replace = text.find("(", start);
          text.insert(replace+1, "&");
          start = replace;
        } else {
          break;
        }
      }
      findAndReplace(text, "galois::add", "atomicAdd");
    }

    {
      std::size_t found = text.find("ctx.push(");
      if (found != std::string::npos) {
        std::size_t begin = text.rfind("(");
        std::size_t end = text.find(")", begin);
        std::string var = text.substr(begin+1, end - begin - 1);
        text = "WL.push(\"" + var + "\")";
      }
    }

    // replace GRAPH->getGID(VAR) with GRAPH.node_data[VAR]
    while (1) {
      std::size_t found = text.find("->getGID");
      if (found != std::string::npos) {
        std::size_t begin = text.find("(", found);
        std::size_t end = text.find(")", begin);
        std::string var = text.substr(begin+1, end - begin - 1);
        std::string replace = ".node_data[" + var + "]";
        text = text.replace(found, end - found + 1, replace);
      } else {
        break;
      }
    }

    findAndReplace(text, " this->", " "); // because IrGL is not aware of the "this" variable
    findAndReplace(text, "->getEdgeDst", ".getAbsDestination");
    findAndReplace(text, "->getEdgeData", ".getAbsWeight");
    findAndReplace(text, "std::fabs", "fabs");
    findAndReplace(text, "\\", "\\\\"); // FIXME: should it be only within quoted strings?

    if (!accumulatorName.empty() && (text.find(accumulatorName) == 0)) {
      std::size_t begin = text.find("=");
      std::size_t end = text.find(";", begin);
      std::string value = text.substr(begin+1, end - begin - 3);
      text = "ReturnFromParallelFor(\"" + value + "\")";
    }

    return text;
  }

  void WriteCBlock(std::string text) {
    text = FormatCBlock(text);
    std::size_t found = text.find("printf");
    if (found != std::string::npos) {
      bodyString << "CBlock(['" << text << "']),\n";
    } else {
      if (text.find("WL.push") != std::string::npos) {
        bodyString << text << ",\n";
      } else if (text.find("ReturnFromParallelFor") != std::string::npos) {
        bodyString << text << ",\n";
      } else {
        bodyString << "CBlock([\"" << text << "\"]),\n";
      }
    }
  }

  std::string FormatType(std::string text) {
    findAndReplace(text, "GNode", "index_type");
    findAndReplace(text, "_Bool", "bool");

    // remove reference (&)
    if (text.rfind("&") == (text.length() - 1)) {
      text = text.substr(0, text.length() - 1);
    }

    // get basic type
    if ((text.find("std::atomic") == 0) || (text.find("struct std::atomic") == 0)) {
      std::size_t begin = text.find("<");
      std::size_t end = text.find_last_of(">");
      text = text.substr(begin+1, end - begin - 1);
    }
    if (text.find("cll::opt") == 0) {
      std::size_t begin = text.find("<");
      std::size_t end = text.find_last_of(">");
      text = text.substr(begin+1, end - begin - 1);
    }

    return text;
  }

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLOperatorVisitor>::TraverseDecl(D);
    if (traverse && D && isa<CXXMethodDecl>(D)) {
      if (conditional_pop) {
        bodyString << "]),\n"; // end If "pop" (no edge-iterator)
        conditional_pop = false;
      }
      assert(conditional == 0);
      bodyString << "]),\n"; // end ForAll
      bodyString << "]),\n"; // end Kernel
      std::vector<std::string> arguments, globalArguments;
      for (auto& global : globalVariableToTypeMap) {
        declString << ", ('" << global.second << "', '" << global.first << "')";
        globalArguments.push_back(global.first);
      }
      for (auto& param : parameterToTypeMap) {
        declString << ", ('" << param.second << " *', 'p_" << param.first << "')";
        arguments.push_back(param.first);
        // FIXME: assert type matches if it exists
        SharedVariablesToTypeMap[param.first] = param.second;
      }
      KernelToArgumentsMap[funcName] = std::make_pair(globalArguments, arguments);
      if (!accumulatorName.empty()) KernelsHavingReturnValue.insert(funcName);
      declString << "],\n[\n";
      Output.open(COMPUTE_FILENAME, std::ios::app);
      Output << declString.str();
      Output << varDeclString.str();
      Output << bodyString.str();
      Output.close();
    }
    return traverse;
  }

  virtual bool TraverseStmt(Stmt *S) {
    bool traverse = RecursiveASTVisitor<IrGLOperatorVisitor>::TraverseStmt(S);
    if (traverse && S) {
      if (isa<CXXForRangeStmt>(S) || isa<ForStmt>(S)) {
        bodyString << "]),\n"; // end ForAll
        bodyString << "),\n"; // end ClosureHint
        // FIXME: assert that there is no code after the edge-iterator
      } else if (isa<IfStmt>(S)) {
        if (conditional > 0) {
          bodyString << "]),\n"; // end If
          --conditional;
        }
      }
    }
    return traverse;
  }

  virtual bool VisitCXXMethodDecl(CXXMethodDecl* func) {
    assert(declString.rdbuf()->in_avail() == 0);
    declString.str("");
    bodyString.str("");
    funcName = func->getParent()->getNameAsString();
    declString << "Kernel(\"" << funcName << "\", [G.param(), ('unsigned int', '__nowned')";
    // FIXME: assuming first parameter is GNode
    std::string vertexName = func->getParamDecl(0)->getNameAsString();
    if (isTopological) {
      declString << ", ('unsigned int', '__begin'), ('unsigned int', '__end')";
      bodyString << "ForAll(\"" << vertexName << "\", G.nodes(\"__begin\", \"__end\"),\n[\n";
      bodyString << "CDecl([(\"bool\", \"pop\", \" = " << vertexName << " < __end\")]),\n"; // hack for IrGL compiler
      bodyString << "If(\"pop\", [\n"; // hack to handle nested parallelism (should be handled by IrGL compiler in the future)
      conditional_pop = true;
    } else {
      bodyString << "ForAll(\"wlvertex\", WL.items(),\n[\n";
      bodyString << "CDecl([(\"int\", \"" << vertexName << "\", \"\")]),\n";
      bodyString << "CDecl([(\"bool\", \"pop\", \"\")]),\n";
      bodyString << "WL.pop(\"pop\", \"wlvertex\", \"" << vertexName << "\"),\n";
      bodyString << "If(\"pop\", [\n"; // hack to handle unsuccessful pop (should be handled by IrGL compiler in the future)
      conditional_pop = true;
    }
    symbolTable.insert(vertexName);
    return true;
  }

  virtual bool VisitCXXForRangeStmt(CXXForRangeStmt* forStmt) {
    assert(forStmt->getLoopVariable());
    skipDecls.insert(forStmt->getLoopVariable());
    DeclStmt *rangeStmt = forStmt->getRangeStmt();
    for (auto decl : rangeStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
      }
    }
    std::string variableName = forStmt->getLoopVariable()->getNameAsString();
    symbolTable.insert(variableName);
    const Expr *expr = forStmt->getRangeInit();
    std::string vertexName = rewriter.getRewrittenText(expr->getSourceRange());
    std::size_t found = vertexName.find("graph->edges");
    assert(found != std::string::npos);
    std::size_t begin = vertexName.find("(", found);
    std::size_t end = vertexName.find(",", begin);
    if (end == std::string::npos) end = vertexName.find(")", begin);
    vertexName = vertexName.substr(begin+1, end - begin - 1);
    while (conditional > 0) {
      bodyString << "], [ CBlock([\"pop = false\"]), ]),\n"; // end If
      --conditional;
    }
    bodyString << "]),\n"; // end If "pop"
    conditional_pop = false;
    bodyString << "UniformConditional(If(\"!pop\", [CBlock(\"continue\")]), uniform_only = False, _only_if_np = True),\n"; // hack to handle non-nested parallelism
    bodyString << "ClosureHint(\n";
    bodyString << "ForAll(\"" << variableName << "\", G.edges(\"" << vertexName << "\"),\n[\n";
    return true;
  }

  virtual bool VisitForStmt(ForStmt* forStmt) {
    skipStmts.insert(forStmt->getInit());
    skipStmts.insert(forStmt->getCond());
    skipStmts.insert(forStmt->getInc());
    std::string variableName, vertexName;
    DeclStmt *declStmt = dyn_cast<DeclStmt>(forStmt->getInit());
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        std::string text = rewriter.getRewrittenText(varDecl->getSourceRange());
        std::size_t found = text.find("graph->edge_begin");
        if (found != std::string::npos) {
          std::size_t begin = text.find("(", found);
          std::size_t end = text.find(",", begin);
          if (end == std::string::npos) end = text.find(")", begin);
          vertexName = text.substr(begin+1, end - begin - 1);
          variableName = varDecl->getNameAsString();
          break;
        }
      }
    }
    symbolTable.insert(variableName);
    while (conditional > 0) {
      bodyString << "], [ CBlock([\"pop = false\"]), ]),\n"; // end If
      --conditional;
    }
    bodyString << "]),\n"; // end If "pop"
    conditional_pop = false;
    bodyString << "UniformConditional(If(\"!pop\", [CBlock(\"continue\")]), uniform_only = False, _only_if_np = True),\n"; // hack to handle non-nested parallelism
    bodyString << "ClosureHint(\n";
    bodyString << "ForAll(\"" << variableName << "\", G.edges(\"" << vertexName << "\"),\n[\n";
    return true;
  }

  virtual bool VisitDeclStmt(DeclStmt *declStmt) {
    bool skipStmt = skipStmts.find(declStmt) != skipStmts.end();
    for (auto decl : declStmt->decls()) {
      if (VarDecl *varDecl = dyn_cast<VarDecl>(decl)) {
        const Expr *expr = varDecl->getAnyInitializer();
        skipStmts.insert(expr);
        if (!skipStmt && (skipDecls.find(varDecl) == skipDecls.end())) {
          symbolTable.insert(varDecl->getNameAsString());
          std::size_t pos = std::string::npos;
          std::string initText;
          if (expr != NULL) {
            initText = rewriter.getRewrittenText(expr->getSourceRange());
            pos = initText.find("graph->getData");
          }
          if (pos == std::string::npos) {
            std::string decl = varDecl->getType().getAsString();
            if (conditional_pop) {
              varDeclString << "CDecl([(\"" << FormatType(decl)
                << "\", \"" << varDecl->getNameAsString() << "\", \"\")]),\n";
            } else {
              bodyString << "CDecl([(\"" << FormatType(decl)
                << "\", \"" << varDecl->getNameAsString() << "\", \"\")]),\n";
            }

            if (!initText.empty()) {
              WriteCBlock(varDecl->getNameAsString() + " = " + initText);
            }
          }
          else {
            // get the first argument to graph->getData()
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
      WriteCBlock(rewriter.getRewrittenText(binaryOp->getSourceRange()));
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
        std::string var = rewriter.getRewrittenText(argument->getSourceRange());

        WriteCBlock("p_" + reduceVar + "[vertex] = " + var);
        std::string type = argument->getType().getUnqualifiedType().getAsString();
        parameterToTypeMap[reduceVar] = FormatType(type);
      } else {
        WriteCBlock(rewriter.getRewrittenText(callExpr->getSourceRange()));
      }
    }
    for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
      skipStmts.insert(callExpr->getArg(i));
    }
    return true;
  }

  virtual bool VisitIfStmt(IfStmt *ifStmt) {
    const Expr *expr = ifStmt->getCond();
    std::string text = FormatCBlock(rewriter.getRewrittenText(expr->getSourceRange()));
    bodyString << "If(\"" << text << "\",\n[\n";
    ++conditional;
    skipStmts.insert(expr);
    return true;
  }

  virtual bool VisitCastExpr(CastExpr *castExpr) {
    if (skipStmts.find(castExpr) != skipStmts.end()) {
      skipStmts.insert(castExpr->getSubExpr());
    }
    return true;
  }

  virtual bool VisitCXXConstructExpr(CXXConstructExpr *constructExpr) {
    if (skipStmts.find(constructExpr) != skipStmts.end()) {
      for (auto arg : constructExpr->arguments()) {
        skipStmts.insert(arg);
      }
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

  virtual bool VisitConditionalOperator(ConditionalOperator *conditionalOperator) {
    if (skipStmts.find(conditionalOperator) != skipStmts.end()) {
      skipStmts.insert(conditionalOperator->getCond());
      skipStmts.insert(conditionalOperator->getTrueExpr());
      skipStmts.insert(conditionalOperator->getFalseExpr());
    }
    return true;
  }

  virtual bool VisitMemberExpr(MemberExpr *memberExpr) {
    std::string objectName = rewriter.getRewrittenText(memberExpr->getBase()->getSourceRange());
    const std::string &varName = memberExpr->getMemberNameInfo().getAsString();
    std::string type = memberExpr->getMemberDecl()->getType().getUnqualifiedType().getAsString();
    if (nodeMap.find(objectName) != nodeMap.end()) {
      parameterToTypeMap[varName] = FormatType(type);
    }
    else if ((symbolTable.find(varName) == symbolTable.end()) 
        && (varName.compare(accumulatorName) != 0)
        && (type.find("Graph") != 0)) {
      if (globalVariableToTypeMap.find(varName) == globalVariableToTypeMap.end()) {
        assert(varName.find(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__) == 0);
        globalVariableToTypeMap[varName] = FormatType(type);
      }
    }
    return true;
  }
};

class IrGLOrchestratorVisitor : public RecursiveASTVisitor<IrGLOrchestratorVisitor> {
private:
  ASTContext* astContext;
  Rewriter &rewriter; // FIXME: is this necessary? can ASTContext give source code?
  bool requiresWorklist; // shared across kernels
  std::string accumulatorName;
  std::ofstream Output;

  std::map<std::string, std::string> SharedVariablesToTypeMap;
  std::map<std::string, std::pair<std::vector<std::string>, std::vector<std::string> > > KernelToArgumentsMap;
  std::map<std::string, std::vector<std::string> > HostKernelsToArgumentsMap;
  std::set<std::string> KernelsHavingReturnValue;

  std::string FileNamePath;

public:
  explicit IrGLOrchestratorVisitor(ASTContext *context, Rewriter &R) : 
    astContext(context),
    rewriter(R),
    requiresWorklist(false)
  {}
  
  virtual ~IrGLOrchestratorVisitor() {}

  void GenerateIrGLASTToFile() const {
    std::string filename;
    std::size_t found = FileNamePath.rfind("/");
    if (found != std::string::npos) filename = FileNamePath.substr(found+1, FileNamePath.length()-1);
    else filename = FileNamePath;

    std::ofstream header;
    header.open(FileNamePath + "_cuda.h");
    header << "#pragma once\n";
    header << "#include \"Galois/Runtime/Cuda/cuda_mtypes.h\"\n";
    header << "#include \"Galois/Runtime/DataCommMode.h\"\n";
    header << "\nstruct CUDA_Context;\n";
    if (requiresWorklist) {
      header << "\nstruct CUDA_Worklist {\n";
      header << "\tint *in_items;\n";
      header << "\tint num_in_items;\n";
      header << "\tint *out_items;\n";
      header << "\tint num_out_items;\n";
      header << "\tint max_size;\n";
      header << "};\n";
    }
    header << "\nstruct CUDA_Context *get_CUDA_context(int id);\n";
    header << "bool init_CUDA_context(struct CUDA_Context *ctx, int device);\n";
    header << "void load_graph_CUDA(struct CUDA_Context *ctx, ";
    if (requiresWorklist) {
      header << "struct CUDA_Worklist *wl, double wl_dup_factor, ";
    }
    header << "MarshalGraph &g, unsigned num_hosts);\n\n";
    header << "void reset_CUDA_context(struct CUDA_Context *ctx);\n\n";
    for (auto& var : SharedVariablesToTypeMap) {
      header << "void bitset_" << var.first << "_clear_cuda(struct CUDA_Context *ctx);\n";
      header << var.second << " get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID);\n";
      header << "void set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v);\n";
      header << "void add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v);\n";
      header << "void min_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v);\n";
      header << "void batch_get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v);\n";
      header << "void batch_get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode);\n";
      header << "void batch_get_slave_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v);\n";
      header << "void batch_get_slave_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode);\n";
      header << "void batch_get_reset_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v, " << var.second << " i);\n";
      header << "void batch_get_reset_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode, " << var.second << " i);\n";
      header << "void batch_set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode);\n";
      header << "void batch_add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode);\n";
      header << "void batch_min_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode);\n\n";
    }
    for (auto& kernel : HostKernelsToArgumentsMap) {
      header << "void " << kernel.first << "_cuda(";
      for (auto& argument : kernel.second) {
        header << argument << ", ";
      }
      header << "struct CUDA_Context *ctx);\n";
    }
    header.close();

    std::ofstream cuheader;
    cuheader.open(FileNamePath + "_cuda.cuh");
    cuheader << "#pragma once\n";
    cuheader << "#include <cuda.h>\n";
    cuheader << "#include <stdio.h>\n";
    cuheader << "#include <sys/types.h>\n";
    cuheader << "#include <unistd.h>\n";
    cuheader << "#include \"" << filename << "_cuda.h\"\n";
    cuheader << "#include \"Galois/Runtime/Cuda/cuda_helpers.h\"\n\n";
    cuheader << "struct CUDA_Context : public CUDA_Context_Common {\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tstruct CUDA_Context_Field<" << var.second << "> " << var.first << ";\n";
    }
    if (requiresWorklist) {
      cuheader << "\tWorklist2 in_wl;\n";
      cuheader << "\tWorklist2 out_wl;\n";
      cuheader << "\tstruct CUDA_Worklist *shared_wl;\n";
    }
    cuheader << "};\n\n";
    cuheader << "struct CUDA_Context *get_CUDA_context(int id) {\n";
    cuheader << "\tstruct CUDA_Context *ctx;\n";
    cuheader << "\tctx = (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));\n";
    cuheader << "\tctx->id = id;\n";
    cuheader << "\treturn ctx;\n";
    cuheader << "}\n\n";
    cuheader << "bool init_CUDA_context(struct CUDA_Context *ctx, int device) {\n";
    cuheader << "\treturn init_CUDA_context_common(ctx, device);\n";
    cuheader << "}\n\n";
    cuheader << "void load_graph_CUDA(struct CUDA_Context *ctx, ";
    if (requiresWorklist) {
      cuheader << "struct CUDA_Worklist *wl, double wl_dup_factor, ";
    }
    cuheader << "MarshalGraph &g, unsigned num_hosts) {\n";
    cuheader << "\tsize_t mem_usage = mem_usage_CUDA_common(g, num_hosts);\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tmem_usage += mem_usage_CUDA_field(&ctx->" << var.first << ", g, num_hosts);\n";
    }
    cuheader << "\tprintf(\"[%d] Host memory for communication context: %3u MB\\n\", ctx->id, mem_usage/1048756);\n";
    cuheader << "\tload_graph_CUDA_common(ctx, g, num_hosts);\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tload_graph_CUDA_field(ctx, &ctx->" << var.first << ", num_hosts);\n";
    }
    if (requiresWorklist) {
      cuheader << "\twl->max_size = wl_dup_factor*g.nnodes;\n";
      cuheader << "\tctx->in_wl = Worklist2((size_t)wl->max_size);\n";
      cuheader << "\tctx->out_wl = Worklist2((size_t)wl->max_size);\n";
      cuheader << "\twl->num_in_items = -1;\n";
      cuheader << "\twl->num_out_items = -1;\n";
      cuheader << "\twl->in_items = ctx->in_wl.wl;\n";
      cuheader << "\twl->out_items = ctx->out_wl.wl;\n";
      cuheader << "\tctx->shared_wl = wl;\n";
      cuheader << "\tprintf(\"[%d] load_graph_GPU: worklist size %d\\n\", ctx->id, (size_t)wl_dup_factor*g.nnodes);\n"; 
    }
    cuheader << "\treset_CUDA_context(ctx);\n";
    cuheader << "}\n\n";
    cuheader << "void reset_CUDA_context(struct CUDA_Context *ctx) {\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tctx->" << var.first << ".data.zero_gpu();\n"; // FIXME: should do this only for std::atomic variables?
    }
    cuheader << "}\n\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "void bitset_" << var.first << "_clear_cuda(struct CUDA_Context *ctx) {\n";
      cuheader << "\tctx->" << var.first << ".is_updated.cpu_rd_ptr()->clear();\n";
      cuheader << "}\n\n";
      cuheader << var.second << " get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".data.cpu_rd_ptr();\n";
      cuheader << "\treturn " << var.first << "[LID];\n";
      cuheader << "}\n\n";
      cuheader << "void set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".data.cpu_wr_ptr();\n";
      cuheader << "\t" << var.first << "[LID] = v;\n";
      cuheader << "}\n\n";
      cuheader << "void add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".data.cpu_wr_ptr();\n";
      cuheader << "\t" << var.first << "[LID] += v;\n";
      cuheader << "}\n\n";
      cuheader << "void min_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".data.cpu_wr_ptr();\n";
      cuheader << "\tif (" << var.first << "[LID] > v)\n";
      cuheader << "\t\t" << var.first << "[LID] = v;\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedMaster, false>(ctx, &ctx->" << var.first << ", from_id, v);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedMaster, false>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_slave_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedSlave, false>(ctx, &ctx->" << var.first << ", from_id, v);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_slave_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedSlave, false>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_reset_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, " << var.second << " *v, " << var.second << " i) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedSlave, true>(ctx, &ctx->" << var.first << ", from_id, v, i);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_get_reset_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t *v_size, DataCommMode *data_mode, " << var.second << " i) {\n";
      cuheader << "\tbatch_get_shared_field<" << var.second << ", sharedSlave, true>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode, i);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode) {\n";
      cuheader << "\tbatch_set_shared_field<" << var.second << ", sharedSlave, setOp>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode) {\n";
      cuheader << "\tbatch_set_shared_field<" << var.second << ", sharedMaster, addOp>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode);\n";
      cuheader << "}\n\n";
      cuheader << "void batch_min_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned from_id, unsigned long long int *bitset_comm, unsigned int *offsets, " << var.second << " *v, size_t v_size, DataCommMode data_mode) {\n";
      cuheader << "\tbatch_set_shared_field<" << var.second << ", sharedMaster, minOp>(ctx, &ctx->" << var.first << ", from_id, bitset_comm, offsets, v, v_size, data_mode);\n";
      cuheader << "}\n\n";
    }
    cuheader.close();

    std::ofstream IrGLAST;
    IrGLAST.open(FileNamePath + "_cuda.py");
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
    IrGLAST << "CBlock([cgen.Include(\"" << filename << "_cuda.cuh\", "
      << "system = False)], parse = False),\n";
    for (auto& var : SharedVariablesToTypeMap) {
      std::string upper = var.first;
      std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
      IrGLAST << "CDeclGlobal([(\"" << var.second 
        << " *\", \"P_" << upper << "\", \"\")]),\n";
    }

    std::ifstream compute(COMPUTE_FILENAME);
    IrGLAST << compute.rdbuf(); 
    compute.close();
    remove(COMPUTE_FILENAME);

    std::ifstream orchestrator(ORCHESTRATOR_FILENAME);
    IrGLAST << orchestrator.rdbuf();
    orchestrator.close();
    remove(ORCHESTRATOR_FILENAME);

    IrGLAST << "])\n"; // end Module
    IrGLAST.close();

    llvm::errs() << "IrGL file and headers generated:\n";
    llvm::errs() << FileNamePath << "_cuda.py\n";
    llvm::errs() << FileNamePath << "_cuda.cuh\n";
    llvm::errs() << FileNamePath << "_cuda.h\n";
  }

  std::string FormatType(std::string text) {
    findAndReplace(text, "GNode", "index_type");
    findAndReplace(text, "_Bool", "bool");

    // get basic type
    if (text.find("std::atomic") == 0) {
      std::size_t begin = text.find("<");
      std::size_t end = text.find_last_of(">");
      text = text.substr(begin+1, end - begin - 1);
    }
    if (text.find("cll::opt") == 0) {
      std::size_t begin = text.find("<");
      std::size_t end = text.find_last_of(">");
      text = text.substr(begin+1, end - begin - 1);
    }

    return text;
  }

  virtual bool TraverseDecl(Decl *D) {
    bool traverse = RecursiveASTVisitor<IrGLOrchestratorVisitor>::TraverseDecl(D);
    return traverse;
  }

  virtual bool VisitCallExpr(CallExpr *callExpr) {
    std::string text = rewriter.getRewrittenText(callExpr->getSourceRange());
    bool isTopological = true;
    std::size_t begin = text.find("galois::do_all");
    if (begin != 0) {
      begin = text.find("galois::for_each");
      isTopological = false;
    }
    if (begin == 0) {
      // generate kernel for operator
      assert(callExpr->getNumArgs() > 2);
      auto arg_type = callExpr->getArg(2)->getType();
      assert(arg_type->isStructureType());
      RecordDecl *record = arg_type->getAsStructureType()->getDecl();
      CXXRecordDecl *cxxrecord = dyn_cast<CXXRecordDecl>(record);
      // find if there are any accumulators being used
      for (auto decl : cxxrecord->decls()) {
        auto var = dyn_cast<VarDecl>(decl);
        if (var && var->isStaticDataMember()) {
          std::string type = var->getType().getAsString();
          if (type.find(__GALOIS_ACCUMULATOR_TYPE__) == 0) {
            accumulatorName = var->getNameAsString();
            break;
          }
        }
      }
      for (auto method : cxxrecord->methods()) {
        // FIXME: handle overloading/polymorphism
        if (method->getNameAsString().compare("operator()") == 0) {
          // FIXME: should the arguments to the call be parsed?
          //llvm::errs() << "\nVertex Operator:\n" << 
          //  rewriter.getRewrittenText(method->getSourceRange()) << "\n";
          //method->dump();
          IrGLOperatorVisitor operatorVisitor(astContext, rewriter,
              isTopological,
              accumulatorName,
              SharedVariablesToTypeMap,
              KernelToArgumentsMap,
              KernelsHavingReturnValue);
          operatorVisitor.TraverseDecl(method);
          break;
        }
      }

      std::string kernelName = cxxrecord->getNameAsString();

      // kernel signature
      Output.open(ORCHESTRATOR_FILENAME, std::ios::app);
      std::stringstream Output_topo_all, Output_topo_all_body;
      Output << "Kernel(\"" << kernelName << "_cuda\", [";
      if (isTopological) {
        Output_topo_all << "Kernel(\"" << kernelName << "_all_cuda\", [";
        Output_topo_all_body << "CBlock([\"" << kernelName << "_cuda(";
        Output_topo_all_body << "0, ctx->nowned, ";
      }
      std::vector<std::string> arguments, arguments_topo_all;
      if (isTopological) {
        Output << "('unsigned int ', '__begin'), ('unsigned int ', '__end'), ";
        arguments.push_back("unsigned int __begin");
        arguments.push_back("unsigned int __end");
      }
      if (!accumulatorName.empty()) {
        Output << "('int &', '__retval'), ";
        arguments.push_back("int & __retval");
        if (isTopological) {
          Output_topo_all << "('int &', '__retval'), ";
          Output_topo_all_body << "__retval, ";
          arguments_topo_all.push_back("int & __retval");
        }
      }
      for (auto field : cxxrecord->fields()) {
        std::string name = field->getNameAsString();
        if (name.find(__GALOIS_PREPROCESS_GLOBAL_VARIABLE_PREFIX__) == 0) {
          std::string type = FormatType(field->getType().getAsString());
          Output << "('" << type << "', '" << name << "'), ";
          arguments.push_back(type + " " + name);
          if (isTopological) {
            Output_topo_all << "('" << type << "', '" << name << "'), ";
            Output_topo_all_body << name << ", ";
            arguments_topo_all.push_back(type + " " + name);
          }
        } 
      }
      HostKernelsToArgumentsMap[kernelName] = arguments;
      Output << "('struct CUDA_Context *', 'ctx')],\n[\n";
      if (isTopological) {
        HostKernelsToArgumentsMap[kernelName + "_all"] = arguments_topo_all;
        Output_topo_all << "('struct CUDA_Context *', 'ctx')],\n[\n";
        Output_topo_all_body << "ctx)\"]),\n";
      }

      // initialize blocks and threads
      Output << "CDecl([(\"dim3\", \"blocks\", \"\")]),\n";
      Output << "CDecl([(\"dim3\", \"threads\", \"\")]),\n";
      Output << "CBlock([\"kernel_sizing(blocks, threads)\"]),\n";

      // generate call to kernel
      bool hasAReturnValue = (KernelsHavingReturnValue.find(kernelName) != KernelsHavingReturnValue.end());
      if (!isTopological) {
        requiresWorklist = true;
        Output << "CBlock([\"ctx->in_wl.update_gpu(ctx->shared_wl->num_in_items)\"]),\n";
        Output << "CBlock([\"ctx->out_wl.will_write()\"]),\n";
        Output << "CBlock([\"ctx->out_wl.reset()\"]),\n";
      }
      Output << "Invoke(\"" << kernelName << "\", ";
      Output << "(\"ctx->gg\", \"ctx->nowned\"";
      if (isTopological) {
        Output << ", \"__begin\", \"__end\"";
      }
      auto& karguments = KernelToArgumentsMap[kernelName];
      for (auto& argument : karguments.first) {
        Output << ", \"" << argument << "\"";
      }
      for (auto& argument : karguments.second) {
        Output << ", \"ctx->" << argument << ".data.gpu_wr_ptr()\"";
      }
      if (!isTopological) {
        Output << ", \"ctx->in_wl\"";
        Output << ", \"ctx->out_wl\"";
      }
      Output << ")";
      if (hasAReturnValue) {
        Output << ", \"SUM\"";
      }
      Output << "),\n";
      Output << "CBlock([\"check_cuda_kernel\"], parse = False),\n";
      if (hasAReturnValue) {
        Output << "CBlock([\"__retval = *(retval.cpu_rd_ptr())\"], parse = False),\n";
      }
      if (!isTopological) {
        Output << "CBlock([\"ctx->out_wl.update_cpu()\"]),\n";
        Output << "CBlock([\"ctx->shared_wl->num_out_items = ctx->out_wl.nitems()\"]),\n";
      }

      Output << "], host = True),\n"; // end Kernel

      if (isTopological) {
        Output << Output_topo_all.str();
        Output << Output_topo_all_body.str();
        Output << "], host = True),\n"; // end Kernel
      }

      Output.close();
      accumulatorName.clear();

      std::string fileName = rewriter.getSourceMgr().getFilename(cxxrecord->getLocStart()).str();
      std::size_t found = fileName.rfind(".");
      assert(found != std::string::npos);
      FileNamePath = fileName.substr(0, found);
    }
    return true;
  }
};
  
class FunctionDeclHandler : public MatchFinder::MatchCallback {
private:
  //ASTContext *astContext;
  //Rewriter &rewriter;
  IrGLOrchestratorVisitor orchestratorVisitor;
public:
  FunctionDeclHandler(ASTContext *context, Rewriter &R): 
    //astContext(context),
    //rewriter(R),
    orchestratorVisitor(context, R)
  {}
  const IrGLOrchestratorVisitor& getOrchestrator() {
    return orchestratorVisitor;
  }
  virtual void run(const MatchFinder::MatchResult &Results) {
    const CXXMethodDecl* decl = Results.Nodes.getNodeAs<clang::CXXMethodDecl>("graphAlgorithm");
    if (decl) {
      //llvm::errs() << "\nGraph Operator:\n" << 
      //  rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
      //decl->dump();
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
          hasName("go"),
          //hasOverloadedOperatorName("()"),
          hasAnyParameter(hasName("_graph"))).
          bind("graphAlgorithm"), &functionDeclHandler);
  }

  virtual void HandleTranslationUnit(ASTContext &Context){
    Matchers.matchAST(Context);
    functionDeclHandler.getOrchestrator().GenerateIrGLASTToFile();
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
