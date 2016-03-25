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

  std::map<std::string, std::pair<std::string, std::string> > &GlobalVariablesToTypeMap;
  std::map<std::string, std::string> &SharedVariablesToTypeMap;
  std::map<std::string, std::vector<std::string> > &KernelToArgumentsMap;

public:
  explicit IrGLOperatorVisitor(ASTContext *context, Rewriter &R, 
      bool isTopo,
      std::map<std::string, std::pair<std::string, std::string> > &globalVariables,
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

    while (1) {
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
    } 

    // use & (address-of) operator for atomic variables in atomicAdd() etc
    std::size_t start = 0;
    while (1) {
      std::size_t found = text.find("Galois::atomic", start);
      if (found != std::string::npos) {
        std::size_t replace = text.find("(", start);
        text.insert(replace+1, "&");
        start = replace;
      } else {
        break;
      }
    }
    findAndReplace(text, "Galois::atomic", "atomic");

    findAndReplace(text, "->getEdgeDst", ".getAbsDestination");
    findAndReplace(text, "->getEdgeData", ".getAbsWeight");
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
      bodyString << "]),\n"; // end Kernel
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
      if (isa<CXXForRangeStmt>(S) || isa<ForStmt>(S)) {
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
    declString << "Kernel(\"" << funcName << "\", [G.param(), ('int ', 'nowned') ";
    // FIXME: assuming first parameter is GNode
    std::string vertexName = func->getParamDecl(0)->getNameAsString();
    if (isTopological) {
      bodyString << "ForAll(\"" << vertexName << "\", G.nodes(None, \"nowned\"),\n[\n";
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
    SourceRange range(expr->getLocStart(), expr->getLocEnd());
    std::string vertexName = rewriter.getRewrittenText(range);
    std::size_t found = vertexName.find("graph->edges");
    assert(found != std::string::npos);
    std::size_t begin = vertexName.find("(", found);
    std::size_t end = vertexName.find(",", begin);
    if (end == std::string::npos) end = vertexName.find(")", begin);
    vertexName = vertexName.substr(begin+1, end - begin - 1);
    bodyString << "ForAll(\"" << variableName << "\", G.edges(\"" << vertexName << "\"),\n[\n";
    return true;
  }

  virtual bool VisitForStmt(ForStmt* forStmt) {
    skipStmts.insert(forStmt->getInit());
    skipStmts.insert(forStmt->getCond());
    skipStmts.insert(forStmt->getInc());
    std::string variableName;
    if (forStmt->getConditionVariable()) {
      skipDecls.insert(forStmt->getConditionVariable());
      variableName = forStmt->getConditionVariable()->getNameAsString();
    } else {
      variableName = dyn_cast<VarDecl>(dyn_cast<DeclStmt>(forStmt->getInit())->getSingleDecl())->getNameAsString();
    }
    symbolTable.insert(variableName);
    const Expr *expr = forStmt->getCond();
    SourceRange range(expr->getLocStart(), expr->getLocEnd());
    std::string vertexName = rewriter.getRewrittenText(range);
    std::size_t found = vertexName.find("graph->edge_end");
    assert(found != std::string::npos);
    std::size_t begin = vertexName.find("(", found);
    std::size_t end = vertexName.find(",", begin);
    if (end == std::string::npos) end = vertexName.find(")", begin);
    vertexName = vertexName.substr(begin+1, end - begin - 1);
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
            SourceRange range(expr->getLocStart(), expr->getLocEnd());
            initText = rewriter.getRewrittenText(range);
            pos = initText.find("graph->getData");
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
        std::string type = decl->getType().getAsString();
        std::string declStr = rewriter.getRewrittenText(decl->getSourceRange());
        std::size_t found = declStr.find("=");
        std::string value;
        if (found != std::string::npos) {
          value = declStr.substr(found, declStr.length()-found);
        } else {
          value = "";
        }
        GlobalVariablesToTypeMap[varName] = std::make_pair(type, value);
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

  std::map<std::string, std::pair<std::string, std::string> > GlobalVariablesToTypeMap;
  std::map<std::string, std::string> SharedVariablesToTypeMap;
  std::map<std::string, std::vector<std::string> > KernelToArgumentsMap;
  std::vector<std::string> Kernels;

  std::string FileNamePath;

public:
  explicit IrGLOrchestratorVisitor(ASTContext *context, Rewriter &R) : 
    astContext(context),
    rewriter(R)
  {}
  
  virtual ~IrGLOrchestratorVisitor() {}

  void GenerateIrGLASTToFile() const {
    std::string filename;
    std::size_t found = FileNamePath.rfind("/");
    if (found != std::string::npos) filename = FileNamePath.substr(found+1, FileNamePath.length()-1);

    std::ofstream header;
    header.open(FileNamePath + "_cuda.h");
    header << "#pragma once\n";
    header << "#include \"Galois/Cuda/cuda_mtypes.h\"\n";
    header << "\nstruct CUDA_Context;\n";
    header << "\nstruct CUDA_Context *get_CUDA_context(int id);\n";
    header << "bool init_CUDA_context(struct CUDA_Context *ctx, int device);\n";
    header << "void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g);\n\n";
    for (auto& var : SharedVariablesToTypeMap) {
      header << var.second << " get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID);\n";
      header << "void set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v);\n";
      header << "void add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v);\n";
    }
    for (auto& kernelName : Kernels) {
      header << "void " << kernelName << "_cuda(struct CUDA_Context *ctx);\n";
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
    cuheader << "\n#ifdef __GALOIS_CUDA_CHECK_ERROR__\n";
    cuheader << "#define check_cuda_kernel check_cuda(cudaGetLastError()); check_cuda(cudaDeviceSynchronize());\n";
    cuheader << "#else\n";
    cuheader << "#define check_cuda_kernel  \n";
    cuheader << "#endif\n";
    cuheader << "\nstruct CUDA_Context {\n";
    cuheader << "\tint device;\n";
    cuheader << "\tint id;\n";
    cuheader << "\tsize_t nowned;\n";
    cuheader << "\tsize_t g_offset;\n";
    cuheader << "\tCSRGraphTex hg;\n";
    cuheader << "\tCSRGraphTex gg;\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tShared<" << var.second << "> " << var.first << ";\n";
    }
    cuheader << "};\n\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << var.second << " get_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".cpu_rd_ptr();\n";
      cuheader << "\treturn " << var.first << "[LID];\n";
      cuheader << "}\n\n";
      cuheader << "void set_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".cpu_wr_ptr();\n";
      cuheader << "\t" << var.first << "[LID] = v;\n";
      cuheader << "}\n\n";
      cuheader << "void add_node_" << var.first << "_cuda(struct CUDA_Context *ctx, unsigned LID, " << var.second << " v) {\n";
      cuheader << "\t" << var.second << " *" << var.first << " = ctx->" << var.first << ".cpu_wr_ptr();\n";
      cuheader << "\t" << var.first << "[LID] += v;\n";
      cuheader << "}\n\n";
    }
    cuheader << "struct CUDA_Context *get_CUDA_context(int id) {\n";
    cuheader << "\tstruct CUDA_Context *ctx;\n";
    cuheader << "\tctx = (struct CUDA_Context *) calloc(1, sizeof(struct CUDA_Context));\n";
    cuheader << "\treturn ctx;\n";
    cuheader << "}\n\n";
    cuheader << "bool init_CUDA_context(struct CUDA_Context *ctx, int device) {\n";
    cuheader << "\tstruct cudaDeviceProp dev;\n";
    cuheader << "\tif(device == -1) {\n";
    cuheader << "\t\tcheck_cuda(cudaGetDevice(&device));\n";
    cuheader << "\t} else {\n";
    cuheader << "\t\tint count;\n";
    cuheader << "\t\tcheck_cuda(cudaGetDeviceCount(&count));\n";
    cuheader << "\t\tif(device > count) {\n";
    cuheader << "\t\t\tfprintf(stderr, \"Error: Out-of-range GPU %d specified (%d total GPUs)\", device, count);\n";
    cuheader << "\t\t\treturn false;\n";
    cuheader << "\t\t}\n";
    cuheader << "\t\tcheck_cuda(cudaSetDevice(device));\n";
    cuheader << "\t}\n";
    cuheader << "\tctx->device = device;\n";
    cuheader << "\tcheck_cuda(cudaGetDeviceProperties(&dev, device));\n";
    cuheader << "\tfprintf(stderr, \"%d: Using GPU %d: %s\\n\", ctx->id, device, dev.name);\n";
    cuheader << "\treturn true;\n";
    cuheader << "}\n\n";
    cuheader << "void load_graph_CUDA(struct CUDA_Context *ctx, MarshalGraph &g) {\n";
    cuheader << "\tCSRGraphTex &graph = ctx->hg;\n";
    cuheader << "\tctx->nowned = g.nowned;\n";
    cuheader << "\tctx->id = g.id;\n";
    cuheader << "\tgraph.nnodes = g.nnodes;\n";
    cuheader << "\tgraph.nedges = g.nedges;\n";
    cuheader << "\tif(!graph.allocOnHost()) {\n";
    cuheader << "\t\tfprintf(stderr, \"Unable to alloc space for graph!\");\n";
    cuheader << "\t\texit(1);\n";
    cuheader << "\t}\n";
    cuheader << "\tmemcpy(graph.row_start, g.row_start, sizeof(index_type) * (g.nnodes + 1));\n";
    cuheader << "\tmemcpy(graph.edge_dst, g.edge_dst, sizeof(index_type) * g.nedges);\n";
    cuheader << "\tif(g.node_data) memcpy(graph.node_data, g.node_data, sizeof(node_data_type) * g.nnodes);\n";
    cuheader << "\tif(g.edge_data) memcpy(graph.edge_data, g.edge_data, sizeof(edge_data_type) * g.nedges);\n";
    cuheader << "\tgraph.copy_to_gpu(ctx->gg);\n";
    for (auto& var : SharedVariablesToTypeMap) {
      cuheader << "\tctx->" << var.first << ".alloc(graph.nnodes);\n";
    }
    cuheader << "\tprintf(\"load_graph_GPU: %d owned nodes of total %d resident, %d edges\\n\", ctx->nowned, graph.nnodes, graph.nedges);\n"; 
    cuheader << "}\n\n";
    cuheader << "void kernel_sizing(CSRGraphTex & g, dim3 &blocks, dim3 &threads) {\n";
    cuheader << "\tthreads.x = 256;\n";
    cuheader << "\tthreads.y = threads.z = 1;\n";
    cuheader << "\tblocks.x = 14 * 8;\n";
    cuheader << "\tblocks.y = blocks.z = 1;\n";
    cuheader << "}\n\n";
    cuheader.close();

    std::ofstream IrGLAST;
    IrGLAST.open(FileNamePath + ".py");
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
    for (auto& var : GlobalVariablesToTypeMap) {
      size_t found = var.second.first.find("Galois::");
      if (found == std::string::npos) {
        found = var.second.first.find("cll::opt");
        std::string type;
        if (found == std::string::npos) {
          type = var.second.first;
        } else {
          size_t start = var.second.first.find("<", found);
          // FIXME: nested < < > >
          size_t end = var.second.first.find(">", found);
          type = var.second.first.substr(start+1, end-start-1);
        }
        IrGLAST << "CDeclGlobal([(\"extern " << type 
          << "\", \"" << var.first << "\", \"" << var.second.second << "\")]),\n";
      }
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
    llvm::errs() << FileNamePath << ".py\n";
    llvm::errs() << FileNamePath << "_cuda.cuh\n";
    llvm::errs() << FileNamePath << "_cuda.h\n";
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
        Output << "CBlock([\"" << text << "\"], parse = False),\n";
      } else {
        Output << "CBlock([\"" << text << "\"]),\n";
      }
    }
  }

  virtual bool TraverseDecl(Decl *D) {
    CXXMethodDecl *method = dyn_cast_or_null<CXXMethodDecl>(D);
    if (method) {
      Output.open(ORCHESTRATOR_FILENAME, std::ios::app);
      std::string kernelName = method->getParent()->getNameAsString();
      Kernels.push_back(kernelName);
      Output << "Kernel(\"" << kernelName << "_cuda\", ";
      Output << "[('struct CUDA_Context *', 'ctx')],\n[\n";
    }
    bool traverse = RecursiveASTVisitor<IrGLOrchestratorVisitor>::TraverseDecl(D);
    if (traverse && method) {
      Output << "], host = True),\n"; // end Kernel
      Output.close();

      std::string fileName = rewriter.getSourceMgr().getFilename(D->getLocStart()).str();
      std::size_t found = fileName.rfind(".");
      assert(found != std::string::npos);
      FileNamePath = fileName.substr(0, found);
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
          reduceCall << "mgpu::Reduce(p_" << reduceVar << ".gpu_rd_ptr(ctx->device+1), ctx->hg.nnodes";
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
        // FIXME: pick the right argument
        Expr *op = callExpr->getArg(2);
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
          Output << "Invoke(\"__init_worklist__\", (\"ctx->gg\", )),\n";
          Output << "Pipe([\n";
        }

        // generate call to kernel
        Output << "CDecl([(\"dim3\", \"blocks\", \"\")]),\n";
        Output << "CDecl([(\"dim3\", \"threads\", \"\")]),\n";
        Output << "CBlock([\"kernel_sizing(ctx->gg, blocks, threads)\"]),\n";
        std::string kernelName = record->getNameAsString();
        Output << "Invoke(\"" << kernelName << "\", ";
        Output << "(\"ctx->gg\", \"ctx->nowned\"";
        auto& arguments = KernelToArgumentsMap[kernelName];
        for (auto& argument : arguments) {
          Output << ", \"ctx->" << argument << ".gpu_wr_ptr()\"";
        }
        Output << ")),\n";
        Output << "CBlock([\"check_cuda_kernel\"], parse = False),\n";

        if (!isTopological) { // data driven
          Output << "], ),\n";
          Output << "], once=True, wlinit=WLInit(\"ctx->hg.nedges\", [])),\n";
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
        std::string type = decl->getType().getAsString();
        std::string declStr = rewriter.getRewrittenText(decl->getSourceRange());
        std::size_t found = declStr.find("=");
        std::string value;
        if (found != std::string::npos) {
          value = declStr.substr(found, declStr.length()-found);
        } else {
          value = "";
        }
        GlobalVariablesToTypeMap[varName] = std::make_pair(type, value);
      }
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
