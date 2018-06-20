/*
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of the 3-Clause BSD License (a
 * copy is located in LICENSE.txt at the top-level directory).
 *
 * Copyright (C) 2018, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 */

/*
 * kernel_gen.h
 *
 *  Created on: Mar 9, 2016
 *      Author: rashid
 */

#include <sstream>
#include <climits>
#include <vector>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "fstream"

/*
 *  * Matchers
 *   */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"
#include "utils.h"
#include "gAST.h"

#ifndef SRC_PLUGINS_CLIFY_KERNEL_GEN_H_
#define SRC_PLUGINS_CLIFY_KERNEL_GEN_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

#define MARKER_CODE(X)                                                         \
  {}
#define _MARKER_CODE(X)                                                        \
  { X }

struct CLVar {
  bool isReference;
  bool isPointer;
  bool isConst;
  bool isPrimitive;
  const bool isParameter;
  string typeName;
  string varName;
  Decl& parent;
  CLVar(bool isParam, FieldDecl& pvd) : isParameter(isParam), parent(pvd) {
    isPrimitive = false; // to be set by type_convert.
    isReference = pvd.getType().getTypePtr()->isReferenceType();
    isPointer   = pvd.getType().getTypePtr()->isPointerType();
    //      isConst = pvd.getType().getCanonicalType().isConstQualified();
    {
      string tname = pvd.getType().getAsString();
      // isConst was not reliable from the type system. Search for "const"
      // string instead.
      isConst = tname.find("const") != string::npos;
    }
    typeName = pvd.getType().getUnqualifiedType().getAsString();
    if (isConst) {
      if (typeName.find("const ") != string::npos)
        typeName.replace(typeName.find("const "), 5, "");
    }
    if (isPointer) {
      if (typeName.find("*") != string::npos)
        typeName = typeName.substr(0, typeName.find("*") - 1);
    }
    if (isReference) {
      if (typeName.find("&") != string::npos)
        typeName = typeName.substr(0, typeName.find("&") - 1);
    }
    varName  = pvd.getNameAsString();
    typeName = type_convert(typeName);
    llvm::outs() << "Created CLVar (Field) " << typeName << ",var=" << varName
                 << ",isPrimitive=" << isPrimitive << ",isRef=" << isReference
                 << ",isPtr=" << isPointer << ",isConst=" << isConst << "\n";
  }
  CLVar(bool isParam, VarDecl& pvd) : isParameter(isParam), parent(pvd) {
    isPrimitive = false; // to be set by type_convert.
    isReference = pvd.getType().getTypePtr()->isReferenceType();
    isPointer   = pvd.getType().getTypePtr()->isPointerType();
    //      isConst = pvd.getType().getCanonicalType().isConstQualified();
    {
      string tname = pvd.getType().getAsString();
      isConst      = tname.find("const") != string::npos;
    }

    typeName = pvd.getType().getUnqualifiedType().getAsString();
    if (isConst) {
      if (typeName.find("const ") != string::npos)
        typeName.replace(typeName.find("const "), 5, "");
    }
    if (isPointer) {
      if (typeName.find("*") != string::npos)
        typeName = typeName.substr(0, typeName.find("*") - 1);
    }
    if (isReference) {
      if (typeName.find("&") != string::npos)
        typeName = typeName.substr(0, typeName.find("&") - 1);
    }
    varName  = pvd.getNameAsString();
    typeName = type_convert(typeName);
    llvm::outs() << "Created CLVar (Var) " << typeName << ",var=" << varName
                 << ",isPrimitive=" << isPrimitive << ",isRef=" << isReference
                 << ",isPtr=" << isPointer << ",isConst=" << isConst << "\n";
  }
  string deref() {
    string res = "";
    //      if(isPrimitive==false){
    //         return "*" + varName;
    //      }
    if ((isReference || isPointer) && isParameter) {
      if (isPrimitive) {
        res = "*" + varName;
      } else {
        res = varName + "->";
      }
    } else {
      if (isParameter) {
        res = varName;
      } else if (isPrimitive) {
        res = varName;
      } else {
        res = varName + " -> ";
      }
    }
    //      return "$"+res+"$";
    return res;
  }
  string localDecl() {
    string res = "";
    res += getType();
    //      res += " /*VLD*/" + varName + " ";
    res += varName + " ";
    return res;
  }
  string getType() {
    string res = "";
    if (isPrimitive == false) {
      res = "__global " + typeName + " * ";
    } else {
      if (isReference || isPointer)
        res = "__global " + typeName + " * ";
      else
        res = typeName + " ";
    }
    /*if ((isReference || isPointer) && isParameter) {
     res = "__global " + typeName + " * ";
     } else {
     if (isParameter) {
     res = typeName + " ";
     } else if (isPrimitive) {
     res = typeName + " ";
     } else {
     res = " __global " + typeName + " * ";
     }
     }*/
    return res; //+ "/*TYPE*/";
  }

private:
  string type_convert(string tname) {
    // Atomic stripper
    if (tname.find("std::atomic") != string::npos) {
      isPrimitive = true; // Assume atomics over primitives only
      return tname.substr(tname.rfind("<") + 1,
                          tname.rfind(">") - tname.rfind("<") - 1);
    }
    // vector stripper
    if (tname.find("std::vector") != string::npos) {
      std::string qtStr = tname;
      auto start_index  = tname.find("<");
      auto end_index    = qtStr.find(",");
      std::string ret;
      if (end_index != string::npos) {
        llvm::outs() << " [ COMMA ]";
        ret = qtStr.substr(start_index + 1, end_index - start_index - 1);
      } else {
        end_index = qtStr.find(">");
        llvm::outs() << " [ ANGLED-B ]";
        ret = qtStr.substr(start_index + 1, end_index - start_index - 1);
      }
      ret += " * ";
      llvm::outs() << "Conversion " << qtStr << "  to " << ret << "\n";
      isPrimitive = true; // TODO RK - We can assume arrays of primitives only
      return ret;
    }

    if (tname.find("DGAccumulator<int>") != string::npos) {
      isPrimitive = true;
      isPointer   = true;
      return "int";
    }
    if (tname.find("unsigned long long") != string::npos) {
      isPrimitive = true;
      isPointer   = false;
      return "unsigned long long";
    }

    if (tname.find("int") != string::npos ||
        tname.find("float") != string::npos ||
        tname.find("double") != string::npos ||
        tname.find("char") != string::npos ||
        tname.find("uint32_t") != string::npos ||
        tname.find("bool") != string::npos) {
      isPrimitive = true;
      return tname;
    } else if (tname == "_Bool") {
      isPrimitive = true;
      return "bool";
    } else if (tname == "unsigned int") {
      isPrimitive = true;
      return "uint";
    }

    // Iterator stripper
    if (tname.find("boost::iterators::counting_iterator") != string::npos) {
      isPrimitive = true; // iterator is just an int
      return "edge_iterator ";
    }
    if (tname.find("GNode") != string::npos) {
      isPrimitive = true; // iterator is just an int
      return "node_iterator";
    }
    // Wrapper for nodedata.
    if (tname.find("NodeData") != string::npos) {
      isPrimitive = false;
      return "NodeData";
    }
    if (tname.find("EdgeData") != string::npos) {
      isPrimitive = false;
      return "EdgeData";
    }
    if (tname.find("Graph") != string::npos) {
      isPrimitive = false;
      return "Graph";
    }

    return tname + "/*UNKNOWN*/";
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceRewriter : public RecursiveASTVisitor<OpenCLDeviceRewriter> {
  ASTContext& astContext;
  Rewriter& rewriter;
  std::map<Decl*, CLVar*> varMapping;

public:
  explicit OpenCLDeviceRewriter(ASTContext& context, Rewriter& R)
      : astContext(context), rewriter(R) {
    (void)astContext;
  }
  virtual ~OpenCLDeviceRewriter() {}
  void GenerateCLCode(CXXMethodDecl* method) { this->CodeGenMethod(method); }
  virtual bool CodeGenMethod(CXXMethodDecl* method) {
    std::string kernel_name = "__kernel void ";
    if (method->isConst()) {
      QualType q = method->getType();
      q.removeLocalConst();
      method->setType(q);
    }
    kernel_name += method->getParent()->getNameAsString();
    kernel_name += "";
    string param_list = "";
    bool needComma    = false;
    for (auto p : method->getParent()->fields()) {
      varMapping[p] = new CLVar(true, *p);
      if (needComma) {
        param_list += ",";
      }
      needComma = true;
      llvm::outs() << "##Converting parameter :: " << p->getType().getAsString()
                   << " , " << p->getNameAsString() << " \n";
      param_list += varMapping[p]->localDecl();
    }
    llvm::outs() << " ========================================================="
                    "=======================\n";
    for (auto v : method->getLexicalParent()->decls()) {
      llvm::outs() << " DECL :: " << v->getDeclKindName() << " \n";
      if (strcmp(v->getDeclKindName(), "Var") == 0) {
        VarDecl* vd    = (VarDecl*)v;
        varMapping[vd] = new CLVar(true, *vd);
        if (needComma)
          param_list += ", ";
        needComma = true;
        param_list += varMapping[vd]->localDecl();
        llvm::outs() << " VARFOUND!!! " << vd->getNameAsString() << "\n";
      }
    }
    llvm::outs() << " ========================================================="
                    "=======================\n";
    if (needComma) {
      param_list += ",";
    }
    param_list += " uint num_items ";

    // TODO RK - handle edge operators?
    string numNodeCondition = "{\nconst uint ";
    numNodeCondition += method->getParamDecl(0)->getNameAsString();
    numNodeCondition += " = get_global_id(0);\n if (";
    numNodeCondition += method->getParamDecl(0)->getNameAsString();
    numNodeCondition += " < num_items) {\n";
    rewriter.ReplaceText(method->getNameInfo().getSourceRange(),
                         kernel_name.c_str());
    SourceRange s(method->getParamDecl(0)->getLocEnd(),
                  method->getBody()->getLocStart());
    param_list += " ) \n";
    rewriter.ReplaceText(
        SourceRange(method->getParamDecl(0)->getSourceRange().getBegin(),
                    method->getBody()->getLocStart()),
        param_list);
    rewriter.RemoveText(method->getReturnTypeSourceRange());
    rewriter.InsertText(method->getBody()->getLocStart(), numNodeCondition);
    rewriter.InsertText(method->getBodyRBrace(), "}//End if num_items\n");
    // Visit the method body.
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(method->getBody());
    return true;
  }
  virtual bool TraverseVarDecl(VarDecl* var) {
    MARKER_CODE(llvm::outs() << "[VARDEL]" << var->getNameAsString() << "\n";)
    varMapping[var] = new CLVar(false, *var);
    SourceRange sr(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                   var->getTypeSourceInfo()->getTypeLoc().getEndLoc());
    //      rewriter.ReplaceText(sr,
    //      OpenCLConversionDB::type_convert(var->getType()));
    rewriter.ReplaceText(sr, varMapping[var]->getType());
    if (var->hasInit()) {
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(var->getInit());
    }
    MARKER_CODE(llvm::outs() << "[END-VARDEL]";)
    return true;
  }
  virtual bool TraverseCallExpr(CallExpr* call) {
    MARKER_CODE(cl_file << "[CallExpr]";)
    llvm::outs()
        << " CallExpression ["
        << call->getDirectCallee()->getCanonicalDecl()->getNameAsString()
        << " ]\n";
    string fnameReplacement =
        OpenCLConversionDB::get_cl_implementation(call->getDirectCallee());
    rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(),
                                     call->getCallee()->getLocEnd()),
                         fnameReplacement);
    if (fnameReplacement.find("atomic") != string::npos) {
      rewriter.InsertTextBefore(call->getArg(0)->getLocStart(), "&");
    }
    for (auto a : call->arguments()) {
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
    }
    MARKER_CODE(cl_file << "[EndCallExpr]\n";)
    return true;
  }
  /*
   * */
  virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr* call) {
    MARKER_CODE(cl_file << "[CXXMemberCallExpr]";)
    llvm::outs()
        << "CXXMemberCallExpression ["
        << call->getDirectCallee()->getCanonicalDecl()->getNameAsString()
        << " ]\n";
    std::string fnameReplacement =
        OpenCLConversionDB::get_cl_implementation(call->getDirectCallee());
    if (fnameReplacement == "") {
      // Cast operator or just a load from an atomic type.
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(
          call->getImplicitObjectArgument());
      string objArg = rewriter.getRewrittenText(
          SourceRange(call->getImplicitObjectArgument()->getLocStart(),
                      call->getImplicitObjectArgument()->getLocEnd()));
      rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(),
                                       call->getCallee()->getLocEnd()),
                           objArg);
    } else {
      std::string objArg = "";
      llvm::outs() << " Replacing function :: " << fnameReplacement << "\n";
      if (fnameReplacement.find("atomic") != string::npos) {
        // If the replacement is an atomic_* operator, the first argument is a
        // pointer, hence use address-of.
        objArg += "&";
      }
      if (!call->isImplicitCXXThis()) {
        // If the 'this' argument is not implicit (hence this is not an
        // operator's instance method), we need to convert the 'this' argument to
        // a parameter to be passed to the translated function. graph->begin()
        // becomes begin(graph)
        RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(
            call->getImplicitObjectArgument());
        objArg += rewriter.getRewrittenText(
            SourceRange(call->getImplicitObjectArgument()->getLocStart(),
                        call->getImplicitObjectArgument()->getLocEnd()));
      }

      SourceLocation argStart;
      argStart = call->getRParenLoc();
      if (call->getNumArgs() == 0) {
        //            rewriter.InsertTextBefore(argStart, objArg);
      } else {
        objArg = "," + objArg;
        //            argStart=call->getArg(0)->getLocStart() ;
      }
      rewriter.InsertTextBefore(argStart, objArg);
      rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(),
                                       call->getCallee()->getLocEnd()),
                           fnameReplacement);
      for (auto a : call->arguments()) {
        RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
      }
    }
    MARKER_CODE(cl_file << "[EndCXXMemberCallExpr]\n";)
    return true;
  }
  virtual bool TraverseDeclStmt(DeclStmt* ds) {
    MARKER_CODE(cl_file << "[DECLStmt]";)
    auto curr_decl = ds->decl_begin();
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(*curr_decl);
    curr_decl++;
    while (curr_decl != ds->decl_end()) {
      VarDecl* decl = dyn_cast<VarDecl>(*curr_decl);
      if (decl) {
        if (decl->hasInit()) {
          RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(
              decl->getInit());
        }
      }
      curr_decl++;
    }
    MARKER_CODE(cl_file << "[EndDeclStmt]";)
    return true;
  }
  virtual bool TraverseDeclRefExpr(DeclRefExpr* ref) {
    MARKER_CODE(cl_file << "[DeclRef]";)
    //         cl_file << ref->getNameInfo().getAsString();
    if (varMapping.find(ref->getDecl()) != varMapping.end()) {
      SourceRange sr(ref->getLocStart(), ref->getLocEnd());
      rewriter.ReplaceText(sr, varMapping[ref->getDecl()]->deref());
      //         rewriter.InsertText(ref->getLocStart(), "/*T*/");
    } else {
      //         rewriter.InsertText(ref->getLocStart(), "/*F*/");
    }
    MARKER_CODE(cl_file << "[EndDeclRef]";)
    return true;
  }
  virtual bool TraverseMemberExpr(MemberExpr* mem) {
    MARKER_CODE(cl_file << "[MemberExpr]";)
    /*if (varMapping.find(mem->getMemberDecl()) != varMapping.end()) {
     if (varMapping[mem->getMemberDecl()]->isPointer) {
     llvm::outs()<<"MemberExpression ------> :: \n";
     mem->dump(llvm::outs());
     llvm::outs()<<"EndMemberExpression <--------------------\n";
     //            rewriter.ReplaceText(mem->getOperatorLoc(), 1, "*");
     rewriter.InsertText(mem->getLocStart(), "*");
     }
     }*/
    if (OpenCLConversionDB::type_convert(mem->getBase()->getType()) ==
        "__global NodeData *") {
      rewriter.ReplaceText(mem->getOperatorLoc(), 1, "->");
    }
    //      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(mem->getBase());
    //      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(mem->getMemberDecl());
    MARKER_CODE(cl_file << "[EndMemberExpr]";)
    return true;
  }
  virtual bool TraverseBinAssign(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinAddAssign(CompoundAssignOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinAdd(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinSub(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinMul(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinDiv(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinGT(BinaryOperator* bop) {
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseCCstyleCastExpr(clang::CStyleCastExpr* cexp) {
    llvm::outs() << " CASTEXPR:: ";
    cexp->dump(llvm::outs());
    llvm::outs() << "\n";
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(cexp->getSubExpr());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseParenExpr(clang::ParenExpr* pexp) {
    llvm::outs() << " PARENEXPR:: ";
    pexp->dump(llvm::outs());
    llvm::outs() << "\n";
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(pexp->getSubExpr());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseImplicitCastExpr(ImplicitCastExpr* icexpr) {
    llvm::outs() << " ImplicitCast:: ";
    icexpr->dump(llvm::outs());
    llvm::outs() << "\n";
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(
        icexpr->getSubExpr());
    return true; // this->TraversBinaryOperatorGeneric(bop);
  }
  //   virtual bool TraverseMemberExpr(MemberExpr* texpr){
  //      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(texpr->getMemberDecl());
  //      return true;
  //   }
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceHandler : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  galois::GAST::GaloisApp& app_data;

public:
  OpenCLDeviceHandler(Rewriter& rewriter, galois::GAST::GaloisApp& a)
      : rewriter(rewriter), app_data(a) {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    CXXMethodDecl* decl = const_cast<CXXMethodDecl*>(
        Results.Nodes.getNodeAs<CXXMethodDecl>("kernel"));
    llvm::outs() << "=========================================================="
                    "==================\n";
    llvm::outs() << "Processing Method :: "
                 << decl->getParent()->getNameAsString() << "\n";
    llvm::outs() << "=========================================================="
                    "==================\n";
    // TODO Fix the type system to use a central type repository for graph
    // accesses.
    OpenCLDeviceRewriter devVisitor(decl->getASTContext(), rewriter);
    devVisitor.GenerateCLCode(decl);
    char filename[1024];
    sprintf(filename, "%s/%s.cl", app_data.dir,
            decl->getParent()->getNameAsString().c_str());
    ofstream impl_file(filename);
    impl_file << " // Implementation for kernel in "
              << decl->getParent()->getNameAsString() << "\n";
    impl_file << " // Generated at " << ast_utility.get_timestamp() << "\n";
    impl_file << " #include \"app_header.h\"\n\n";
    impl_file << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
    impl_file.close();
  }
};

#endif /* SRC_PLUGINS_CLIFY_KERNEL_GEN_H_ */
