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
 * Analysis.h
 *
 *  Created on: Dec 8, 2015
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
#include "ClangUtils.h"

/*
 *  * Matchers
 *   */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

#ifndef SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_
#define SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

#define MARKER_CODE(X)                                                         \
  {}
#define _MARKER_CODE(X)                                                        \
  { X }
namespace {
class CLTypeMapper {
  std::map<string, string> known_types;
  CLTypeMapper() {}
  ~CLTypeMapper() {}
  static std::string get_cl_type(std::string src_name) {
    return string(" __global uint * ");
  }
};
/////////////////////////////////////////////////////////////////////////
/*********************************************************************************************************
 *
 *********************************************************************************************************/

class TraverseOrderCheck : public RecursiveASTVisitor<TraverseOrderCheck> {
  int t_counter, v_counter, d_counter;

public:
  explicit TraverseOrderCheck() {
    v_counter = 0;
    t_counter = 0;
    d_counter = 0;
  }
  virtual ~TraverseOrderCheck() {}
  void Check(Decl* method) {
    RecursiveASTVisitor<TraverseOrderCheck>::TraverseDecl(method);
  }
  virtual bool TraverseStmt(Stmt* s) {
    llvm::outs() << " TraversingStmt " << t_counter++ << "\n";
    s->dump(llvm::outs());
    RecursiveASTVisitor<TraverseOrderCheck>::TraverseStmt(s);
    llvm::outs() << " EndTraversingStmt " << --t_counter << "\n";
    return true;
  }
  virtual bool TraverseBinaryOperator(BinaryOperator* bop) {
    llvm::outs() << "\n################BINOP##################\n";
    //      RecursiveASTVisitor<OpenCLASTGen>::TraverseE
    return true;
  }
  virtual bool TraverseBinDiv(BinaryOperator* bop) {
    llvm::outs() << "\n################BINDIV##################\n";
    RecursiveASTVisitor<TraverseOrderCheck>::TraverseBinDiv(bop);
    return true;
  }
  virtual bool TraverseDecl(Decl* s) {
    llvm::outs() << " TraversingDecl " << d_counter++ << "\n";
    s->dump(llvm::outs());
    RecursiveASTVisitor<TraverseOrderCheck>::TraverseDecl(s);
    llvm::outs() << " EndTraversingDecl " << --d_counter << "\n";
    return true;
  }
  virtual bool VisitStmt(Stmt* s) {
    llvm::outs() << " Visiting " << v_counter++ << "\n";
    s->dump(llvm::outs());
    RecursiveASTVisitor<TraverseOrderCheck>::VisitStmt(s);
    llvm::outs() << " EndVisiting " << --v_counter << "\n";
    return true;
  }

}; /////////////////////////////////////////////////////////////////////////

class TypeCollector : public RecursiveASTVisitor<TypeCollector> {
  ASTContext& astContext;
  Rewriter& rewriter;
  int t_counter, v_counter, d_counter;
  std::set<string> named_types;

public:
  explicit TypeCollector(ASTContext& context, Rewriter& R)
      : astContext(context), rewriter(R) {
    v_counter = 0;
    t_counter = 0;
    d_counter = 0;
  }
  virtual ~TypeCollector() {}
  void Check(Decl* d) {
    llvm::outs() << "About to collect types\n";
    RecursiveASTVisitor<TypeCollector>::TraverseDecl(d);
    llvm::outs() << "Done collecting types\n";
  }
  virtual bool VisitVarDecl(VarDecl* d) {
    llvm::outs() << " Declaration[VD] " << d->getType().getAsString() << ", "
                 << d->getNameAsString() << " \n";
    return true;
  }
  virtual bool VisitParmVarDecl(ParmVarDecl* d) {
    llvm::outs() << " Declaration[PVD] " << d->getType().getAsString() << ", "
                 << d->getNameAsString() << " \n";
    return true;
  }
  virtual bool VisitFieldDecl(FieldDecl* d) {
    llvm::outs() << " Declaration[FD] " << d->getType().getAsString() << ", "
                 << d->getNameAsString() << " \n";
    return true;
  }
}; /////////////////////////////////////////////////////////////////////////

struct OpenCLConversionDB {
  static const char* get_cl_implementation(FunctionDecl* d) {
    //      llvm::outs()<<"About to process call :: " << d->getNameAsString() <<
    //      "\n";
    const string fname = d->getNameAsString();
    if (d->isOverloadedOperator()) {
      const char* op_name = getOperatorSpelling(d->getOverloadedOperator());
      if (strcmp(op_name, "operator int") == 0) {
        return "CAST_OP";
      }
      return getOperatorSpelling(d->getOverloadedOperator());
    }
    if (d->getNumParams() == 0) {
      if (fname == "min") {
        return "INT_MIN";
      } else if (fname == "max") {
        return "INT_MAX";
      } else if (fname == "operator int") {
        return "";
      }
    } else if (d->getNumParams() == 1) {
      if (fname == "compare_exchange_strong") {
        return "atomic_cmpxchg";
      } else if (fname == "edge_begin") {
        return "edge_begin";
      } else if (fname == "edge_end") {
        return "edge_end";
      } else if (fname == "getEdgeDst") {
        return "getEdgeDst";
      }
    } else if (d->getNumParams() == 2) {
      if (fname == "compare_exchange_strong") {
        return "atomic_cmpxchg";
      } else if (fname == "getData") {
        return "getData";
      } else if (fname == "getEdgeData") {
        return "getEdgeData";
      }
    } else if (d->getNumParams() == 3) {
      if (fname == "compare_exchange_strong") {
        return "atomic_cmpxchg";
      }
    }
    {
      string s = fname;
      s += " /*UNINTERPRETED-";
      s += d->getNumParams();
      s += "*/";
      return s.c_str();
    }
  }
  static string type_convert(const QualType& qt) {
    // TODO RK - Filter primitive types.
    llvm::outs() << " TypeConversion :: " << qt.getAsString() << "\n";
    if (qt.getAsString() == "int") {
      return "int ";
    } else if (qt.getAsString() == "float") {
      return "float";
    } else if (qt.getAsString() == "double") {
      return "double";
    } else if (qt.getAsString() == "char") {
      return "char";
    }
    // Atomic stripper
    if (qt.getAsString().find("std::atomic") != string::npos) {
      return qt.getAsString().substr(qt.getAsString().rfind("<") + 1,
                                     qt.getAsString().rfind(">") -
                                         qt.getAsString().rfind("<") - 1);
    }
    // Iterator stripper
    if (qt.getAsString().find("boost::iterators::counting_iterator") !=
        string::npos) {
      return "edge_iterator ";
    }
    if (qt.getAsString().find("GNode") != string::npos) {
      return "node_iterator";
    }
    // Wrapper for nodedata.
    if (qt.getAsString().find("NodeData") != string::npos) {
      return "__global NodeData *";
    }
    if (qt.getAsString().find("Graph") != string::npos) {
      return "__global Graph *";
    }
    return qt.getAsString();
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceVisitor : public RecursiveASTVisitor<OpenCLDeviceVisitor> {
  ASTContext& astContext;
  Rewriter& rewriter;
  std::ofstream cl_file;
  long stmtIndex;
  std::map<const Type*, string>& type_sub;

public:
  explicit OpenCLDeviceVisitor(ASTContext& context, Rewriter& R,
                               std::map<const Type*, string>& t)
      : astContext(context), rewriter(R), stmtIndex(0), type_sub(t) {}
  virtual ~OpenCLDeviceVisitor() {}
  void GenerateCLCode(CXXMethodDecl* method) {
    std::string gen_name = method->getParent()->getNameAsString();
    gen_name.append(".cl");
    cl_file.open(gen_name.c_str());
    cl_file << " //Generating OpenCL code for " << gen_name << "\n";
    cl_file << " //Timestamp :: " << ast_utility.get_timestamp() << "\n";
    cl_file << " #include \"app_header.h\"\n";
    this->CodeGenMethod(method);
    cl_file << "//END CODE GEN\n";
  }
  virtual bool CodeGenMethod(CXXMethodDecl* method) {
    std::string kernel_name = "__kernel void ";
    kernel_name += method->getNameAsString();
    cl_file << " __kernel void " << method->getParent()->getNameAsString()
            << " ( ";
    for (auto p : method->getParent()->fields()) {
      cl_file << "__global " << p->getType().getAsString() << " ";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(p);
      cl_file << " , ";
    }
    cl_file << " uint num_items";
    cl_file << "){\n";
    if (method->parameters().size() > 0) {
      cl_file << " const uint " << method->parameters()[0]->getNameAsString()
              << " = get_global_id(0);\n";
      cl_file << " if ( " << method->parameters()[0]->getNameAsString()
              << " < num_items) { \n";
    }
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(method->getBody());
    cl_file << " }//End if \n";
    cl_file << " }\n";
    return true;
  }
  virtual bool TraverseParmVarDecl(ParmVarDecl* p) {
    _MARKER_CODE(cl_file << "[ParmVarDecl]";)
    string tname = p->getType().getAsString();
    cl_file << tname << " " << p->getNameAsString() << " ";
    _MARKER_CODE(cl_file << "[EndParmVarDecl]";)
    return true;
  }
  virtual bool TraverseFieldDecl(FieldDecl* p) {
    MARKER_CODE(cl_file << "[ParmVarDecl]";)
    cl_file << p->getNameAsString() << " ";
    MARKER_CODE(cl_file << "[EndParmVarDecl]";)
    return true;
  }
  virtual bool TraverseDeclStmt(DeclStmt* ds) {
    MARKER_CODE(cl_file << "[DECLStmt]";)
    auto curr_decl = ds->decl_begin();
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(*curr_decl);
    curr_decl++;
    while (curr_decl != ds->decl_end()) {
      VarDecl* decl = dyn_cast<VarDecl>(*curr_decl);
      if (decl) {
        cl_file << ", " << decl->getNameAsString() << " ";
        if (decl->hasInit()) {
          cl_file << " = ";
          RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(
              decl->getInit());
        }
      }
      curr_decl++;
    }
    MARKER_CODE(cl_file << "[EndDeclStmt]";)
    return true;
  }
  //   string getTypeName(const QualType& qt){
  //      llvm::outs() << " ===============================@@@@@@@" <<
  //      qt.getAsString()<<"@@@@@@@@@=============================\n";
  //      //TODO RK - Filter primitive types.
  //      if(qt.getAsString()=="int"){
  //         return "int";
  //      }else if(qt.getAsString()=="float"){
  //         return "float";
  //      }else if(qt.getAsString()=="double"){
  //         return "double";
  //      }else if(qt.getAsString()=="char"){
  //         return "char";
  //      }
  //      if(qt.getAsString().find ("std::atomic")!=string::npos){
  //         return
  //         qt.getAsString().substr(qt.getAsString().rfind("<")+1,qt.getAsString().rfind(">")-qt.getAsString().rfind("<")-1);
  //      }
  //      if (qt.getAsString().find ("boost::iterators::counting_iterator") ||
  //      qt.getAsString().find("GNode")!=string::npos){
  //         return "int";
  //      }
  //      if(qt.getAsString().find("NodeData")!=string::npos){
  //         return "__global NodeData *";
  //      }
  //      /*for(auto t : type_sub){
  //         if(qt.getTypePtr()->getCanonicalTypeUnqualified()==t.first->getCanonicalTypeUnqualified()){
  //            llvm::outs() << "TRUE";
  //         }else{
  //            llvm::outs() << "FALSE";
  //         }
  //
  //         llvm::outs() << " Comparing " << qt.getAsString() << " AND " <<
  //         QualType::getAsString(t.first,Qualifiers()) << " MAP " << t.second
  //         <<"\n";
  //      }
  //
  //      auto it = type_sub.find(qt.getTypePtr());
  //      if(it!=type_sub.end()){
  //         return it->second;
  //      }
  //      else{
  //         return qt.getAsString( ) ; //QualType::getAsString(t,
  //         Qualifiers());
  //      }*/
  //      return qt.getAsString();
  //   }
  virtual bool TraverseVarDecl(VarDecl* var) {
    MARKER_CODE(cl_file << "[VARDEL]";)
    /*string tname = var->getType().getAsString();
    if(tname.find("iterator")!=string::npos){
       cl_file << " int " ;
    }else{
       cl_file << var->getType().getAsString() ;
    }*/
    cl_file << OpenCLConversionDB::type_convert(var->getType()) << " "
            << var->getNameAsString() << " ";
    if (var->hasInit()) {
      cl_file << " = ";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(var->getInit());
    }
    MARKER_CODE(cl_file << "[END-VARDEL]";)
    return true;
  }
  virtual bool TraverseCallExpr(CallExpr* call) {
    MARKER_CODE(cl_file << "[CallExpr]";)
    llvm::outs()
        << " CallExpression ["
        << call->getDirectCallee()->getCanonicalDecl()->getNameAsString()
        << " ]\n";
    if (const char* c = OpenCLConversionDB::get_cl_implementation(
            call->getDirectCallee())) {
      cl_file << c;
      return true;
    }
    cl_file << call->getDirectCallee()->getNameAsString() << " (";
    bool first = true;
    for (auto a : call->arguments()) {
      if (!a->isDefaultArgument()) {
        if (!first) {
          cl_file << ",";
        }
        first = false;
        MARKER_CODE(cl_file << " ARG(";)
        RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
        MARKER_CODE(cl_file << ")";)
      }
    }
    cl_file << " )";
    MARKER_CODE(cl_file << "[EndCallExpr]\n";)
    return true;
  }
  virtual bool TraverseDeclRefExpr(DeclRefExpr* ref) {
    MARKER_CODE(cl_file << "[DeclRef]";)
    cl_file << ref->getNameInfo().getAsString();
    MARKER_CODE(cl_file << "[EndDeclRef]";)
    return true;
  }
  virtual bool TraverseCompoundStmt(CompoundStmt* s) {
    stmtIndex *= 10;
    for (auto a : s->body()) {
      cl_file << " /*" << stmtIndex++ << "*/";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
      cl_file << ";\n";
    }
    stmtIndex /= 10;
    return true;
  }
  virtual bool TraverseStmt(Stmt* s) {
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(s);
    if (s && !isa<Expr>(s))
      cl_file << ";/**/\n";
    return true;
  }
  virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr* call) {
    MARKER_CODE(cl_file << "[CXXMemberCallExpr]";)
    //      llvm::outs() << "Processing callExpr";
    //      llvm::outs()<<call->getDirectCallee()->getNameAsString() << ", "<<
    //      call->getDirectCallee()->getNumParams( ); llvm::outs()<<"\n";
    //      cl_file << call->getDirectCallee()->getNameAsString() << " (";
    //      cl_file << call->getDirectCallee()->getNameAsString() << " (";
    cl_file << OpenCLConversionDB::get_cl_implementation(
                   call->getDirectCallee())
            << "(";
    bool firstArg = true;
    if (!call->isImplicitCXXThis()) {
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(
          call->getImplicitObjectArgument());
      firstArg = false;
    }
    for (auto a : call->arguments()) {
      if (!a->isDefaultArgument()) {
        MARKER_CODE(cl_file << " MARG[";)
        if (!firstArg)
          cl_file << ",";
        RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
        firstArg = false;
        MARKER_CODE(cl_file << "]";)
      }
    }
    cl_file << " )";
    MARKER_CODE(cl_file << "[EndCXXMemberCallExpr]\n";)
    return true;
  }
  // TODO RK - Fix this to be correct for operator implementations
  // (pre/post/multi-args)
  virtual bool TraverseCXXOperatorCallExpr(CXXOperatorCallExpr* op) {
    MARKER_CODE(cl_file << " [CXXOp]";)
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(op->getArg(0));
    cl_file << toString(op->getOperator());
    if (std::distance(op->arg_begin(), op->arg_end()) > 1)
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(op->getArg(1));

    MARKER_CODE(cl_file << " [EndCXXOp]";)
    return true;
  }
  virtual bool TraversBinaryOperatorGeneric(BinaryOperator* bop) {
    MARKER_CODE(cl_file << " [BinOp]";)
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(bop->getLHS());
    cl_file << " " << bop->getOpcodeStr().str() << " ";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(bop->getRHS());
    MARKER_CODE(cl_file << " [EndBinOp]";)
    return true;
  }
  virtual bool TraverseBinDiv(BinaryOperator* bop) {
    return this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinAdd(BinaryOperator* bop) {
    return this->TraversBinaryOperatorGeneric(bop);
  }

  virtual bool TraverseBinAssign(BinaryOperator* bop) {
    return this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinNE(BinaryOperator* bop) {
    return this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseBinLT(BinaryOperator* bop) {
    return this->TraversBinaryOperatorGeneric(bop);
  }
  virtual bool TraverseUnaryPostInc(UnaryOperator* uop) {
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(uop->getSubExpr());
    cl_file << "++";
    return true;
  }
  virtual bool TraverseUnaryPreInc(UnaryOperator* uop) {
    cl_file << "++";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(uop->getSubExpr());
    return true;
  }
  virtual bool TraverseMemberExpr(MemberExpr* mem) {
    MARKER_CODE(cl_file << "[MemberExpr]";)
    if (!mem->isImplicitCXXThis() && !mem->getBase()->isImplicitCXXThis()) {
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(mem->getBase());
      if (mem->isArrow()) {
        cl_file << " ->";
      } else {
        cl_file << ".";
      }
    }
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(
        mem->getMemberDecl());
    MARKER_CODE(cl_file << "[EndMemberExpr]";)
    return true;
  }
  virtual bool TraverseIntegerLiteral(IntegerLiteral* i) {
    cl_file << i->getValue().toString(10, true);
    return true;
  }
  virtual bool TraverseStringLiteral(StringLiteral* bl) {
    cl_file << "\"" << bl->getBytes().str() << "\"";
    return true;
  }
  virtual bool TraverseCXXBoolLiteralExpr(CXXBoolLiteralExpr* bl) {
    cl_file << ((bl->getValue() != 0) ? "true" : "false");
    return true;
  }
  virtual bool TraverseIfStmt(IfStmt* i) {
    cl_file << " if ( ";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getCond());
    cl_file << "){\n";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getThen());
    cl_file << " \n}/*End if*/ else{\n";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getElse());
    cl_file << " \n}/*End if-else*/ \n";
    return true;
  }
  virtual bool TraverseForStmt(ForStmt* fs) {
    cl_file << " for (";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getInit());
    cl_file << ";";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getCond());
    cl_file << ";";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getInc());
    cl_file << ") {\n";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getBody());
    cl_file << "\n}//End For\n";
    return true;
  }
  virtual bool TraverseWhileStmt(WhileStmt* fs) {
    cl_file << " while(";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getCond());
    cl_file << ") {\n";
    RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getBody());
    cl_file << "\n}//End While\n";
    return true;
  }
  string toString(Stmt* s) {
    std::string res;
    raw_string_ostream s_stream(res);
    s->dump(s_stream);
    return s_stream.str();
  }
  string toString(OverloadedOperatorKind op) {
    return string(clang::getOperatorSpelling(op));
  }
  //   virtual bool _TraverseImplicitCastExpr(ImplicitCastExpr * ce){
  //      cl_file << "\nImplicitCastExpr!!!\n";
  //      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(ce->getSubExprAsWritten());
  //      return true;
  //   }
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceRewriter : public RecursiveASTVisitor<OpenCLDeviceRewriter> {
  ASTContext& astContext;
  Rewriter& rewriter;
  std::map<const Type*, string>& type_sub;

public:
  explicit OpenCLDeviceRewriter(ASTContext& context, Rewriter& R,
                                std::map<const Type*, string>& t)
      : astContext(context), rewriter(R), type_sub(t) {}
  virtual ~OpenCLDeviceRewriter() {}
  void GenerateCLCode(CXXMethodDecl* method) { this->CodeGenMethod(method); }
  virtual bool CodeGenMethod(CXXMethodDecl* method) {
    std::string kernel_name = "__kernel void ";
    kernel_name += method->getParent()->getNameAsString();
    kernel_name += "";
    string param_list = "";
    bool needComma    = false;
    for (auto p : method->getParent()->fields()) {
      if (needComma) {
        param_list += ",";
      }
      needComma = true;
      //         param_list+= "__global ";
      param_list += OpenCLConversionDB::type_convert(p->getType());
      //         param_list+= " * ";
      param_list += p->getNameAsString();
    }
    if (needComma) {
      param_list += ",";
    }
    param_list += " uint num_items ";

    string numNodeCondition = "{\nconst uint ";
    numNodeCondition += method->getParamDecl(0)->getNameAsString();
    numNodeCondition += " = get_global_id(0);\n if (";
    numNodeCondition += method->getParamDecl(0)->getNameAsString();
    numNodeCondition += " < num_nodes)";
    rewriter.ReplaceText(method->getNameInfo().getSourceRange(),
                         kernel_name.c_str());
    rewriter.ReplaceText(method->getParamDecl(0)->getSourceRange(), param_list);
    rewriter.RemoveText(method->getReturnTypeSourceRange());
    rewriter.InsertText(method->getBody()->getLocStart(), numNodeCondition);
    rewriter.InsertText(method->getBodyRBrace(), "}//End if num_items\n");
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(method->getBody());
    return true;
  }
  virtual bool TraverseVarDecl(VarDecl* var) {
    MARKER_CODE(llvm::outs() << "[VARDEL]" << var->getNameAsString() << "\n";)
    SourceRange sr(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                   var->getTypeSourceInfo()->getTypeLoc().getEndLoc());
    rewriter.ReplaceText(sr, OpenCLConversionDB::type_convert(var->getType()));

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
    rewriter.ReplaceText(
        SourceRange(call->getCallee()->getLocStart(),
                    call->getCallee()->getLocEnd()),
        OpenCLConversionDB::get_cl_implementation(call->getDirectCallee()));
    for (auto a : call->arguments()) {
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
    }
    MARKER_CODE(cl_file << "[EndCallExpr]\n";)
    return true;
  }
  virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr* call) {
    MARKER_CODE(cl_file << "[CXXMemberCallExpr]";)
    std::string fnameReplacement =
        OpenCLConversionDB::get_cl_implementation(call->getDirectCallee());
    if (fnameReplacement == "") {
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
      if (!call->isImplicitCXXThis()) {
        RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(
            call->getImplicitObjectArgument());
        objArg = rewriter.getRewrittenText(
            SourceRange(call->getImplicitObjectArgument()->getLocStart(),
                        call->getImplicitObjectArgument()->getLocEnd()));
        //            SourceRange sr(call->getLocStart(),
        //            call->getCallee()->getLocStart());
        //            rewriter.RemoveText(sr);
      }

      if (call->getNumArgs() == 0) {
        rewriter.InsertTextAfter(call->getRParenLoc(), objArg);
      } else {
        objArg += ", ";
        rewriter.InsertTextBefore(call->getArg(0)->getExprLoc(), objArg);
      }
      for (auto a : call->arguments()) {
        RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
      }
      rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(),
                                       call->getCallee()->getLocEnd()),
                           fnameReplacement);
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

    MARKER_CODE(cl_file << "[EndDeclRef]";)
    return true;
  }
  virtual bool TraverseMemberExpr(MemberExpr* mem) {
    MARKER_CODE(cl_file << "[MemberExpr]";)
    if (OpenCLConversionDB::type_convert(mem->getBase()->getType()) ==
        "__global NodeData *") {
      rewriter.ReplaceText(mem->getOperatorLoc(), 1, "->");
    }
    RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(
        mem->getMemberDecl());
    MARKER_CODE(cl_file << "[EndMemberExpr]";)
    return true;
  }
#if 0
   virtual bool TraverseParmVarDecl(ParmVarDecl * p){
      _MARKER_CODE(cl_file << "[ParmVarDecl]";)
      string tname = p->getType().getAsString();
      cl_file << tname << " " << p->getNameAsString() << " ";
      _MARKER_CODE(cl_file << "[EndParmVarDecl]";)
      return true;
   }
   virtual bool TraverseFieldDecl(FieldDecl * p){
      MARKER_CODE(cl_file << "[ParmVarDecl]";)
      cl_file << p->getNameAsString() << " ";
      MARKER_CODE(cl_file << "[EndParmVarDecl]";)
      return true;
   }



    virtual bool TraverseCompoundStmt (CompoundStmt * s ){
      stmtIndex*=10;
      for(auto a : s->body()){
         cl_file << " /*" <<  stmtIndex++ << "*/";
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
         cl_file << ";\n";
      }
      stmtIndex/=10;
      return true;
   }
   virtual bool TraverseStmt (Stmt * s ){
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(s);
         if(s && !isa<Expr>(s))
            cl_file << ";/**/\n";
         return true;
   }

   //TODO RK - Fix this to be correct for operator implementations (pre/post/multi-args)
   virtual bool TraverseCXXOperatorCallExpr (CXXOperatorCallExpr * op){
      MARKER_CODE(cl_file << " [CXXOp]";)
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(op->getArg(0));
      cl_file << toString(op->getOperator());
      if(std::distance(op->arg_begin(), op->arg_end())>1)
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(op->getArg(1));

      MARKER_CODE(cl_file << " [EndCXXOp]";)
      return true;
   }
   virtual bool TraversBinaryOperatorGeneric(BinaryOperator * bop) {
      MARKER_CODE(cl_file << " [BinOp]";)
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getLHS());
      cl_file << " "<<bop->getOpcodeStr().str()<<" ";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(bop->getRHS());
      MARKER_CODE(cl_file << " [EndBinOp]";)
      return true;
      }
      virtual bool TraverseBinDiv(BinaryOperator * bop) {
         return this->TraversBinaryOperatorGeneric(bop);
   }
      virtual bool TraverseBinAdd(BinaryOperator * bop) {
         return this->TraversBinaryOperatorGeneric(bop);
   }

      virtual bool TraverseBinAssign(BinaryOperator * bop) {
      return this->TraversBinaryOperatorGeneric(bop);
   }
   virtual bool TraverseBinNE(BinaryOperator * bop) {
         return this->TraversBinaryOperatorGeneric(bop);
      }
   virtual bool TraverseBinLT(BinaryOperator * bop) {
            return this->TraversBinaryOperatorGeneric(bop);
         }
   virtual bool TraverseUnaryPostInc(UnaryOperator * uop) {
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(uop->getSubExpr());
      cl_file << "++";
      return true;
      }
   virtual bool TraverseUnaryPreInc(UnaryOperator * uop) {
         cl_file << "++";
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(uop->getSubExpr());
         return true;
         }
      virtual bool TraverseMemberExpr(MemberExpr * mem) {
      MARKER_CODE(cl_file << "[MemberExpr]";)
      if (!mem->isImplicitCXXThis()  && !mem->getBase()->isImplicitCXXThis()) {
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(mem->getBase());
         if ( mem->isArrow()) {
            cl_file << " ->";
         } else {
            cl_file << ".";
         }
      }
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(mem->getMemberDecl());
      MARKER_CODE(cl_file << "[EndMemberExpr]";)
      return true;
   }
   virtual bool TraverseIntegerLiteral(IntegerLiteral * i){
         cl_file << i->getValue().toString(10,true) ;
         return true;
      }
   virtual bool TraverseStringLiteral(StringLiteral * bl){
      cl_file << "\""<<bl->getBytes().str()<<"\"";
      return true;
   }
   virtual bool TraverseCXXBoolLiteralExpr(CXXBoolLiteralExpr * bl){
      cl_file << ((bl->getValue()!=0)?"true":"false");
      return true;
   }
   virtual bool TraverseIfStmt(IfStmt * i){
      cl_file << " if ( ";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(i->getCond());
      cl_file << "){\n";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(i->getThen());
      cl_file << " \n}/*End if*/ else{\n";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(i->getElse());
      cl_file << " \n}/*End if-else*/ \n";
      return true;
   }
   virtual bool TraverseForStmt(ForStmt  * fs){
      cl_file << " for (";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getInit());
      cl_file<<";";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getCond());
      cl_file<<";";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getInc());
      cl_file << ") {\n";
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getBody());
      cl_file << "\n}//End For\n";
      return true;
   }
   virtual bool TraverseWhileStmt(WhileStmt  * fs){
         cl_file << " while(";
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getCond());
         cl_file << ") {\n";
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(fs->getBody());
         cl_file << "\n}//End While\n";
         return true;
      }
   string toString(Stmt * s){
      std::string res;
      raw_string_ostream s_stream(res);
      s->dump(s_stream);
      return s_stream.str();
   }
   string toString(OverloadedOperatorKind op){
      return string(clang::getOperatorSpelling(op));
   }
//   virtual bool _TraverseImplicitCastExpr(ImplicitCastExpr * ce){
//      cl_file << "\nImplicitCastExpr!!!\n";
//      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(ce->getSubExprAsWritten());
//      return true;
//   }
#endif
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class GraphTypeCollector : public RecursiveASTVisitor<GraphTypeCollector> {
public:
  std::set<string> named_types;
  std::map<string, string> name_mapping;
  std::map<const Type*, string> type_sub;
  explicit GraphTypeCollector() {}
  virtual ~GraphTypeCollector() {}
  virtual bool VisitCXXMethodDecl(CXXMethodDecl* md) {
    string retType = QualType::getAsString(
        md->getReturnType()->getUnqualifiedDesugaredType(), Qualifiers());
    //      Type * retTypePtr = md->getReturnType().getTypePtr();
    //      retTypePtr->getCanonicalTypeUnqualified();
    if (md->getNameAsString().compare("getData") == 0) {
      type_sub[md->getReturnType().getTypePtr()] = "CLNodeData";
      name_mapping["NodeData"]                   = retType;
      //      }else if(md->getNameAsString().compare("getEdgeDst")==0){
      //         name_mapping["NodeData"]=md->getReturnType().getAsString();
    } else if (md->getNameAsString().compare("getEdgeData") == 0) {
      type_sub[md->getReturnType().getTypePtr()] = "CLEdgeData";
      name_mapping["EdgeData"]                   = retType;
    } else if (md->getNameAsString().compare("begin") == 0) {
      type_sub[md->getReturnType().getTypePtr()] = "CLNodeIterator";
      name_mapping["NodeIterator"]               = retType;
    } else if (md->getNameAsString().compare("edge_begin") == 0) {
      type_sub[md->getReturnType().getTypePtr()] = "CLEdgeIterator";
      name_mapping["EdgeIterator"]               = retType;
    }
    //      llvm::outs() << " Method found " << md->getType().getAsString()  <<
    //      " , " <<md->getNameAsString() << "\n";
    return true;
  }
  void summary() {
    llvm::outs() << " Summary :: \n";
    for (auto i : name_mapping) {
      llvm::outs() << i.first << " mappedTo " << i.second << "\n";
    }
    llvm::outs() << "===============\n";
    for (auto i : type_sub) {
      llvm::outs() << QualType::getAsString(i.first, Qualifiers()) << "("
                   << QualType::getAsString(
                          i.first->getCanonicalTypeUnqualified()->getTypePtr(),
                          Qualifiers())
                   << ")"
                   << " mappedTo " << i.second << "\n";
    }
    llvm::outs() << " End Summary\n";
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceHandler : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  CXXRecordDecl* gClass;

public:
  OpenCLDeviceHandler(Rewriter& rewriter, CXXRecordDecl* g)
      : rewriter(rewriter), gClass(g) {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    CXXMethodDecl* decl = const_cast<CXXMethodDecl*>(
        Results.Nodes.getNodeAs<CXXMethodDecl>("kernel"));
    llvm::outs() << "Processing Method :: "
                 << decl->getParent()->getNameAsString() << "\n";
    // TODO Fix the type system to use a central type repository for graph
    // accesses.
    GraphTypeCollector gtc;
    //      llvm::outs()<< "GraphClass :: " <<gClass->getKindName().str()<< "
    //      \n"; gtc.TraverseDecl(gClass->getTemplateInstantiationPattern());
    //      gtc.summary();
    OpenCLDeviceRewriter devVisitor(decl->getASTContext(), rewriter,
                                    gtc.type_sub);
    devVisitor.GenerateCLCode(decl);
    //      llvm::outs() << "REWRITTEN TEXT :: " <<
    //      decl->getParent()->getNameAsString() << "\n";
    string filename = decl->getParent()->getNameAsString();
    filename += ".cl";
    ofstream impl_file(filename);
    impl_file << " // Implementation for kernel in "
              << decl->getParent()->getNameAsString() << "\n";
    impl_file << " // Generated at " << ast_utility.get_timestamp() << "\n";
    impl_file << " #include \"app_header.h\"\n\n";
    impl_file << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
    impl_file.close();
    //      llvm::outs() << rewriter.getRewrittenText(decl->getSourceRange()) <<
    //      "\n"; llvm::outs() << "END REWRITTEN TEXT :: " <<
    //      decl->getParent()->getNameAsString() << "\n";
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class GraphCollector : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  CXXRecordDecl* graphClass;

public:
  GraphCollector(Rewriter& rewriter)
      : rewriter(rewriter), graphClass(nullptr) {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    CXXRecordDecl* decl = const_cast<CXXRecordDecl*>(
        Results.Nodes.getNodeAs<CXXRecordDecl>("graphClass"));
    graphClass = decl;
    llvm::outs() << " GraphClass Found :: " << decl->getNameAsString() << ", "
                 << decl->getDefinition()->getNameAsString() << "\n";
    //      llvm::outs() << " About to _scan for method declaration in " <<
    //      decl->getNameAsString() <<"\n"; GraphTypeCollector gtc;
    //      gtc.TraverseDecl(decl->getTemplateInstantiationPattern());
    //      gtc.summary();
    //      llvm::outs() << " Done scanning for method declaration\n";
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class NodeDataGen : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  ASTContext& ctx;
  std::ofstream header_file;

public:
  NodeDataGen(Rewriter& rewriter, ASTContext& c) : rewriter(rewriter), ctx(c) {
    header_file.open("app_header.h");
    header_file << "//CL Graph declarations \n";
    header_file << "//Generated : " << ast_utility.get_timestamp() << "\n";
  }
  ~NodeDataGen() {
    header_file << " #include \"graph_header.h\"\n";
    { // Begin graph implementation.
      const char* g_dev = "typedef struct _GraphType { \n\
            uint _num_nodes;\n\
            uint _num_edges;\n\
             uint _node_data_size;\n\
             uint _edge_data_size;\n\
             __global NodeData *_node_data;\n\
             __global uint *_out_index;\n\
             __global uint *_out_neighbors;\n\
             __global EdgeData *_out_edge_data;\n\
             }GraphType;";
      header_file << "\n" << g_dev << "\n";
    }
    header_file << " typedef uint node_iterator;\n";
    header_file << " typedef uint edge_iterator;\n";
    header_file.close();
  }
  void GenCLType(const QualType& q, string sname) {
    std::string res = "typedef ";
    if (q.getTypePtr()->isRecordType()) {
      llvm::outs() << " Processing POD :" << q.getAsString() << " \n";
      RecordDecl* decl = q.getTypePtrOrNull()->getAsCXXRecordDecl();
      res += " _" + sname + "{\n";
      for (auto f : decl->fields()) {
        llvm::outs() << " Field :: " << f->getType().getAsString() << " "
                     << f->getNameAsString() << " \n";
        res += OpenCLConversionDB::type_convert(f->getType());
        res += " ";
        res += f->getNameAsString();
        res += "; \n";
      }
      res += "}";
      res += sname;
      res += ";//End typedef\n";
    } else if (q.isTrivialType(ctx)) {
      llvm::outs() << " Processing trivial type :: " << q.getAsString() << "\n";
      res += q.getAsString();
      res += " ";
      res += sname;
      res += ";\n";
    }
    header_file << res;
    //      llvm::outs() << "Result :: " << res;
  }
  virtual void run(const MatchFinder::MatchResult& Results) {
    {
      TemplateArgument* decl = const_cast<TemplateArgument*>(
          Results.Nodes.getNodeAs<TemplateArgument>("NData"));
      if (decl) {
        llvm::outs() << " NodeData Found :: " << decl->getAsType().getAsString()
                     << "\n";
        GenCLType(decl->getAsType(), "NodeData");
      }
      RecordDecl* gDecl = const_cast<RecordDecl*>(
          Results.Nodes.getNodeAs<RecordDecl>("GraphType"));
      if (gDecl) {
        for (auto f : gDecl->fields()) {
          if (f->getNameAsString().compare("init_kernel_str_CL_LC_Graph") ==
              0) {
            llvm::outs() << " Match found ";
            f->getInClassInitializer()->dump();
            llvm::outs() << "\n";
          }
        }
      }
    }
    {
      TemplateArgument* decl = const_cast<TemplateArgument*>(
          Results.Nodes.getNodeAs<TemplateArgument>("EData"));
      if (decl) {
        llvm::outs() << " EdgeData Found :: " << decl->getAsType().getAsString()
                     << "\n";
        GenCLType(decl->getAsType(), "EdgeData");
      }
    }
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/
} // End anonymous namespace

#endif /* SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_ */
