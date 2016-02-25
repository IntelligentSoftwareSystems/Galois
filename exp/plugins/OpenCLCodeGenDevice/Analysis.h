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

//#include "OpenCLAST.h"

#ifndef SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_
#define SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

#define MARKER_CODE(X) {}
#define _MARKER_CODE(X) {X}
namespace {
class CLTypeMapper{
   static std::string get_cl_type(std::string src_name){
      return string(" __global uint * ");
   }
};
/////////////////////////////////////////////////////////////////////////
/*********************************************************************************************************
 *
 *********************************************************************************************************/

class TraverseOrderCheck: public RecursiveASTVisitor<TraverseOrderCheck> {
   ASTContext & astContext;
   Rewriter & rewriter;
   int t_counter , v_counter, d_counter;
public:
   explicit TraverseOrderCheck(ASTContext & context, Rewriter & R) :
         astContext(context), rewriter(R) {
      v_counter = 0;
      t_counter = 0;
      d_counter = 0;
   }
   virtual ~TraverseOrderCheck() {
   }
   void Check(CXXMethodDecl * method) {
      RecursiveASTVisitor<TraverseOrderCheck>::TraverseDecl(method);
     }
   virtual bool TraverseStmt(Stmt * s){
      llvm::outs() << " TraversingStmt " << t_counter++ <<"\n";
      s->dump(llvm::outs());
      RecursiveASTVisitor<TraverseOrderCheck>::TraverseStmt(s);
      llvm::outs() << " EndTraversingStmt " << --t_counter <<"\n";
      return true;
   }
   virtual bool TraverseBinaryOperator ( BinaryOperator * bop){
      llvm::outs() << "\n################BINOP##################\n";
//      RecursiveASTVisitor<OpenCLASTGen>::TraverseE
      return true;
   }
   virtual bool TraverseBinDiv ( BinaryOperator * bop){
         llvm::outs() << "\n################BINDIV##################\n";
         RecursiveASTVisitor<TraverseOrderCheck>::TraverseBinDiv(bop);
         return true;
      }
   virtual bool TraverseDecl(Decl * s){
      llvm::outs() << " TraversingDecl " << d_counter++ <<"\n";
      s->dump(llvm::outs());
      RecursiveASTVisitor<TraverseOrderCheck>::TraverseDecl(s);
      llvm::outs() << " EndTraversingDecl " << --d_counter <<"\n";
      return true;
   }
   virtual bool VisitStmt( Stmt * s){
      llvm::outs() << " Visiting " << v_counter++ <<"\n";
      s->dump(llvm::outs());
      RecursiveASTVisitor<TraverseOrderCheck>::VisitStmt(s);
      llvm::outs() << " EndVisiting " << --v_counter <<"\n";
      return true;
   }

};/////////////////////////////////////////////////////////////////////////
/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceVisitor: public RecursiveASTVisitor<OpenCLDeviceVisitor> {
   ASTContext & astContext;
   Rewriter & rewriter;
   std::ofstream cl_file;
//   typedef std::pair<int, std::vector<CLNode*>> StackEntry;
//   std::vector<StackEntry> stack;
   long stmtIndex;
public:
   explicit OpenCLDeviceVisitor(ASTContext & context, Rewriter & R) :
         astContext(context), rewriter(R),stmtIndex(0) {
   }
   virtual ~OpenCLDeviceVisitor() {
   }
   void GenerateCLCode(CXXMethodDecl * method) {
      std::string gen_name = method->getParent()->getNameAsString();
      gen_name.append(".cl");
      cl_file.open(gen_name.c_str());
      cl_file << " //Generating OpenCL code for " << gen_name << "\n";
      cl_file << " //Timestamp :: " << ast_utility.get_timestamp() << "\n";
      this->_VisitCXXMethodDecl(method);
      cl_file << "//END CODE GEN\n";
   }
   virtual bool _VisitCXXMethodDecl(CXXMethodDecl * method) {
      std::string kernel_name = "__kernel void ";
      kernel_name += method->getNameAsString();
      cl_file << " __kernel void " << method->getParent()->getNameAsString() << " ( ";
      cl_file << " /*ADD ARGS HERE*/";
      for (auto p : method->getParent()->fields()) {
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(p);
         cl_file << " , ";
      }
      cl_file << " uint num_items";
      cl_file << "){\n";
      if (method->parameters().size() > 0) {
         cl_file << " const uint " << method->parameters()[0]->getNameAsString() << " = get_global_id(0);\n";
         cl_file << " if ( " << method->parameters()[0]->getNameAsString() << " < num_items) { \n";
      }
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(method->getBody());
      cl_file << " }//End if \n";
      cl_file << " }\n";
      return true;
   }
   virtual bool TraverseDeclStmt(DeclStmt * ds){
      MARKER_CODE(cl_file << "[DECLStmt]";)
      auto curr_decl = ds->decl_begin();
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(*curr_decl);
      curr_decl++;
      while(curr_decl != ds->decl_end()){
         VarDecl * decl = dyn_cast<VarDecl>(*curr_decl);
         if(decl){
            cl_file << ", "<<decl->getNameAsString() << " ";
            if (decl->hasInit()){
                  cl_file << " = ";
                  RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(decl->getInit());
            }
         }
         curr_decl++;
      }
      MARKER_CODE(cl_file << "[EndDeclStmt]";)
      return true;
   }
   virtual bool TraverseVarDecl(VarDecl * var) {
      MARKER_CODE(cl_file << "[VARDEL]";)
            string tname = var->getType().getAsString();
            if(tname.find("iterator")!=string::npos){
               cl_file << " int " << var->getNameAsString() << " ";
            }else{
               cl_file << var->getType().getAsString() << " " << var->getNameAsString() << " ";
            }
      if (var->hasInit()) {
         cl_file << " = ";
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(var->getInit());
      }
      MARKER_CODE(cl_file << "[END-VARDEL]";)
      return true;
   }
   virtual bool TraverseCallExpr(CallExpr * call) {
      MARKER_CODE(cl_file << "[CallExpr]";)
//      if (CXXMemberCallExpr * mcall = dyn_cast<CXXMemberCallExpr>(call)) {
//         MARKER_CODE(cl_file << "MCALL ";)
//         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(mcall->getImplicitObjectArgument());
//         cl_file << " , ";
//      }
//      SourceRange sr(call->getLocStart(), call->getLocEnd());
      cl_file << call->getDirectCallee()->getNameAsString() << " (";
      bool first=true;
      for (auto a : call->arguments()) {
         if (!a->isDefaultArgument()) {
            if(!first){
               cl_file << ",";
            }
            first=false;
            MARKER_CODE(cl_file << " ARG(";)
            RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
            MARKER_CODE(cl_file << ")";)
         }
      }
      cl_file << " )";
      MARKER_CODE(cl_file << "[EndCallExpr]\n";)
      return true;
   }
   virtual bool TraverseDeclRefExpr(DeclRefExpr *ref) {
    MARKER_CODE(cl_file << "[DeclRef]";)
      cl_file << ref->getNameInfo().getAsString();
    MARKER_CODE(cl_file << "[EndDeclRef]";)
      return true;
   }
   virtual bool TraverseCompoundStmt (CompoundStmt * s ){
      stmtIndex*=10;
      for(auto a : s->body()){
         cl_file << " /*" <<  stmtIndex++ << "*/";
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
         cl_file << ";\n";
      }
      stmtIndex/=10;
      return true;
   }
   virtual bool TraverseStmt (Stmt * s ){
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(s);
         if(s && !isa<Expr>(s))
            cl_file << ";/**/\n";
         return true;
   }
   virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr * call) {
      MARKER_CODE(cl_file << "[CXXMemberCallExpr]";)
      cl_file << call->getDirectCallee()->getNameAsString() << " (";
      bool firstArg = true;
      if(!call->isImplicitCXXThis()){
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(call->getImplicitObjectArgument());
//         if(call->getType().getTypePtr()->isAnyPointerType()){
//            cl_file << " ,";
//         }else{
//            cl_file << " , ";
//         }
         firstArg=false;
      }
      for (auto a : call->arguments()) {
         if (!a->isDefaultArgument()) {
            MARKER_CODE(cl_file << " MARG[";)
            if(!firstArg)
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
   //TODO RK - Fix this to be correct for operator implementations (pre/post/multi-args)
   virtual bool TraverseCXXOperatorCallExpr (CXXOperatorCallExpr * op){
      MARKER_CODE(cl_file << " [CXXOp]";)
//      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(op->getCallee());
//      cl_file << " ## ";
//      cl_file<<OverloadedOperatorKind:: op->getDirectCallee()->getNameAsString();
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(op->getArg(0));
      cl_file << toString(op->getOperator());
      if(std::distance(op->arg_begin(), op->arg_end())>1)
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(op->getArg(1));

//      bool isFirst = true;
//      for(auto a : op->arguments()){
//         if(!isFirst)
//            cl_file << " ,";
//         isFirst=false;
//         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
//      }
//      cl_file<< ")";
      MARKER_CODE(cl_file << " [EndCXXOp]";)
      return true;
   }
//   virtual bool TraverseExpr(Expr * e){
//      MARKER_CODE(cl_file << " [Expr]";)
//      MARKER_CODE(cl_file << " [EndExpr]";)
//      return true;
//   }
//   virtual bool TraverseCompoundAssignOperator(CompoundAssignOperator * cap) {
//      MARKER_CODE(cl_file << " [Cap]";)
//      MARKER_CODE(cl_file << " [EndCap]";)
//      return true;
//   }
   virtual bool TraversBinaryOperatorGeneric(BinaryOperator * bop) {
      MARKER_CODE(cl_file << " [BinOp]";)
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(bop->getLHS());
      cl_file << " "<<bop->getOpcodeStr().str()<<" ";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(bop->getRHS());
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
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(uop->getSubExpr());
      cl_file << "++";
      return true;
      }
   virtual bool TraverseUnaryPreInc(UnaryOperator * uop) {
         cl_file << "++";
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(uop->getSubExpr());
         return true;
         }
      virtual bool TraverseMemberExpr(MemberExpr * mem) {
      MARKER_CODE(cl_file << "[MemberExpr]";)
      if (!mem->isImplicitCXXThis()  && !mem->getBase()->isImplicitCXXThis()) {
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(mem->getBase());
         if ( mem->isArrow()) {
            cl_file << " ->";
         } else {
            cl_file << ".";
         }
      }
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseDecl(mem->getMemberDecl());
      MARKER_CODE(cl_file << "[EndMemberExpr]";)
      return true;
   }
   virtual bool TraverseFieldDecl(FieldDecl * fd){
      cl_file << fd->getNameAsString() ;
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
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getCond());
      cl_file << "){\n";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getThen());
      cl_file << " \n}/*End if*/ else{\n";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(i->getElse());
      cl_file << " \n}/*End if-else*/ \n";
      return true;
   }
   virtual bool TraverseForStmt(ForStmt  * fs){
//      fs->dump();
      cl_file << " for (";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getInit());
//      cl_file << toString(fs->getInit());
      cl_file<<";";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getCond());
//      cl_file << toString(fs->getCond());
      cl_file<<";";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getInc());
//      cl_file << toString(fs->getInc());
      cl_file << ") {\n";
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(fs->getBody());
      cl_file << "\n}//End For\n";
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
//   virtual bool _VisitStmt(Stmt * s){
//      static int counter = 0;
//      cl_file << "\nVisiting Stmt :: " << counter++ << ", " << std::distance(s->child_begin(), s->child_end())<<
//            "[ " << ast_utility.get_string(s->getLocStart(), s->getLocEnd()) << "\n";
//      std::string str;
//      raw_string_ostream raw_s(str);
//      s->dump(raw_s);
//      cl_file << raw_s.str();
//      cl_file<< " ]\n";
////         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(s);
//      return true;
//   }
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/

class OpenCLDeviceHandler: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
public:
   OpenCLDeviceHandler(Rewriter &rewriter) :
         rewriter(rewriter) {
   }
   virtual void run(const MatchFinder::MatchResult &Results) {
      CXXMethodDecl* decl = const_cast<CXXMethodDecl*>(Results.Nodes.getNodeAs<CXXMethodDecl>("kernel"));
      OpenCLDeviceVisitor devVisitor(decl->getASTContext(), rewriter);
      devVisitor.GenerateCLCode(decl);
//      OpenCLASTGen astGen(decl->getASTContext(), rewriter);
//      if(decl->getParent()->getNameAsString().compare("InitializeGraph")==0)
//         astGen.GenerateCLCode(decl);
   }

};
/*********************************************************************************************************
 *
 *********************************************************************************************************/

} //End anonymous namespace

#endif /* SRC_PLUGINS_OPENCLCODEGEN_ANALYSIS_H_ */
