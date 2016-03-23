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

#define MARKER_CODE(X) {}
#define _MARKER_CODE(X) {X}

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceRewriter: public RecursiveASTVisitor<OpenCLDeviceRewriter> {
   ASTContext & astContext;
   Rewriter & rewriter;
public:
   explicit OpenCLDeviceRewriter(ASTContext & context, Rewriter & R) :
         astContext(context), rewriter(R){
   }
   virtual ~OpenCLDeviceRewriter() {
   }
   void GenerateCLCode(CXXMethodDecl * method) {
      this->CodeGenMethod(method);
   }
   virtual bool CodeGenMethod(CXXMethodDecl * method) {
      std::string kernel_name = "__kernel void ";
      if(method->isConst()){
         QualType q = method->getType();
         q.removeLocalConst();
         method->setType(q);
      }
      kernel_name += method->getParent()->getNameAsString();
      kernel_name += "";
      string param_list = "";
      bool needComma=false;
      for (auto p : method->getParent()->fields()) {
         if(needComma){
            param_list+=",";
         }
         needComma=true;
//         param_list+= "__global ";
         param_list+= OpenCLConversionDB::type_convert(p->getType());
//         param_list+= " * ";
         param_list+= p->getNameAsString();
      }
      if(needComma){
         param_list+=",";
      }
      param_list += " uint num_items ";

      string numNodeCondition = "{\nconst uint ";
      numNodeCondition+= method->getParamDecl(0)->getNameAsString();
      numNodeCondition+= " = get_global_id(0);\n if (";
      numNodeCondition+= method->getParamDecl(0)->getNameAsString();
      numNodeCondition+= " < num_items) {\n";
      rewriter.ReplaceText(method->getNameInfo().getSourceRange(),kernel_name.c_str());
      SourceRange s(method->getParamDecl(0)->getLocEnd(), method->getBody()->getLocStart());
//      rewriter.ReplaceText(s, " )/*RemoveConst*/{");
      param_list += " ) \n";
      rewriter.ReplaceText(SourceRange(method->getParamDecl(0)->getSourceRange().getBegin(), method->getBody()->getLocStart()),param_list );
      rewriter.RemoveText(method->getReturnTypeSourceRange());
      rewriter.InsertText(method->getBody()->getLocStart(), numNodeCondition);
      rewriter.InsertText(method->getBodyRBrace(), "}//End if num_items\n");
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(method->getBody());
      return true;
   }
   virtual bool TraverseVarDecl(VarDecl * var) {
         MARKER_CODE(llvm::outs() << "[VARDEL]" << var->getNameAsString() << "\n";)
         SourceRange sr(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(), var->getTypeSourceInfo()->getTypeLoc().getEndLoc());
         rewriter.ReplaceText(sr, OpenCLConversionDB::type_convert(var->getType()));

         if (var->hasInit()) {
            RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(var->getInit());
         }
         MARKER_CODE(llvm::outs() << "[END-VARDEL]";)
         return true;
      }
   virtual bool TraverseCallExpr(CallExpr * call) {
      MARKER_CODE(cl_file << "[CallExpr]";)
      llvm::outs() << " CallExpression [" << call->getDirectCallee()->getCanonicalDecl()->getNameAsString() << " ]\n";
      string fnameReplacement = OpenCLConversionDB::get_cl_implementation(call->getDirectCallee());
      rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(), call->getCallee()->getLocEnd()), fnameReplacement);
      if(fnameReplacement.find("atomic")!=string::npos){
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
   virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr * call) {
         MARKER_CODE(cl_file << "[CXXMemberCallExpr]";)
            llvm::outs() << "CXXMemberCallExpression [" << call->getDirectCallee()->getCanonicalDecl()->getNameAsString() << " ]\n";
         std::string fnameReplacement = OpenCLConversionDB::get_cl_implementation(call->getDirectCallee());
         if(fnameReplacement==""){
            RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(call->getImplicitObjectArgument());
            string objArg = rewriter.getRewrittenText(SourceRange(call->getImplicitObjectArgument()->getLocStart(), call->getImplicitObjectArgument()->getLocEnd()));
            rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(), call->getCallee()->getLocEnd()), objArg);
         }
         else{
         std::string objArg = "";
         llvm::outs() << " Replacing function :: " << fnameReplacement << "\n";
         if(fnameReplacement.find("atomic")!=string::npos){
            objArg+= "&";
         }
         if(!call->isImplicitCXXThis()){
            RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(call->getImplicitObjectArgument());
            objArg += rewriter.getRewrittenText(SourceRange(call->getImplicitObjectArgument()->getLocStart(), call->getImplicitObjectArgument()->getLocEnd()));
//            SourceRange sr(call->getLocStart(),  call->getCallee()->getLocStart());
//            rewriter.RemoveText(sr);
         }

         if(call->getNumArgs()==0){
            rewriter.InsertTextAfter(call->getRParenLoc(), objArg);
         }else{
            objArg+=", ";
            rewriter.InsertTextBefore(call->getArg(0)->getExprLoc(), objArg);
         }
         for (auto a : call->arguments()) {
               RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(a);
         }
         rewriter.ReplaceText(SourceRange(call->getCallee()->getLocStart(), call->getCallee()->getLocEnd()), fnameReplacement);
         }
         MARKER_CODE(cl_file << "[EndCXXMemberCallExpr]\n";)
         return true;
      }
   virtual bool TraverseDeclStmt(DeclStmt * ds){
      MARKER_CODE(cl_file << "[DECLStmt]";)
      auto curr_decl = ds->decl_begin();
      RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(*curr_decl);
      curr_decl++;
      while(curr_decl != ds->decl_end()){
         VarDecl * decl = dyn_cast<VarDecl>(*curr_decl);
         if(decl){
            if (decl->hasInit()){
                  RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseStmt(decl->getInit());
            }
         }
         curr_decl++;
      }
      MARKER_CODE(cl_file << "[EndDeclStmt]";)
      return true;
   }
   virtual bool TraverseDeclRefExpr(DeclRefExpr *ref) {
       MARKER_CODE(cl_file << "[DeclRef]";)
//         cl_file << ref->getNameInfo().getAsString();

       MARKER_CODE(cl_file << "[EndDeclRef]";)
         return true;
      }
   virtual bool TraverseMemberExpr(MemberExpr * mem) {
         MARKER_CODE(cl_file << "[MemberExpr]";)
      if(OpenCLConversionDB::type_convert(mem->getBase()->getType()) == "__global NodeData *"){
               rewriter.ReplaceText(mem->getOperatorLoc(), 1, "->");
            }
         RecursiveASTVisitor<OpenCLDeviceRewriter>::TraverseDecl(mem->getMemberDecl());
         MARKER_CODE(cl_file << "[EndMemberExpr]";)
         return true;
      }
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class OpenCLDeviceHandler: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
   Galois::GAST::GaloisApp & app_data;

public:
   OpenCLDeviceHandler(Rewriter &rewriter, Galois::GAST::GaloisApp & a) :
         rewriter(rewriter), app_data(a) {
   }
   virtual void run(const MatchFinder::MatchResult &Results) {
      CXXMethodDecl* decl = const_cast<CXXMethodDecl*>(Results.Nodes.getNodeAs<CXXMethodDecl>("kernel"));
      llvm::outs() << "Processing Method :: " << decl->getParent()->getNameAsString() << "\n";
      // TODO Fix the type system to use a central type repository for graph accesses.
      OpenCLDeviceRewriter devVisitor(decl->getASTContext(), rewriter);
      devVisitor.GenerateCLCode(decl);
      char filename [1024];
      sprintf(filename, "%s/%s.cl", app_data.dir, decl->getParent()->getNameAsString().c_str());
      ofstream impl_file(filename);
      impl_file << " // Implementation for kernel in " << decl->getParent()->getNameAsString() << "\n";
      impl_file << " // Generated at " << ast_utility.get_timestamp() << "\n";
      impl_file << " #include \"app_header.h\"\n\n";
      impl_file << rewriter.getRewrittenText(decl->getSourceRange()) << "\n";
      impl_file.close();
   }

};




#endif /* SRC_PLUGINS_CLIFY_KERNEL_GEN_H_ */
