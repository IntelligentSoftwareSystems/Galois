/*
 * GaloisCLGen.h
 *
 *  Created on: Jan 27, 2016
 *      Author: rashid
 */



#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include <fstream>

/*
 *  * Matchers
 *   */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/*
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

#ifndef SRC_PLUGINS_OPENCLCODEGEN_GALOISCLGEN_H_
#define SRC_PLUGINS_OPENCLCODEGEN_GALOISCLGEN_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;
using namespace Galois::GAST;

extern ASTUtility ast_utility;

namespace {

//#################################################################################################################//
//#################################################################################################################//
class GaloisCLGenVisitor: public RecursiveASTVisitor<GaloisCLGenVisitor> {
   ASTContext & astContext;
   Rewriter & rewriter;
   std::ofstream & cl_file;
   GaloisApp *app_data;
public:
   explicit GaloisCLGenVisitor(ASTContext & context, Rewriter & R, std::ofstream & f, GaloisApp * a) :
         astContext(context), rewriter(R), cl_file(f), app_data(a) {

   }
   virtual ~GaloisCLGenVisitor() {
   }
   string toString(Stmt * S) {
      string s;
#if 1
      llvm::raw_string_ostream raw_s(s);
      S->dump(raw_s);
      return raw_s.str();
#endif
      return s;
   }
   //##############################
   bool VisitCXXMemberCallExpr(CXXMemberCallExpr * m) {
      llvm::outs() << "MemberCallExpr at " << m->getCalleeDecl()->getAsFunction()->getNameAsString() << "\n";
      llvm::outs() << "Object called with :: " << ast_utility.toString(m->getImplicitObjectArgument()) << "\n";
      Type * objType = const_cast<Type *>(m->getImplicitObjectArgument()->getType()->getCanonicalTypeUnqualified().getTypePtr());
      if(objType->isPointerType()){
         objType=const_cast<Type *>(objType->getPointeeType().getCanonicalType().getTypePtr());
      }
      std::string tname = QualType::getAsString(objType, Qualifiers());
      llvm::outs() << "Typename :: " << tname << " known ?? " << ((app_data->known_types.find(objType) != app_data->known_types.end()) ? "Yes" : "No") << "\n";
      if (tname.compare("LC_CS_Graph")) {
         llvm::outs() << "\nGraph method called!\n";
      }
      for (auto a : m->arguments()) {
         llvm::outs() << "Arg :: " << ast_utility.get_string(a->getExprLoc()) << "\n";
      }
//         llvm::outs()<<ast_utility.toString(m);
      llvm::outs() << "\n==================================================================================\n";
      return true;
   }
};
//#################################################################################################################//
//#################################################################################################################//
class GraphTypedefCollector: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
   std::set<TypedefDecl *> typeDecls;
   CXXRecordDecl * graphClass;
   GaloisApp * app_data;
public:
   GraphTypedefCollector(Rewriter &rewriter, GaloisApp * app, CXXRecordDecl * g) :
         rewriter(rewriter), graphClass(g), app_data(app)  {
   }
   virtual ~GraphTypedefCollector() {

   }
   virtual void run(const MatchFinder::MatchResult &Results) {
//      Results.Nodes.
      TypedefDecl * decl = const_cast<TypedefDecl *>(Results.Nodes.getNodeAs<clang::TypedefDecl>("typedef"));
      typeDecls.insert(decl);
      Type * exposed_type = const_cast<Type *>(decl->getCanonicalDecl()->getTypeForDecl());
      Type * real_type = const_cast<Type *>(decl->getUnderlyingType().getTypePtr());
//      if (exposed_type) //&& decl->getParentFunctionOrMethod()==class_def)
      {
         std::string s = QualType::getAsString(exposed_type, Qualifiers());
         if(decl->getDeclContext() && decl->getDeclContext()==graphClass){
            llvm::outs() << "\nExposedType :: " << QualType::getAsString(exposed_type, Qualifiers()) << ", RealType:: " << QualType::getAsString(real_type, Qualifiers()) << " ";
            llvm::outs() << "\nParent [ " << decl->getDeclContext()->getDeclKindName() << "]";
            llvm::outs() << "\nType for decl :: " << QualType::getAsString(decl->getTypeForDecl(), Qualifiers()) ;
            llvm::outs() << "\nDeclName :: " << decl->getDeclName().getAsString() ;
            for(auto redecl : decl->redecls()){
               llvm::outs() << "\nRedecl :: " << redecl->getDeclName().getAsString() << " , " << QualType::getAsString(redecl->getTypeForDecl(), Qualifiers())
               << ", " << QualType::getAsString(redecl->getUnderlyingType().getTypePtr(), Qualifiers()) << ".";
            }
            llvm::outs() << "\n";

            ast_utility.print_expr(decl->getLocStart(), decl->getLocEnd());
            decl->dump(llvm::outs());
//            llvm::outs()<<ast_utility.toString(decl);
            llvm::outs() << "\n";
            app_data->register_type(real_type, decl->getDeclName().getAsString());
//            app_data->register_type(const_cast<Type*>(decl->getTypeForDecl()),decl->getDeclName().getAsString() );
         }
      }
   }

};

/****************************************************************************
 *****************************************************************************/
class GraphTypeSubstitute: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
   GaloisApp * app_data;
   CXXMethodDecl * op_body;
   int  counter;
   std::set<VarDecl * > decls;
public:
   GraphTypeSubstitute(Rewriter &rewriter, GaloisApp * app, CXXMethodDecl * o) :
         rewriter(rewriter), app_data(app), op_body(o),counter(0)  {
   }
   virtual ~GraphTypeSubstitute() {

   }
   bool within_op_lexical(const SourceLocation & l){
      return ast_utility.get_src_ptr(l) > ast_utility.get_src_ptr(op_body->getLocStart()) &&
            ast_utility.get_src_ptr(l) < ast_utility.get_src_ptr(op_body->getLocEnd());
   }
   virtual void run(const MatchFinder::MatchResult &Results) {
      VarDecl * decl = const_cast<VarDecl *>(Results.Nodes.getNodeAs<clang::VarDecl>("decl"));
      if(!within_op_lexical(decl->getLocStart())){
/*         Type * t = const_cast<Type *> (decl->getType().getCanonicalType().getTypePtr());
         llvm::outs()<< " MatchedVarDecl Skip " << QualType::getAsString(t, Qualifiers()) << " :: ";
         ast_utility.print_expr(decl->getSourceRange()) ;
         llvm::outs() << "\n";*/
         return;
      }
      Type * typeDecl = const_cast<Type *> (decl->getType().getTypePtr());
      if(typeDecl->isPointerType() || typeDecl->isReferenceType())
         typeDecl= const_cast<Type*>(typeDecl->getPointeeType().getTypePtr());

      Type * t = const_cast<Type *> (decl->getType().getCanonicalType().getTypePtr());
      if(t->isPointerType() || t->isReferenceType())
         t= const_cast<Type*>(t->getPointeeType().getTypePtr());
      if(app_data->find_type(t) || app_data->find_type(typeDecl)){
         counter++;

//         char str[1000];
//         sprintf(str, " %s_XXX_%d_", QualType::getAsString(t, Qualifiers()).c_str(), counter++);
//         rewriter.InsertText(decl->getLocStart(),str);

         if(decls.insert(decl).second){
            llvm::outs()<< " MatchedVarDecl \n" << QualType::getAsString(t, Qualifiers()) <<  "\n" << QualType::getAsString(decl->getType().getTypePtr(), Qualifiers())<<"\n";
            ast_utility.print_expr(decl->getSourceRange()) ;
            llvm::outs() << "\n";
         }
      }else{
//         llvm::outs()<< " MatchedVarDecl NOTFOUND" << QualType::getAsString(t, Qualifiers()) << " :: ";
//                  ast_utility.print_expr(decl->getSourceRange()) ;
//                  llvm::outs() << "\n";
      }
   }
   void doRewrite(){
      counter=0;
      for(auto decl : decls){
         Type * t = const_cast<Type *> (decl->getType().getTypePtr());
         if(t->isPointerType() || t->isReferenceType())
            t= const_cast<Type*>(t->getPointeeType().getTypePtr());
         char str[1000];
         llvm::outs() << "Declaration replaced :: " << QualType::getAsString(t, Qualifiers()) << " WITH " << app_data->replace_type(t).c_str()<<"\n";
         SourceRange range(decl->getTypeSourceInfo()->getTypeLoc().getLocStart(), decl->getTypeSourceInfo()->getTypeLoc().getLocEnd());
         string s = ast_utility.get_string(range);
         sprintf(str, "%s /*%d-%s, %s*/", app_data->replace_type(t).c_str() ,counter++, s.c_str(), ast_utility.get_string(decl->getSourceRange()).c_str());
//         rewriter.InsertText(decl->getLocStart(),str);
//         if(counter==3 || counter==4)
            rewriter.ReplaceText(range,str);
//         decls.insert(decl);
      }
   }

};

/****************************************************************************
 *****************************************************************************/

}//End anonymous namespace
#endif /* SRC_PLUGINS_OPENCLCODEGEN_GALOISCLGEN_H_ */
