/*
 * syncHandler.h
 *
 *  Created on: Apr 26, 2016
 *      Author: rashid
 */
//@author Rashid Kaleem <rashid.kaleem@gmail.com>
#include "utils.h"
#include "gAST.h"

#ifndef SRC_PLUGINS_CLIFY_SYNCHANDLER_H_
#define SRC_PLUGINS_CLIFY_SYNCHANDLER_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

class SyncHandler: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
   ASTContext & ctx;
   galois::GAST::GaloisApp & app;
public:
   SyncHandler(Rewriter &rewriter, ASTContext & c, galois::GAST::GaloisApp & a) :
         rewriter(rewriter), ctx(c), app(a){
   }
   //TODO - RK - Clean up graph generation code.
   ~SyncHandler(){

   }
   virtual void run(const MatchFinder::MatchResult &Results) {
         CXXMethodDecl * decl = const_cast<CXXMethodDecl *> (Results.Nodes.getNodeAs<CXXMethodDecl>("methodDecl"));
         if (decl) {
            llvm::outs() << " Template arguments for " << decl->getNameAsString()<< ", " << decl->isFunctionTemplateSpecialization() <<
                  ", " << decl->isFunctionOrFunctionTemplate() << " , "<< decl->isTemplateInstantiation() << "\n";
            llvm::outs() << " Template arguments to sync_pull " << decl->getNumTemplateParameterLists() << "\n";
            for(size_t i=0; i<decl->getTemplateSpecializationArgs()->size(); i++){
               QualType qt = decl->getTemplateSpecializationArgs()->get(i).getAsType();
               genSync(qt.getTypePtr()->getAsCXXRecordDecl());
               llvm::outs()<< i <<", " << qt.getAsString()<< "\n";
            }
         }
   }
   void genSync(CXXRecordDecl * syncer){
      llvm::outs() << "==============================================================\n";
      llvm::outs() << "Generating sync routine for " << syncer->getNameAsString() << "\n";
      for(auto m : syncer->methods()){
         llvm::outs() << "Method :: " << m->getNameAsString( ) << " \n";
         for(auto p : m->parameters()){
            llvm::outs() << " Parameter :: " << p->getNameAsString() << "\n";
         }
      }
      return;
   }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/



#endif /* SRC_PLUGINS_CLIFY_SYNCHANDLER_H_ */
