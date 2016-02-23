/*
 * Generation.h
 *
 *  Created on: Dec 8, 2015
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
#include "GaloisCLGen.h"
#ifndef SRC_PLUGINS_OPENCLCODEGEN_GENERATION_H_
#define SRC_PLUGINS_OPENCLCODEGEN_GENERATION_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;

#define MARKER_CODE(X) {X}

namespace {

//#################################################################################################################//
//#################################################################################################################//
class OpenCLDeviceVisitor: public RecursiveASTVisitor<OpenCLDeviceVisitor> {
   std::vector<FieldDecl *> * fields;
   ASTContext * astContext;
   Rewriter & rewriter;
   std::ofstream & cl_file;
   clang::Stmt * cl_impl;
   std::map<string, string> symbol_map;
   int counter;
public:
   explicit OpenCLDeviceVisitor(ASTContext * context, Rewriter & R, std::ofstream & f) :
         fields(nullptr), astContext(context), rewriter(R), cl_file(f), cl_impl(nullptr), counter(0) {

   }
   virtual ~OpenCLDeviceVisitor() {
//      rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(llvm::outs());

   }
   /*
    * Entry point for device code generator. This is supposed to go over the
    * method declaration.
    * */
   void process_method(std::vector<FieldDecl *> * f, CXXMethodDecl * method) {
      fields = f;
      symbol_map.clear();
      for (auto field : *fields) {
         symbol_map[field->getNameAsString()] = field->getNameAsString();
      }
      RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(method->getBody());
      std::string str;
      raw_string_ostream s(str);
      rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(s);
      cl_file<< "\n/*Rewritten text*/\n"<<str<<"\n/*End-Rewritten text*/\n";
//      cl_file << "/****************************************/\n" << method->getNameAsString() << " \n";
//      for (auto sentry : this->symbol_map) {
//         cl_file << (sentry).first << " mappedTo " << (sentry).second << "\n";
//      }

//      cl_file << "END SYMBOL TABLE \n";
//      cl_file << rewriter.getRewrittenText(method->getSourceRange());
   }
   virtual bool TraverseVarDecl(VarDecl * d) {
      string var_name = d->getNameAsString();
      MARKER_CODE(cl_file << " [@VarDecl]";)
      if (symbol_map.find(var_name) != symbol_map.end()) {
         llvm::outs() << "ERROR Duplicate symbols? '" << var_name << ", " << d->getDeclName().getAsString() << ", "
               << QualType::getAsString(d->getType()->getCanonicalTypeUnqualified().split()) << "'\n";
         cl_file << "UNDECLARED(" << var_name << " )";
      } else {
         cl_file << QualType::getAsString(d->getType().split()) << " " << var_name << "";
      }
      if (d->hasInit()) {
//         MARKER_CODE(cl_file << " /*INIT*/ ";)
         cl_file<<" = ";
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(d->getInit());
      }
      cl_file << ";\n";
      MARKER_CODE(cl_file << " /*[END-VarDecl]*/";)
      return true;
   }
   virtual bool TraverseDeclRefExpr(DeclRefExpr * dre){
      static int counter = 0;
////      cl_file << dre->getNameInfo().getAsString() << "\n";
      char s [128];
      sprintf(s, "/*[%d]*/\n",counter++);
//      cl_file<< s<< " ";
      cl_file<<dre->getNameInfo().getAsString() << "";
      return true;
   }
   virtual bool TraverseBinaryOperator(BinaryOperator * bo){
      cl_file<<bo->getOpcodeStr().str();
      return true;
   }
   virtual bool TraverseMemberExpr(MemberExpr * me){
      if (me->isImplicitCXXThis()){

      }else{
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(me->getBase());
         if(me->isArrow()){
            cl_file << " @->";
         }
         else{
            cl_file<<"@.";
         }
      }
         cl_file<<me->getMemberNameInfo().getAsString();
      return true;
   }
   virtual bool TraverseCXXMemberCallExpr(CXXMemberCallExpr * mce){
      if(mce->isImplicitCXXThis()){
      }
      else{
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(mce->getCallee());
         cl_file<<"#.";
      }
      cl_file << mce->getMethodDecl()->getNameAsString() << "(";
      for(auto a : mce->arguments()){
         RecursiveASTVisitor<OpenCLDeviceVisitor>::TraverseStmt(a);
      }
      cl_file << ")";
      return true;
   }

};
//End class OpenCLDeviceVisitor
//#################################################################################################################//
//#################################################################################################################//
/*
 * This class should visit the AST to scan for operator implementations. For each
 * operator, it calls the OpenCLDeviceVisitor to generate device specific code.
 * This requires a list of arguments to be generated for each member of the class.
 * A mapping from CPU-Members to CL-Members is created for the operator object, and passed
 * to the CL code generator.
 * */
class OpenCLOperatorVisitor: public RecursiveASTVisitor<OpenCLOperatorVisitor> {
private:
   int declDepthCounter;
   ASTContext* astContext;
   Rewriter &rewriter;
   std::ofstream header_file;
   std::ofstream cl_file;
   OpenCLDeviceVisitor device_visitor;
   std::vector<CXXRecordDecl *> op_classes;
   std::vector<std::vector<FieldDecl *> *> op_class_fields;
   CXXRecordDecl * operator_class;bool graphFound;
   std::map<CXXRecordDecl*, std::vector<Type *> *> &graphDecl;

public:
   explicit OpenCLOperatorVisitor(ASTContext *context, Rewriter &R, std::map<CXXRecordDecl*, std::vector<Type *> *> &_gd) :
         declDepthCounter(0), astContext(context), rewriter(R), device_visitor(context, R, cl_file), operator_class(nullptr), graphFound(false), graphDecl(_gd) {
      header_file.open("cl_header.h");
      cl_file.open("cl_implementation.cl");

      for (auto g_entry : graphDecl) {
         CXXRecordDecl * decl = g_entry.first;
         std::vector<Type *> *tdecls = g_entry.second;
         llvm::outs() << "Printing typedeclarations for graph :: " << decl->getNameAsString() << "\n";
         for (Type * t : *tdecls) {
            llvm::outs() << "Typdecl:: " << QualType::getAsString(t->getCanonicalTypeInternal().split()) << "\n";
         }
      }
   }
   //Destructor
   virtual ~OpenCLOperatorVisitor() {
      header_file.close();
      cl_file.close();
      for (auto c : op_classes) {
         llvm::outs() << "Operator class :: " << c->getNameAsString() << "\n";
      }
   }
   /*******************************************************************
    *
    *******************************************************************/
   string convertName(const string & str) {
      string res("HGCL_");
      return res.append(str);
   }
   /*******************************************************************
    *
    *******************************************************************/

   virtual bool createDeviceImplementation(CXXMethodDecl * Decl) {
      if (!Decl->getNameAsString().compare("operator()"))
         return true;
      return false;
   }
   /*******************************************************************
    *
    *******************************************************************/

   void write_code() {
      string class_name = operator_class->getNameAsString();
      header_file << " struct " << class_name << " { \n";
      assert(op_class_fields.size() >= 1);
      for (auto field : *op_class_fields[op_class_fields.size() - 1]) {
         header_file << QualType::getAsString(field->getType().split()) << "  " << field->getNameAsString() << ";\n";
      }
      header_file << " } ; // End struct  " << class_name << " \n";
   }
   /*******************************************************************
    *
    *******************************************************************/

   bool generate_operator_code(CXXRecordDecl * Decl) {
      size_t num_fields = std::distance(Decl->field_begin(), Decl->field_end());
      llvm::outs() << "About to traverse CXXRecord :: " << Decl->getNameAsString() << " " << num_fields << "\n";
      this->TraverseCXXRecordDecl(Decl);
      this->write_code();
      op_class_fields.pop_back();
      llvm::outs() << "Done traversing CXXRecord :: " << Decl->getNameAsString() << "\n";
      operator_class = nullptr;
      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/

   virtual bool TraverseCXXRecordDecl(CXXRecordDecl * Decl) {
      if (operator_class != nullptr) {
         return true;
      } // For now we avoid recursing into structures.
      operator_class = Decl;
      op_class_fields.push_back(new std::vector<FieldDecl *>());
      RecursiveASTVisitor<OpenCLOperatorVisitor>::TraverseCXXRecordDecl(Decl);
      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/

   virtual bool TraverseCXXMethodDecl(CXXMethodDecl * Decl) {
      if (createDeviceImplementation(Decl)) {
         op_classes.push_back(Decl->getParent());

      } else {

      }
      //Call base

      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/

   virtual bool VisitVarDecl(VarDecl * Decl) {
      std::string tname = QualType::getAsString(Decl->getType().split());
      llvm::outs() << "Visiting VarDecl :: " << tname << " " << Decl->getNameAsString() << "\n";
      header_file << QualType::getAsString(Decl->getType().split()) << " " << Decl->getNameAsString() << "/*VarDecl*/\n";
      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/
   virtual bool VisitParmVarDecl(ParmVarDecl * Decl) {
      llvm::outs() << "Visiting ParmVarDecl :: " << Decl->getNameAsString() << "\n";
      header_file << QualType::getAsString(Decl->getType().split()) << " " << Decl->getNameAsString() << "/*ParmVarDecl*/\n";
      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/
   virtual bool VisitDeclRefExpr(DeclRefExpr* Decl) {
//      llvm::outs() << "Visiting ParmVarDecl :: " << Decl->getNameAsString() << "\n";
      header_file << QualType::getAsString(Decl->getType().split()) << " " << Decl->getDecl()->getNameAsString() << "/*DeclRefExpr*/\n";
      return true;
   }
   /*******************************************************************
    *
    *******************************************************************/

   virtual bool VisitFieldDecl(FieldDecl * Decl) {

      std::string tname = QualType::getAsString(Decl->getType()->getCanonicalTypeInternal().split());
//      CanQualType canQualType = Decl->getType()->getCanonicalTypeUnqualified();
      Type * field_type = const_cast<Type *>(Decl->getType()->getCanonicalTypeUnqualified()->getTypePtr());
      if (field_type->isPointerType()) {
         field_type = const_cast<Type*>(field_type->getPointeeType()->getCanonicalTypeUnqualified()->getTypePtr());
         header_file << "(PointerType, " << field_type->getTypeClassName() << ", ";
         const RecordType * decl = field_type->getAs<RecordType>();
         header_file << " DeclarationName :: " << decl->getDecl()->getNameAsString() << ")\n";

      }
      const RecordType * field_decl = field_type->getAs<RecordType>();
//      header_file << " \n==============================================@@@@@@@@@@@@@=============================================\n";
//      header_file << toString(Decl);
//      header_file << " \n==============================================@@@@@@@@@@@@@=============================================\n";

      for (auto gentry : graphDecl) {
         CXXRecordDecl * decl = gentry.first;
         bool canTypeSame = Decl->getType()->getCanonicalTypeUnqualified() == decl->getTypeForDecl()->getCanonicalTypeUnqualified();
//         header_file << " Comparing :: "<< tname << " and " << QualType::getAsString(decl->getTypeForDecl());
         header_file << " Comparing :: " << decl->getCanonicalDecl()->getDeclName().getAsString() << " and " << Decl->getDeclName().getAsString() << ", " << tname << " ,  "
               << canTypeSame << "\n";

         if (decl->getTypeForDecl() == field_decl) {
            header_file << " \n@1 MATCHED_TYPE_PTRS:::::::::::::: !!!!!! " << Decl->getNameAsString() << "\n";
         }
         if (Decl->getType()->getCanonicalTypeUnqualified()->getTypePtr() == field_type) { // decl->getTypeForDecl()->getCanonicalTypeUnqualified()->getTypePtr()){
            header_file << " \n@2 MATCHED_TYPE_PTRS:::::::::::::: !!!!!! " << Decl->getNameAsString() << "\n";
         }

         if (Decl->getDeclName() == decl->getDeclName()) { // decl->getTypeForDecl()->getCanonicalTypeInternal()==Decl->getType()->getCanonicalTypeInternal()){
            header_file << " \n@3 MATCHED :::::::::::::: !!!!!! " << tname << "\n";
         }
      }
      if (tname.find("Graph") != tname.npos && !graphFound) {
//         header_file << " GraphType :: " << toString(Decl) << "\n";
         graphFound = true;
//         GaloisGraphVisitor ggv(this->astContext, this->rewriter, this->header_file);
//         ggv.TraverseDecl(Decl->getMostRecentDecl());
         const Type * graphType = Decl->getTypeSourceInfo()->getType().getSplitUnqualifiedType().asPair().first;
         CXXRecordDecl * def = Decl->getType()->getUnqualifiedDesugaredType()->getAsCXXRecordDecl();
         header_file << " $$$$$$$$$$$GRAPH?$$$$$$$$$$$$$ " << graphType->getTypeClassName() << "\n";
         if (def)
            header_file << " $$$$$$$$$$$GRAPH?$$$$$$$$$$$$$ " << def->getNameAsString() << "\n";
         else
            header_file << " $$$$$$$$$$$GRAPH CLASS $$$$$$$$$$$$" << Decl->getType()->getTypeClassName() << "\n";
//         ggv.TraverseDecl(Decl->getType()->getAsCXXRecordDecl());
      }

//      header_file << "####Visiting FieldDecl :: " << Decl->getNameAsString() << "\n";
      llvm::outs() << "####Visiting FieldDecl :: " << Decl->getNameAsString() << "\n";
//      header_file << QualType::getAsString(Decl->getType().split() )  << "  " << Decl->getNameAsString() << ";\n";
      op_class_fields[op_class_fields.size() - 1]->push_back(Decl);
      return true;
   }
};
//End class OpenCLOperatorVisitor
//#################################################################################################################//
//#################################################################################################################//

}//End anonymous namespace

#endif /* SRC_PLUGINS_OPENCLCODEGEN_GENERATION_H_ */
