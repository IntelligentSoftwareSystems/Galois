/*
 * ClangUtils.h
 *
 *  Created on: Jan 27, 2016
 *  @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */

#ifndef SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_
#define SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include <fstream>
#include <time.h>

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
using namespace std;

struct ASTUtility{
   Rewriter * rewriter;
   ASTUtility ():rewriter(nullptr){
   }
   void init(Rewriter * r){
      rewriter = r;
   }
   std::string toString(Stmt * S) {
      std::string s;
      llvm::raw_string_ostream raw_s(s);
      S->dump(raw_s);
      return raw_s.str();
   }
   char * get_src_ptr(const clang::SourceLocation & s) {
      return const_cast<char *>(rewriter->getSourceMgr().getCharacterData(s));
   }
   void print_expr(const clang::SourceLocation & b, const clang::SourceLocation & e) {
      for (char * p = get_src_ptr(b); p <= get_src_ptr(e); ++p) {
         llvm::outs() << *p;
      }
   }
   std::string get_string(const clang::SourceLocation & b, const clang::SourceLocation & e) {
      char * b_ptr = get_src_ptr(b);
      char * e_ptr = get_src_ptr(e);
      std::string s(b_ptr, std::distance(b_ptr, e_ptr)+1);
      return s;
   }
   std::string get_string(const clang::SourceRange & e) {
      char * b_ptr = get_src_ptr(e.getBegin());
      char * e_ptr = get_src_ptr(e.getEnd());
      std::string s(b_ptr, std::distance(b_ptr, e_ptr)+1);
      return s;
   }
   void print_expr(const clang::SourceRange & c) {
      for (char * p = get_src_ptr(c.getBegin()); p <= get_src_ptr(c.getEnd()); ++p) {
         llvm::outs() << *p;
      }
   }
   static std::string get_timestamp() {
         time_t now = time(0);
         struct tm tstruct;
         char buf[80];
         tstruct = *localtime(&now);
         // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
         // for more information about date/time format
         strftime(buf, sizeof(buf), "%Y-%m-%d - %X", &tstruct);

         return buf;
      }
};

/*************************************************************************************************
 *
 **************************************************************************************************/
struct OpenCLConversionDB{
   static const char * get_cl_implementation(FunctionDecl *  d){
//      llvm::outs()<<"About to process call :: " << d->getNameAsString() << "\n";
      const string fname = d->getNameAsString();
      if(d->isOverloadedOperator()){
         const char  * op_name = getOperatorSpelling( d->getOverloadedOperator());
         if(strcmp(op_name, "operator int")==0){
            return "CAST_OP";
         }
         return getOperatorSpelling( d->getOverloadedOperator());
      }
      if(d->getNumParams()==0){
         if(fname=="min"){
            return "INT_MIN";
         }else if (fname=="max"){
            return "INT_MAX";
         }else if (fname == "operator int"){
            return "";
         }
      }else if (d->getNumParams()==1){
         if(fname=="compare_exchange_strong"){
            return "atomic_cmpxchg";
         }else if (fname == "edge_begin"){
            return "edge_begin";
         }else if (fname == "edge_end"){
            return "edge_end";
         }else if (fname == "getEdgeDst"){
            return "getEdgeDst";
         }
      }else if (d->getNumParams()==2){
         if(fname=="atomicMin"){
            return "atomic_min";
         }else if(fname=="compare_exchange_strong"){
            return "atomic_cmpxchg";
         }else if(fname=="getData"){
            return "getData";
         }else if (fname == "getEdgeData"){
            //TODO RK - fix hack with edgeData
            return "*getEdgeData";
         }
      }else if (d->getNumParams()==3){
         if(fname=="compare_exchange_strong"){
            return "atomic_cmpxchg";
         }
      }
      {
         string s = fname;
         s+=" /*UNINTERPRETED-";
         s+=d->getNumParams();
         s+="*/";
         return s.c_str();
      }
   }
   static string type_convert(const QualType & qt){
           //TODO RK - Filter primitive types.
           llvm::outs() << " TypeConversion :: " << qt.getAsString() << "\n";
           if(qt.getAsString()=="int"){
              return "int ";
           }else if(qt.getAsString()=="float"){
              return "float";
           }else if(qt.getAsString()=="double"){
              return "double";
           }else if(qt.getAsString()=="char"){
              return "char";
           }
           //Atomic stripper
           if(qt.getAsString().find ("std::atomic")!=string::npos){
              return qt.getAsString().substr(qt.getAsString().rfind("<")+1,qt.getAsString().rfind(">")-qt.getAsString().rfind("<")-1);
           }
           //Iterator stripper
           if (qt.getAsString().find ("boost::iterators::counting_iterator")!=string::npos){
              return "edge_iterator ";
           }
           if (qt.getAsString().find("GNode")!=string::npos){
              return "node_iterator";
           }
           //Wrapper for nodedata.
           if(qt.getAsString().find("NodeData")!=string::npos){
              return "__global NodeData *";
           }
           if(qt.getAsString().find("Graph")!=string::npos){
              return "__global Graph *";
           }
           return qt.getAsString();
   }
};
#endif /* SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_ */
