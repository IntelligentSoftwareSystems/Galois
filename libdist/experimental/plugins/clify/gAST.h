/*
 * GaloisAST.h
 *
 *  Created on: Jan 21, 2016
 * @author Rashid Kaleem <rashid.kaleem@gmail.com>
 */
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "utils.h"

//#include "GaloisCLGen.h"
#ifndef SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_
#define SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_

extern ASTUtility ast_utility;

namespace galois {
namespace GAST {
/****************************************************************************
 *****************************************************************************/
using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

//Forward declarations:
struct GaloisApp;
/****************************************************************************
 *****************************************************************************/

class TypeCollector: public RecursiveASTVisitor<TypeCollector> {
public:
   std::set<Type *> types_used;
   TypeCollector() {

   }
   virtual ~TypeCollector() {

   }
   virtual bool TraverseDecl(Decl * decl) {
      RecursiveASTVisitor<TypeCollector>::TraverseDecl(decl);
      return true;
   }
   virtual bool VisitVarDecl(VarDecl * decl) {
      types_used.insert(const_cast<Type*>(decl->getType().getTypePtr()));
      return true;
   }

};
/****************************************************************************
 *****************************************************************************/
struct GaloisKernel {
   Rewriter & rewriter;
   CXXRecordDecl * kernel;
   CXXMethodDecl * kernel_body;
   GaloisApp * app_data;
   GaloisKernel(Rewriter & r, CXXRecordDecl * d, GaloisApp * a) :
         rewriter(r), kernel(d), kernel_body(nullptr), app_data(a) {
      llvm::outs() << "Creating GaloisKernel :: " << kernel->getNameAsString() << "\n";
      int method_counter = 0;
      for (auto m : kernel->methods()) {
         llvm::outs() << method_counter << "] Is overloaded ? " << (m->isOverloadedOperator() ? "Yes" : "No") << " ";
         std::string m_name = m->getNameAsString();
         if (m_name.find("operator()") != m_name.npos) {
            llvm::outs() << "Found operator :: " << m_name << "";
            if (!kernel_body) {
               init_kernel_body(static_cast<CXXMethodDecl*>(m));
            }
         } else {
            llvm::outs() << "Found method :: " << m_name << "";
         }
         llvm::outs() << "\n";
         method_counter++;
      }
      if (d->isLambda()) {
         llvm::outs() << "Lambda call operator :: " << d->getLambdaCallOperator()->getNameAsString() << "\n";
      }
      //////////Scan for types used.
      {
         llvm::outs() << "Scanning for types used:: \n";
         TypeCollector tc;
         tc.TraverseDecl(kernel);
         for (Type* t : tc.types_used) {
            llvm::outs() << " Type : " << QualType::getAsString(t, Qualifiers()) << "\n\t" << QualType::getAsString(t->getCanonicalTypeInternal().getTypePtr(), Qualifiers())
                  << "\n";
         }
         llvm::outs() << " Done scanning for types.\n";
      }
      //####################################
      rewriter.InsertText(d->method_begin()->getLocStart(), get_impl_string().c_str());
//      rewriter.InsertTextAfter(kernel_body->getLocEnd(), "#endif\n");
//      rewriter.InsertTextBefore(kernel_body->getLocStart(), "#if 0\n");
      rewriter.ReplaceText(SourceRange(kernel_body->getLocStart(), kernel_body->getLocEnd()), "");
      //####################################
   }
   string get_copy_to_host_impl() {
      std::string ret = "\nvoid copy_to_host(){\n";
      for (auto f : this->kernel->fields()) {
         ret += f->getNameAsString();
         ret += ".copy_to_host();\n";
      }
      ret += "//COPY_EACH_MEMBER_TO_HOST_HERE\n";
      ret += "}\n";
      return ret;
   }
   string get_copy_to_device_impl() {
      std::string ret = "\nvoid copy_to_device(){\n";
      for (auto f : this->kernel->fields()) {
         ret += f->getNameAsString();
         if (f->getType().getTypePtr()->isAnyPointerType())
            ret += "->";
         else
            ret += ".";
         ret += "copy_to_device();\n";
      }
      ret += "//COPY_EACH_MEMBER_TO_DEVICE_HERE\n";
      ret += "}\n";
      return ret;
   }
   string get_call_string() {
      return "\ncl_call_wrapper()\n;";
   }
   string get_impl_string() {
      std::string ret_string = "\nCL_Kernel * get_kernel(size_t num_items)const {\n";
      ret_string += " static CL_Kernel kernel(getCLContext()->get_default_device(),\"";
      ret_string += kernel->getNameAsString();
      ret_string += ".cl\", \"";
      ret_string += kernel->getNameAsString();
      ret_string += "\", false);\n";
      int i = 0;
      char s[1024];
//      std::map<Decl*, CLVar*> varMapping;
      for (auto m : this->kernel->fields()) {
//         varMapping[m] = new CLVar(true, m);
         string ref;
         if (m->getType().getTypePtr()->isAnyPointerType()) {
            ref = "->";
            sprintf(s, "kernel.set_arg(%d,sizeof(cl_mem),&(%s%sdevice_ptr()));\n", i, m->getNameAsString().c_str(), ref.c_str());
         } else {
            if (m->getType()->isScalarType()) {
               sprintf(s, "kernel.set_arg(%d,sizeof(%s),&(%s));\n", i, m->getType().getAsString().c_str(), m->getNameAsString().c_str());
            } else {
               ref = ".";
               sprintf(s, "kernel.set_arg(%d,sizeof(%s),&(%s%sdevice_ptr()));\n", i, m->getType().getAsString().c_str(), m->getNameAsString().c_str(), ref.c_str());
            }
         }
         ret_string += s;
         i++;
      }
      for (auto v : this->kernel_body->getLexicalParent()->decls()) {
//         llvm::outs() << " DECL :: " << v->getDeclKindName() << " \n";
         if (strcmp(v->getDeclKindName(), "Var") == 0) {
            string ref;
            VarDecl * vd = (VarDecl*) v;
            if (vd->getType().getTypePtr()->isAnyPointerType()) {
               ref = "->";
               sprintf(s, "kernel.set_arg(%d,sizeof(cl_mem),&(%s%sdevice_ptr()));\n", i, vd->getNameAsString().c_str(), ref.c_str());
            } else {
               if (vd->getType()->isScalarType()) {
                  sprintf(s, "kernel.set_arg(%d,sizeof(%s),&(%s));\n", i, vd->getType().getAsString().c_str(), vd->getNameAsString().c_str());
               } else {
                  ref = ".";
                  sprintf(s, "kernel.set_arg(%d,sizeof(cl_mem),&(%s%sdevice_ptr()));\n", i, vd->getNameAsString().c_str(), ref.c_str());
               }
            }
            ret_string += s;
            i++;
         }

      }
      sprintf(s, "kernel.set_arg(%d,sizeof(cl_uint),&num_items);\n", i);
      ret_string += s;
      ret_string += " return &kernel;\n";
      ret_string += "}\n";
      return ret_string;
   }
   void init_kernel_body(CXXMethodDecl * m) {
      //TODO RK : should check for double initialization!
      kernel_body = m;
      std::string name = m->getNameAsString();
      llvm::outs() << "Initializing kernel_body :: " << name << "\n";
      for (auto p : kernel_body->parameters()) {
         llvm::outs() << "Param :: " << p->getNameAsString() << " \n";
      }
      llvm::outs() << "End init_kernel_body\n";
   }
};

/****************************************************************************
 * A DoAllCallNode represents the site of a do-all call. It also contains the kernel being
 * called as well as the graph object.
 *****************************************************************************/

struct DoAllCallNode {

   Rewriter & rewriter;
   CallExpr * self;
   GaloisKernel * kernel;
   RecordDecl * graph;
   GaloisApp *app_data;

   DoAllCallNode(Rewriter & r, CallExpr * call, CXXRecordDecl * k, GaloisApp * a) :
         rewriter(r), self(call), kernel(nullptr), graph(nullptr), app_data(a) {
      kernel = new GaloisKernel(r, k, app_data);
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      llvm::outs() << "Kernel name :: " << k->getNameAsString() << "\n";
   }
};

/****************************************************************************
 * A GaloisApp is a set of do-all calls. Each do-all call is represented
 * by a difference instance of the DoAllCallNode object which contains all the necessary
 * information.
 *****************************************************************************/
struct GaloisApp {
   Rewriter & rewriter;
   const char * dir;
   std::vector<DoAllCallNode *> doAllCalls;
   std::set<Type *> known_types;
   std::map<Type *, string> replacement;
   GaloisApp(Rewriter & r) :
         rewriter(r), dir(r.getSourceMgr().getFileEntryForID(r.getSourceMgr().getMainFileID())->getDir()->getName()) {
   }
   ~GaloisApp() {
      for (auto a : doAllCalls) {
         delete a;
      }
//      print_types();
   }
   void add_doAll_call(CallExpr * call, CXXRecordDecl * kernel) {
      DoAllCallNode * node = new DoAllCallNode(rewriter, call, kernel, this);
      doAllCalls.push_back(node);
   }
   void _add_known_types(const std::set<Type*> & s) {
      for (auto t : s) {
         llvm::outs() << " Registered known-type :: " << QualType::getAsString(t, Qualifiers()) << " \n";
         known_types.insert(t);
      }
   }
   void register_type(Type * t, string str) {
      known_types.insert(const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr()));
      replacement[(const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr()))] = str;
   }
   void print_types() {
      for (auto i : replacement) {
         llvm::outs() << " Type :: " << QualType::getAsString(i.first, Qualifiers()) << " mappedTo " << i.second << "\n";
      }
   }
   bool find_type(Type * t) {
      return replacement.find(t) != replacement.end() || replacement.find(const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr())) != replacement.end();
   }
   string replace_type(Type * t) {
      return replacement[(const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr()))];
   }
};
/****************************************************************************
 *****************************************************************************/
} //End GAST namespace
} //End Galois namespace

#endif /* SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_ */
