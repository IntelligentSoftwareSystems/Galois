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
 * GaloisAST.h
 *
 *  Created on: Jan 21, 2016
 *      Author: rashid
 */
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "ClangUtils.h"

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

// Forward declarations:
struct GaloisApp;

///////////////////////////////////////////////////////////////////////////
class GraphTypeReplacer : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  std::vector<CXXRecordDecl*> graphDecls;
  std::set<Type*> graph_types;
  std::map<CXXRecordDecl*, std::vector<Type*>*> typeDecls;

public:
  static const char* getReturnType(const char* fname) {
    if (strcmp(fname, "edge_begin") == 0) {
      return "cl_edge_iterator ";
    } else if (strcmp(fname, "edge_end") == 0) {
      return "cl_edge_iterator ";
    } else if (strcmp(fname, "getData") == 0) {
      return "cl_node_data_type ";
    } else if (strcmp(fname, "getEdgeDst") == 0) {
      return "cl_node_iterator ";
    } else if (strcmp(fname, "getEdgeData") == 0) {
      return "cl_edge_data_type ";
    }
    return "invalid_type";
  }
  GraphTypeReplacer(Rewriter& rewriter) : rewriter(rewriter) {}
  virtual ~GraphTypeReplacer() {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    VarDecl* vd =
        const_cast<VarDecl*>(Results.Nodes.getNodeAs<VarDecl>("varDecl"));
    FunctionDecl* graphCall = const_cast<FunctionDecl*>(
        Results.Nodes.getNodeAs<FunctionDecl>("callExpr"));
    if (vd) {
      //         llvm::outs() << "VarDecl using Galois graph object :: \n";
      //         vd->dump();
      //         vd->getTypeSourceInfo();
      //         vd->setType(vd->getASTContext().getUIntMaxType());
      rewriter.ReplaceText(
          vd->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
          getReturnType(graphCall->getNameAsString().c_str()));
      //         llvm::outs() << ast_utility.get_string(vd->getSourceRange()) <<
      //         "\n"; llvm::outs() << vd->getNameAsString() << "\n";
      //         llvm::outs() << graphCall->getNameAsString() << "\n";
      //         llvm::outs() <<
      //         "\n==============================================\n";
    }
  }
};

/****************************************************************************
 *****************************************************************************/

/****************************************************************************
 *****************************************************************************/

class TypeDefCollector : public RecursiveASTVisitor<TypeDefCollector> {
public:
  std::set<Type*> types_used;
  CXXRecordDecl* class_def;
  std::string class_name;
  bool in_class = false;
  TypeDefCollector(CXXRecordDecl* d) : class_def(d) {
    class_name = class_def->getNameAsString(); // QualType::getAsString(class_def->getTypeForDecl(),
                                               // Qualifiers());
    /*
          llvm::outs() << "Initialized typedef collector for class :: " <<
       class_name << " \n"; llvm::outs() <<
       "------------------------------------------------------\n";
          d->dump(llvm::outs());
          llvm::outs() <<
       "------------------------------------------------------\n";
    */
  }
  virtual ~TypeDefCollector() {}
  virtual bool TraverseDecl(Decl* d) {
    if (d) {
      if (isa<CXXRecordDecl>(d)) {
        CXXRecordDecl* cd = const_cast<CXXRecordDecl*>(cast<CXXRecordDecl>(d));
        std::string decl_name = cd->getNameAsString(); // QualType::getAsString(cd->getTypeForDecl(),
                                                       // Qualifiers());
        llvm::outs() << " Traverse class :: " << decl_name << " "
                     << QualType::getAsString(cd->getTypeForDecl(),
                                              Qualifiers())
                     << " \n";
        if (decl_name.compare(class_name) == 0) {
          llvm::outs() << " Graphclass name :: "
                       << QualType::getAsString(class_def->getTypeForDecl(),
                                                Qualifiers())
                       << "\n";
          llvm::outs() << " Recursing class :: " << decl_name << " \n";
          in_class = true;
          RecursiveASTVisitor<TypeDefCollector>::TraverseDecl(cd);
          in_class = false;
          llvm::outs() << " Done recursing class :: " << decl_name << " \n";
        }    // end names do not match
      }      // end of isa<CXXRecordDecl>
      else { // Not a CXXRecordecl
        RecursiveASTVisitor<TypeDefCollector>::TraverseDecl(d);
      }
    } // End d not-null
    return true;
  }
  //   virtual bool TraverseCXXRecordDecl(CXXRecordDecl * decl) {
  //      llvm::outs() << " Traverse class :: " << decl->getNameAsString() << "
  //      \n"; if ( decl == class_def){
  //         llvm::outs() << " Recursing class :: " << decl->getNameAsString( )
  //         << " \n";
  //         RecursiveASTVisitor<TypeDefCollector>::TraverseCXXRecordDecl(decl);
  //         llvm::outs() << " Done recursing class :: " <<
  //         decl->getNameAsString( ) << " \n";
  //      }
  //      return true;
  //   }
  virtual bool VisitTypedefDecl(TypedefDecl* decl) {
    // typedef real_type exposed_type
    if (in_class) {
      Type* exposed_type =
          const_cast<Type*>(decl->getCanonicalDecl()->getTypeForDecl());
      Type* real_type =
          const_cast<Type*>(decl->getUnderlyingType().getTypePtr());
      //         DeclContext * parent =
      //         const_cast<DeclContext*>(decl->getParentFunctionOrMethod());
      if (exposed_type) //&& decl->getParentFunctionOrMethod()==class_def)
      {
        std::string s = QualType::getAsString(exposed_type, Qualifiers());
        //         if(s.find("edge_iterator")!=s.npos)
        {
          llvm::outs() << " ExposedType :: "
                       << QualType::getAsString(exposed_type, Qualifiers())
                       << ", RealType:: "
                       << QualType::getAsString(real_type, Qualifiers()) << " ";
          //            if(parent)
          //               llvm::outs()<< parent->get)
          llvm::outs() << "\n";

          //            llvm::outs()<< "TypeDefDecl" <<
          //            QualType::getAsString(decl->getUnderlyingType().getTypePtr(),
          //            Qualifiers()) << " MappedTo " <<
          //            QualType::getAsString(decl->getCanonicalDecl()->getTypeForDecl(),
          //            Qualifiers()) << "\n";
        }
      }
    } // End if in_class
    //      RecursiveASTVisitor<TypeDefCollector>::VisitTypedefDecl(decl);
    return true;
  }
};
///////////////////////////////////////////////////////////////////////////
class GraphTypeMapper : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  std::vector<CXXRecordDecl*> graphDecls;
  std::set<Type*> graph_types;
  std::map<CXXRecordDecl*, std::vector<Type*>*> typeDecls;

public:
  GraphTypeMapper(Rewriter& rewriter) : rewriter(rewriter) {}
  virtual ~GraphTypeMapper() {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    CXXRecordDecl* decl = const_cast<CXXRecordDecl*>(
        Results.Nodes.getNodeAs<clang::CXXRecordDecl>("GraphType"));
    VarDecl* vd =
        const_cast<VarDecl*>(Results.Nodes.getNodeAs<VarDecl>("graphDecl"));
    assert(decl != nullptr);
    assert(vd != nullptr);
    if (vd) {
      //         rewriter.ReplaceText(vd->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
      //         " CLGraph ");
    }
    llvm::outs() << "GraphClass :: " << decl->getNameAsString() << ", "
                 << decl->getCanonicalDecl()->getNameAsString() << "\n";
    graphDecls.push_back(decl);
    graph_types.insert(
        const_cast<Type*>(decl->getCanonicalDecl()->getTypeForDecl()));
    {
      //         TypeDefCollector tdc(decl);
      //         tdc.TraverseDecl(decl->getTranslationUnitDecl());
    }
    for (auto m : decl->methods()) {
      //         if (m->isConst() && m->param_size() == 1)
      {
        std::string mname = m->getNameAsString();
        if (mname.compare("edge_begin") == 0) {
          llvm::outs() << "EdgeBegin method found " << m->getNameAsString()
                       << "\n";
        } else if (mname.compare("edge_end") == 0) {
          llvm::outs() << "EdgeEnd method found " << m->getNameAsString()
                       << "\n";
        } else if (mname.compare("getData") == 0) {
          llvm::outs() << "GetData method found " << m->getNameAsString()
                       << "\n";
        } else if (mname.compare("getEdgeData") == 0) {
          llvm::outs() << "GetEdgeData method found " << m->getNameAsString()
                       << "\n";
        } else if (mname.compare("getEdgeDst") == 0) {
          llvm::outs() << "GetEdgeDst method found " << m->getNameAsString()
                       << "\n";
        }
      } // End if ( m->isConst() && m->param_size()==2)
    }
  }
};

/****************************************************************************
 *****************************************************************************/

class TypeCollector : public RecursiveASTVisitor<TypeCollector> {
public:
  std::set<Type*> types_used;
  TypeCollector() {}
  virtual ~TypeCollector() {}
  virtual bool TraverseDecl(Decl* decl) {
    RecursiveASTVisitor<TypeCollector>::TraverseDecl(decl);
    return true;
  }
  virtual bool VisitVarDecl(VarDecl* decl) {
    types_used.insert(const_cast<Type*>(decl->getType().getTypePtr()));
    return true;
  }
};
/****************************************************************************
 *****************************************************************************/
struct GaloisKernel {
  Rewriter& rewriter;
  CXXRecordDecl* kernel;
  CXXMethodDecl* kernel_body;
  GaloisApp* app_data;
  GaloisKernel(Rewriter& r, CXXRecordDecl* d, GaloisApp* a)
      : rewriter(r), kernel(d), kernel_body(nullptr), app_data(a) {
    llvm::outs() << "Creating GaloisKernel :: " << kernel->getNameAsString()
                 << "\n";
    int method_counter = 0;
    for (auto m : kernel->methods()) {
      llvm::outs() << method_counter << "] Is overloaded ? "
                   << (m->isOverloadedOperator() ? "Yes" : "No") << " ";
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
      llvm::outs() << "Lambda call operator :: "
                   << d->getLambdaCallOperator()->getNameAsString() << "\n";
    }
    //////////Scan for types used.
    {
      llvm::outs() << "Scanning for types used:: \n";
      TypeCollector tc;
      tc.TraverseDecl(kernel);
      for (Type* t : tc.types_used) {
        llvm::outs() << " Type : " << QualType::getAsString(t, Qualifiers())
                     << "\n\t"
                     << QualType::getAsString(
                            t->getCanonicalTypeInternal().getTypePtr(),
                            Qualifiers())
                     << "\n";
      }
      llvm::outs() << " Done scanning for types.\n";
    }
    //####################################
    rewriter.InsertText(d->method_begin()->getLocStart(),
                        get_impl_string().c_str());
    //      rewriter.InsertTextAfter(kernel_body->getLocEnd(), "#endif\n");
    //      rewriter.InsertTextBefore(kernel_body->getLocStart(), "#if 0\n");
    rewriter.ReplaceText(
        SourceRange(kernel_body->getLocStart(), kernel_body->getLocEnd()), "");
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
  string get_call_string() { return "\ncl_call_wrapper()\n;"; }
  string get_impl_string() {
    std::string ret_string = "\nCL_Kernel * get_kernel(size_t num_items){\n";
    ret_string +=
        " static CL_Kernel kernel(getCLContext()->get_default_device(),\"";
    ret_string += kernel->getNameAsString();
    ret_string += ".cl\", \"";
    ret_string += kernel->getNameAsString();
    ret_string += "\", false);\n";
    int i = 0;
    char s[1024];
    for (auto m : this->kernel->fields()) {
      string ref;
      if (m->getType().getTypePtr()->isAnyPointerType())
        ref = "->";
      else
        ref = ".";
      sprintf(
          s,
          "kernel.set_arg(%d,sizeof(%s%sdevice_ptr()),&(%s%sdevice_ptr()));\n",
          i, m->getNameAsString().c_str(), ref.c_str(),
          m->getNameAsString().c_str(), ref.c_str());
      ret_string += s;
      i++;
    }
    sprintf(s, "kernel.set_arg(%d,sizeof(num_items),&num_items);\n", i);
    ret_string += s;
    ret_string += " return &kernel;\n";
    ret_string += "}\n";
    return ret_string;
  }
  void init_kernel_body(CXXMethodDecl* m) {
    // TODO RK : should check for double initialization!
    kernel_body = m;
    /*
          llvm::outs() << "\n=============@ASTDump for
       operator========================\n"; m->dump(llvm::outs()); llvm::outs()
       << "\n=============@END ASTDump for operator========================\n";
    */
    std::string name = m->getNameAsString();
    llvm::outs() << "Initializing kernel_body :: " << name << "\n";
    for (auto p : kernel_body->parameters()) {
      llvm::outs() << "Param :: " << p->getNameAsString() << " \n";
    }
    llvm::outs() << "End init_kernel_body\n";
  }
};

/****************************************************************************
 * A DoAllCallNode represents the site of a do-all call. It also contains the
 *kernel being called as well as the graph object.
 *****************************************************************************/

struct DoAllCallNode {

  Rewriter& rewriter;
  CallExpr* self;
  GaloisKernel* kernel;
  RecordDecl* graph;
  GaloisApp* app_data;

  DoAllCallNode(Rewriter& r, CallExpr* call, CXXRecordDecl* k, GaloisApp* a)
      : rewriter(r), self(call), kernel(nullptr), graph(nullptr), app_data(a) {
    kernel = new GaloisKernel(r, k, app_data);
    clang::LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    clang::PrintingPolicy Policy(LangOpts);

    //      llvm::outs() << "\nBEGIN@@@@@@@@@@@\n";
    //      llvm::outs() << ast_utility.get_string(call->getLocStart(),
    //      call->getLocEnd()) << "\n";

    llvm::outs() << "Kernel name :: " << k->getNameAsString() << "\n";
    //      ast_utility.print_expr(call->getLocStart(), call->getLocEnd());
    //      rewriter.InsertText((*k->field_begin())->getLocStart(),
    //      "KERNEL_INSTANCE_HERE\n"); for(auto f : k->fields()){
    /*         std::string copy_str(f->getNameAsString().c_str());
             std::string toDev(copy_str);
             toDev=";\n"+toDev+".copy_to_device();\n";

             std::string toHost(copy_str);
             toHost=";\n"+toHost+".copy_to_host();\n";
             rewriter.InsertTextAfter(call->getLocEnd(), toHost.c_str());
             rewriter.InsertText(call->getLocStart(), toDev.c_str());*/
    //         size_t len =
    //         std::distance(ast_utility.get_src_ptr(call->getLocStart()),
    //         ast_utility.get_src_ptr(call->getLocEnd()));
    //      }
    //      llvm::outs() << "\nEND\n";
  }
};

/****************************************************************************
 * A GaloisApp is a set of do-all calls. Each do-all call is represented
 * by a difference instance of the DoAllCallNode object which contains all the
 *necessary information.
 *****************************************************************************/
struct GaloisApp {
  std::vector<DoAllCallNode*> doAllCalls;
  std::set<Type*> known_types;
  std::map<Type*, string> replacement;
  Rewriter& rewriter;
  std::ofstream cl_file;
  GaloisApp(Rewriter& r) : rewriter(r), cl_file("cl_impl.cl") {}
  ~GaloisApp() {
    for (auto a : doAllCalls) {
      delete a;
    }
    print_types();
  }
  void add_doAll_call(CallExpr* call, CXXRecordDecl* kernel) {
    DoAllCallNode* node = new DoAllCallNode(rewriter, call, kernel, this);
    doAllCalls.push_back(node);
  }
  void _add_known_types(const std::set<Type*>& s) {
    for (auto t : s) {
      llvm::outs() << " Registered known-type :: "
                   << QualType::getAsString(t, Qualifiers()) << " \n";
      known_types.insert(t);
    }
  }
  void register_type(Type* t, string str) {
    known_types.insert(
        const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr()));
    replacement[(const_cast<Type*>(
        t->getCanonicalTypeUnqualified().getTypePtr()))] = str;
    //      replacement[t]=str;
  }
  void print_types() {
    for (auto i : replacement) {
      llvm::outs() << " Type :: "
                   << QualType::getAsString(i.first, Qualifiers())
                   << " mappedTo " << i.second << "\n";
    }
  }
  bool find_type(Type* t) {
    //      return known_types.find(t)!=known_types.end();
    return replacement.find(t) != replacement.end() ||
           replacement.find(const_cast<Type*>(
               t->getCanonicalTypeUnqualified().getTypePtr())) !=
               replacement.end();
    //      return replacement.find(t)!=replacement.end();
  }
  string replace_type(Type* t) {
    //      return replacement[t];
    return replacement[(
        const_cast<Type*>(t->getCanonicalTypeUnqualified().getTypePtr()))];
    //      return replacement.find(t)!=replacement.end();
  }
};
/****************************************************************************
 *****************************************************************************/
} // namespace GAST
} // namespace galois

#endif /* SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_ */
