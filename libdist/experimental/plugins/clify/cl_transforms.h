//@author Rashid Kaleem <rashid.kaleem@gmail.com>
#include "utils.h"
#include "gAST.h"
#ifndef _CLIFY_CL_TRANSFORMS_H_
#define _CLIFY_CL_TRANSFORMS_H_

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class GraphTypedefRewriter : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  ASTContext& ctx;

public:
  GraphTypedefRewriter(Rewriter& rewriter, ASTContext& c)
      : rewriter(rewriter), ctx(c) {}
  ~GraphTypedefRewriter() {}

  // TODO RK - Fix the missing part of the declaration. Matcher returns 'G'
  // instead of 'Graph'.
  virtual void run(const MatchFinder::MatchResult& Results) {
    TypedefDecl* decl = const_cast<TypedefDecl*>(
        Results.Nodes.getNodeAs<TypedefDecl>("GraphTypeDef"));
    if (decl) {
      llvm::outs() << " Rewriting typedef for graph \n";
      decl->dump(llvm::outs());
      llvm::outs() << " =================\n";
      llvm::outs() << decl->getUnderlyingType().getAsString() << "\n";
      string s = "#include \"galois/opencl/CL_Header.h\"\nusing namespace "
                 "galois::opencl;\n";
      {
        // TODO - RK - A hack. Since the begin/end location of the decl are not
        // returning
        // the entire decl (terminating at 'G' instead of at 'Graph', we try to
        // hack around it by scanning the end.
        char* ptr = ast_utility.get_src_ptr(decl->getSourceRange().getBegin());
        char* end = ptr;
        while (*end != ';')
          end++;
        string str(ptr, std::distance(ptr, end));
        s += str;
      }
      // s+=ast_utility.get_string(decl->getSourceRange().getBegin(),
      // decl->getSourceRange().getEnd());
      string old_name = decl->getUnderlyingType().getAsString();
      old_name        = old_name.substr(0, old_name.find("<"));
      s.replace(s.find(old_name), old_name.length(), "Graphs::CL_LC_Graph");
      rewriter.ReplaceText(SourceRange(decl->getSourceRange()), s);
    }
  }
};

/*********************************************************************************************************
 *
 *********************************************************************************************************/
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

/*********************************************************************************************************
 *
 *********************************************************************************************************/

class DoAllHandler : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  galois::GAST::GaloisApp& app_data;

public:
  DoAllHandler(Rewriter& rewriter, galois::GAST::GaloisApp& _ad)
      : rewriter(rewriter), app_data(_ad) {}

  virtual void run(const MatchFinder::MatchResult& Results) {
    CallExpr* callFS = const_cast<CallExpr*>(
        Results.Nodes.getNodeAs<clang::CallExpr>("galoisLoop"));
    //      VarDecl * decl =
    //      const_cast<VarDecl*>(Results.Nodes.getNodeAs<VarDecl>("graphDecl"));
    //      if(decl)
    //         rewriter.ReplaceText(decl->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
    //         " CLGraph ");

    if (callFS) {
      llvm::outs() << "GaloisLoop found  - #Args :: " << callFS->getNumArgs()
                   << "\n";
      CXXRecordDecl* kernel = const_cast<CXXRecordDecl*>(
          Results.Nodes.getNodeAs<CXXRecordDecl>("kernelType"));
      assert(kernel != nullptr && "KernelType cast failed.");
      app_data.add_doAll_call(callFS, kernel);
      llvm::outs() << "galois::do_All loop found "
                   << callFS->getCalleeDecl()
                          ->getCanonicalDecl()
                          ->getAsFunction()
                          ->getNameAsString()
                   << "\n";
      // Get the arguments:
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      //         unsigned write_setNum = 0;
      //         string GraphNode;
      // Print the important arguments::
      if (false && callFS->getNumArgs() >= 3) {
        llvm::outs() << "Begin iterator :: ";
        callFS->getArg(0)->printPretty(llvm::outs(), nullptr, Policy);
        llvm::outs() << ", Type :: "
                     << QualType::getAsString(
                            callFS->getArg(0)->getType().split())
                     << "\n";
        llvm::outs() << "End iterator :: ";
        callFS->getArg(1)->printPretty(llvm::outs(), nullptr, Policy);
        llvm::outs() << ", Type :: "
                     << QualType::getAsString(
                            callFS->getArg(1)->getType().split())
                     << "\n";
        llvm::outs() << "Operator instance :: ";
        callFS->getArg(2)->printPretty(llvm::outs(), nullptr, Policy);
        llvm::outs() << ", Type :: "
                     << QualType::getAsString(
                            callFS->getArg(2)->getType().split())
                     << "\n";
        llvm::outs() << "-------->OPERATOR CALLED IN DO_ALL::\n";
        callFS->getArg(2)->getBestDynamicClassType()->dump();
        llvm::outs() << ", Type :: "
                     << callFS->getArg(2)->getBestDynamicClassType()->getName()
                     << "\n";
      }
      rewriter.ReplaceText(SourceRange(callFS->getCallee()->getLocStart(),
                                       callFS->getCallee()->getLocEnd()),
                           "do_all_cl");
    }
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/

/*********************************************************************************************************
 *
 *********************************************************************************************************/
class NodeDataGen : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  ASTContext& ctx;
  galois::GAST::GaloisApp& app;
  std::ofstream header_file;
  bool nodeDataFound, edgeDataFound;

public:
  NodeDataGen(Rewriter& rewriter, ASTContext& c, galois::GAST::GaloisApp& a)
      : rewriter(rewriter), ctx(c), app(a), nodeDataFound(false),
        edgeDataFound(false) {
    char hfilename[1024];
    sprintf(hfilename, "%s/app_header.h", app.dir);
    header_file.open(hfilename);
    header_file << "//CL Graph declarations \n";
    header_file << "//Generated : " << ast_utility.get_timestamp() << "\n";
  }
  // TODO - RK - Clean up graph generation code.
  ~NodeDataGen() {
    header_file << " #include \"graph_header.h\"\n";
    if (false) { // Begin graph implementation.
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
      res += "struct  _" + sname + "{\n";
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
    } else if (q.getAsString() == "void") {
      res += "void ";
      res += sname;
      res += ";\n";
    }
    header_file << res;
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

class GetDataHandler : public MatchFinder::MatchCallback {
public:
  Rewriter& rewriter;
  galois::GAST::GaloisApp& app;

public:
  GetDataHandler(Rewriter& rewriter, galois::GAST::GaloisApp& a)
      : rewriter(rewriter), app(a) {}
  virtual void run(const MatchFinder::MatchResult& Results) {
    CXXMemberCallExpr* decl = const_cast<CXXMemberCallExpr*>(
        Results.Nodes.getNodeAs<CXXMemberCallExpr>("callsite"));
    //      llvm::outs() << "Found - getData :: \n";
    if (decl) {
      string txt = rewriter.getRewrittenText(SourceRange(
          decl->getCallee()->getLocStart(), decl->getCallee()->getLocEnd()));
      llvm::outs() << "Found - getData :: "
                   << ast_utility.get_string(decl->getCallee()->getLocStart(),
                                             decl->getCallee()->getLocEnd())
                   << " w/" << txt << "\n";
      txt.replace(txt.find("getData"), strlen("getData"), "getDataW");
      rewriter.ReplaceText(SourceRange(decl->getCallee()->getLocStart(),
                                       decl->getCallee()->getLocEnd()),
                           txt);
    }
  }
};
/*********************************************************************************************************
 *
 *********************************************************************************************************/

#endif //_CLIFY_CL_TRANSFORMS_H_
