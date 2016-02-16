//Master tool for conversion

// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"



#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("distifier options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");


class LoopMatcher : public MatchFinder::MatchCallback {
public:

  struct LoopInfo {
    const CallExpr* CallSite;
    const QualType Operator;
    const Decl* OpDecl;
    const LambdaExpr* OpLambda;

    LoopInfo(const CallExpr* cs, const QualType op, const Decl* decl, const LambdaExpr* lam)
      : CallSite(cs), Operator(op), OpDecl(decl), OpLambda(lam) {}
  };
  
  StatementMatcher Matcher;
  std::vector<LoopInfo> Loops;
  
  LoopMatcher()
    :Matcher{
    callExpr(callee(functionDecl(
                                 anyOf(
                                       hasName("for_each"),
                                       hasName("for_each_local"),
                                       hasName("do_all"),
                                       hasName("do_all_local")
                                       )
                                 ).bind("gLoopType")
                    )
             ).bind("gLoop")
      }
  {}
  
  virtual void run(const MatchFinder::MatchResult& Result) {
    const CallExpr* FS = Result.Nodes.getNodeAs<clang::CallExpr>("gLoop");
    auto* gFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("gLoopType");
    auto name = gFunc->getNameInfo().getName().getAsString();

    bool forEach = std::string::npos != name.find("for_each");
    bool isLocal = std::string::npos != name.find("local");

    llvm::outs() << name << " " << forEach << " " << isLocal << "\n";
    //gFunc->dump();
    //llvm::outs() << "**\n";
    auto* gOp = FS->getArg(isLocal ? 1 : 2);

    if (const MaterializeTemporaryExpr* t = dyn_cast<const MaterializeTemporaryExpr>(gOp))
      gOp = t->GetTemporaryExpr();

    if (const ImplicitCastExpr* t = dyn_cast<const ImplicitCastExpr>(gOp))
      gOp = t->getSubExpr();

    if (const CXXFunctionalCastExpr* t = dyn_cast<const CXXFunctionalCastExpr>(gOp))
      gOp = t->getSubExpr();

    //Handle common types
    auto* gOpTyPtr = gOp->getType().getTypePtr();
    Decl* dOp = nullptr;
    const LambdaExpr* lOp = dyn_cast<const LambdaExpr>(gOp);

    if (auto* stOp = gOpTyPtr->getAsStructureType()) {
      dOp = stOp->getDecl();
      llvm::outs() << "Op is Decl structure\n";
    } else if (lOp) {
      llvm::outs() << "Op is Lambda\n";      
    } else {
        gOp->dump();
        llvm::outs() << "**\n";
        if (gOp->getType().getTypePtr()->isRecordType()) {
          auto* op = gOp->getType().getTypePtr()->getAsStructureType();
          if (op) {
            op->dump();
            if (op->getDecl())
              op->getDecl()->dump();
            else
              llvm::outs() << "Not RecordType with Decl\n";
          }
          gOp->getType().getTypePtr()->dump();
        }
    }
    
    llvm::outs() << "\n\n";
    Loops.emplace_back(FS, gOp->getType(), dOp, lOp);
  }
};

class GraphNodeMatcher : public MatchFinder::MatchCallback {
public:

  struct VarInfo {
  };
  
  StatementMatcher Matcher;
  std::vector<VarInfo> GraphVars;
  
  GraphNodeMatcher()
    :Matcher{
    callExpr(callee(functionDecl(
                                 anyOf(
                                       hasName("getData"),
                                       hasName("edge_begin"),
                                       hasName("edge_end")
                                       )
                                 )
                    ),
             hasArgument(0, expr().bind("var"))
             ).bind("varUse")
      }
  {}
  
  virtual void run(const MatchFinder::MatchResult& Result) {
    auto* varA = Result.Nodes.getNodeAs<clang::CallExpr>("varUse");
    auto* varE = Result.Nodes.getNodeAs<clang::Expr>("var");

    varA->dump();
    llvm::outs() << "**\n";
    varE->dump();
    llvm::outs() << "++\n\n\n";
  }
};



int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  
  LoopMatcher GFinder;
  GraphNodeMatcher GNodeFinder;
  MatchFinder Finder;
  Finder.addMatcher(GFinder.Matcher, &GFinder);
  Finder.addMatcher(GNodeFinder.Matcher, &GNodeFinder);

  Tool.run(newFrontendActionFactory(&Finder).get());
  return 0;
}

