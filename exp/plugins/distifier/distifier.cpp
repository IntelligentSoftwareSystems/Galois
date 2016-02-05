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

    LoopInfo(const CallExpr* cs) : CallSite(cs) {}
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

    llvm::outs() << name << " " << forEach << " " << isLocal << "\n**\n";
    //gFunc->dump();
    //llvm::outs() << "**\n";
    auto* gOp = FS->getArg(isLocal ? 1 : 2);

    if (const MaterializeTemporaryExpr* t = dyn_cast<const MaterializeTemporaryExpr>(gOp))
      gOp = t->GetTemporaryExpr();

    gOp->dump();
    llvm::outs() << "\n\n";

    Loops.emplace_back(FS);
  }
};




int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());
  
  LoopMatcher GFinder;
  MatchFinder Finder;
  Finder.addMatcher(GFinder.Matcher, &GFinder);

  return Tool.run(newFrontendActionFactory(&Finder).get());
}

