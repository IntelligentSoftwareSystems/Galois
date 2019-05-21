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

// Master tool for conversion

#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#include "patterns.h"

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

    LoopInfo(const CallExpr* cs, const QualType op, const Decl* decl,
             const LambdaExpr* lam)
        : CallSite(cs), Operator(op), OpDecl(decl), OpLambda(lam) {}
  };

  std::vector<LoopInfo> Loops;

  virtual void run(const MatchFinder::MatchResult& Result) {
    const CallExpr* FS = Result.Nodes.getNodeAs<clang::CallExpr>("gLoop");
    auto* gFunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("gLoopType");
    auto name   = gFunc->getNameInfo().getName().getAsString();

    bool forEach = std::string::npos != name.find("for_each");
    bool isLocal = std::string::npos != name.find("local");

    llvm::outs() << name << " " << forEach << " " << isLocal << "\n";
    // gFunc->dump();
    // llvm::outs() << "**\n";
    auto* gOp = FS->getArg(isLocal ? 1 : 2);

    if (const MaterializeTemporaryExpr* t =
            dyn_cast<const MaterializeTemporaryExpr>(gOp))
      gOp = t->GetTemporaryExpr();

    if (const ImplicitCastExpr* t = dyn_cast<const ImplicitCastExpr>(gOp))
      gOp = t->getSubExpr();

    if (const CXXFunctionalCastExpr* t =
            dyn_cast<const CXXFunctionalCastExpr>(gOp))
      gOp = t->getSubExpr();

    // Handle common types
    auto* gOpTyPtr        = gOp->getType().getTypePtr();
    Decl* dOp             = nullptr;
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

  void reset() { Loops.clear(); }
};

// Example SoA to AoS graph converter
class LoopRewriter : public MatchFinder::MatchCallback {
public:
  LoopRewriter(Rewriter& Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult& Result) {
    const MemberExpr* fieldRef = Result.Nodes.getNodeAs<MemberExpr>("fieldRef");
    const Expr* fieldUse       = Result.Nodes.getNodeAs<Expr>("fieldUse");
    const Expr* graphDataVar   = Result.Nodes.getNodeAs<Expr>("getDataVar");
    const NamedDecl* graphVar  = Result.Nodes.getNodeAs<NamedDecl>("graphVar");
    if (!fieldUse)
      fieldUse = fieldRef;
    for (auto& N : Result.Nodes.getMap()) {
      llvm::outs() << N.first << " ";
      N.second.dump(llvm::outs(), *Result.SourceManager);
    }

    llvm::outs() << fieldRef << " " << fieldUse << " " << graphDataVar << " "
                 << graphVar << "\n";
    assert(graphVar && fieldUse && fieldRef);

    auto str1 = fieldRef->getMemberDecl()->getNameAsString();
    //    auto str2 = graphDataVar->getFoundDecl()->getNameAsString();
    auto str3 = graphVar->getNameAsString();
    //    Rewrite.ReplaceText(fieldUse->getSourceRange(), "\\test\\");

    // Get the new text.
    std::string SStr;
    llvm::raw_string_ostream S(SStr);
    S << str3 << "_" << str1 << "[";
    graphDataVar->printPretty(S, 0, PrintingPolicy(Rewrite.getLangOpts()));
    S << "]";
    const std::string& Str = S.str();
    Rewrite.ReplaceText(fieldUse->getSourceRange(), Str);

    //    Rewrite.InsertTextBefore(fieldUse->getLocStart(), str3 + "_" + str1 +
    //    "["); Rewrite.InsertTextAfter(fieldUse->getLocEnd(), "]");
    //    Rewrite.ReplaceText(fieldUse->getSourceRange(),
    //    graphDataVar->getSourceRange());

    //    Rewrite.ReplaceText(fieldUse->getSourceRange(), str3 + "_" + str1 +
    //    "[" + str2 + "]");
  }

private:
  Rewriter& Rewrite;
};

#if 0

class GraphNodeMatcher : public MatchFinder::MatchCallback {
public:

  struct VarInfo {
  };
  
  static constexpr StatementMatcher Matcher = callExpr(callee(functionDecl(anyOf(hasName("getData"),hasName("edge_begin"),hasName("edge_end")))),hasArgument(0, expr().bind("var"))).bind("varUse")

  std::vector<VarInfo> GraphVars;
  
  virtual void run(const MatchFinder::MatchResult& Result) {
    auto* varA = Result.Nodes.getNodeAs<clang::CallExpr>("varUse");
    auto* varE = Result.Nodes.getNodeAs<clang::Expr>("var");

    varA->dump();
    llvm::outs() << "**\n";
    varE->dump();
    llvm::outs() << "++\n\n\n";
  }
};

class GraphValueMatcher : public MatchFinder::MatchCallback {

  StatementMatcher Matcher;
  

};
#endif

class OpenCLKernelConsumer : public ASTConsumer {
public:
  OpenCLKernelConsumer(Rewriter& R) : OpMatcherRewrite(R) {
    OpFinder.addMatcher(galoisLoop, &OpMatcher);
    OpFinderRewrite.addMatcher(allFields, &OpMatcherRewrite);
  }

  void HandleTranslationUnit(ASTContext& Context) override {
    OpMatcher.reset();
    OpFinder.matchAST(Context);
    OpFinderRewrite.matchAST(Context);

    // for (auto& L : OpMatcher.Loops) {
    //   if (L.OpDecl)
    //     OpFinderRewrite.matchAST(L.OpDecl);
    //   if (L.OpLambda)
    //     OpFinderRewrite.matchAST(L.OpLambda);
    // }
  }

private:
  MatchFinder OpFinder;
  LoopMatcher OpMatcher;

  MatchFinder OpFinderRewrite;
  LoopRewriter OpMatcherRewrite;
};

class OpenCLKernelAction : public ASTFrontendAction {
public:
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
        .write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
                                                 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<OpenCLKernelConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char** argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  //  LoopMatcher GFinder;
  //  GraphNodeMatcher GNodeFinder;
  //  MatchFinder Finder;
  //  Finder.addMatcher(GFinder.Matcher, &GFinder);
  //  Finder.addMatcher(GNodeFinder.Matcher, &GNodeFinder);

  //  Tool.run(newFrontendActionFactory(&Finder).get());
  //  return 0;

  return Tool.run(newFrontendActionFactory<OpenCLKernelAction>().get());
}
