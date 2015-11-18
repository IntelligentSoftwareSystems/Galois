/*
 * Author : Gurbinder Gill (gurbinder533@gmail.com)
 */
#include <sstream>
#include <climits>
#include <vector>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

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

namespace {


  class GaloisFunctionsVisitor : public RecursiveASTVisitor<GaloisFunctionsVisitor> {
  private:
    ASTContext* astContext;

  public:
    explicit GaloisFunctionsVisitor(CompilerInstance *CI) : astContext(&(CI->getASTContext())){}

    virtual bool VisitCXXRecordDecl(CXXRecordDecl* Dec){
      Dec->dump();
      return true;
    }
    virtual bool VisitFunctionDecl(FunctionDecl* func){
      //std::string funcName = func->getNameInfo().getName().getAsString();
      std::string funcName = func->getNameAsString();
      if (funcName == "foo"){
        llvm::errs() << "Found" << "\n";
      }
      return true;
    }
  };


  class NameSpaceHandler : public MatchFinder::MatchCallback {
    //CompilerInstance &Instance;
    clang::LangOptions& langOptions; //Instance.getLangOpts();
  public:
    NameSpaceHandler(clang::LangOptions &langOptions) : langOptions(langOptions){}
    virtual void run(const MatchFinder::MatchResult &Results){
      llvm::outs() << "It is coming here\n"
                   << Results.Nodes.getNodeAs<clang::NestedNameSpecifier>("galoisLoop")->getAsNamespace()->getIdentifier()->getName();
      if (const NestedNameSpecifier* NSDecl = Results.Nodes.getNodeAs<clang::NestedNameSpecifier>("galoisLoop")) {
        llvm::outs() << "Found Galois Loop\n";
        //NSDecl->dump(langOptions);
      }
    }
  };

  class ForStmtHandler : public MatchFinder::MatchCallback {
    public:
      virtual void run(const MatchFinder::MatchResult &Results) {
        if (const ForStmt* FS = Results.Nodes.getNodeAs<clang::ForStmt>("forLoop")) {
          llvm::outs() << "for loop found\n";
          FS->dump();
        }
      }
  };

  struct Write_set_PUSH {
    string GRAPH_NAME;
    string NODE_TYPE;
    string FIELD_TYPE;
    string FIELD_NAME;
    string REDUCE_OP_EXPR;
    string VAL_TYPE;
    string RESET_VAL_EXPR;
  };

  struct Write_set_PULL {
    string GRAPH_NAME;
    string NODE_TYPE;
    string FIELD_TYPE;
    string FIELD_NAME;
    string VAL_TYPE;
  };

  class FunctionCallHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
    public:
      FunctionCallHandler(Rewriter &rewriter) :  rewriter(rewriter){}
      virtual void run(const MatchFinder::MatchResult &Results) {
        const CallExpr* callFS = Results.Nodes.getNodeAs<clang::CallExpr>("galoisLoop");
        llvm::outs() << "It is coming here\n" << callFS->getNumArgs() << "\n";

        if(callFS){

          SourceLocation ST_main = callFS->getSourceRange().getBegin();

          llvm::outs() << "Galois::do_All loop found\n";

          //Get the arguments:
          clang::LangOptions LangOpts;
          LangOpts.CPlusPlus = true;
          clang::PrintingPolicy Policy(LangOpts);

          // Vector to store read and write set.
          vector<pair<string, string>>read_set_vec;
          vector<pair<string,string>>write_set_vec;
          vector<string>write_setGNode_vec;
          unsigned write_setNum = 0;
          string GraphNode;

          vector<Write_set_PUSH> write_set_vec_PUSH;
          vector<Write_set_PULL> write_set_vec_PULL;

          /*
             string str_first;
             llvm::raw_string_ostream s_first(str_first);
             callFS->getArg(0)->printPretty(s_first, 0, Policy);
          //llvm::outs() << " //////// > " << s_first.str() << "\n";
          callFS->getArg(0)->dump();
          */

          for(int i = 0, j = callFS->getNumArgs(); i < j; ++i){
            string str_arg;
            llvm::raw_string_ostream s(str_arg);
            callFS->getArg(i)->printPretty(s, 0, Policy);

            const CallExpr* callExpr = dyn_cast<CallExpr>(callFS->getArg(i));
            //callExpr->dump();
            if (callExpr) {
              const FunctionDecl* func = callExpr->getDirectCallee();
              if(func) {
                if (func->getNameInfo().getName().getAsString() == "read_set"){
                  llvm::outs() << "Inside arg read_set: " << i <<  s.str() << "\n\n";
                }
                if (func->getNameInfo().getName().getAsString() == "write_set"){
                  llvm::outs() << "Inside arg write_set: " << i <<  s.str() << "\n\n";

                  // Take out first argument to see if it is push or pull
                  string str_sub_arg;
                  llvm::raw_string_ostream s_sub(str_sub_arg);
                  callExpr->getArg(0)->printPretty(s_sub, 0, Policy);

                  string str_sync_type = s_sub.str();
                  if(str_sync_type.front() == '"'){
                    str_sync_type.erase(0,1);
                    str_sync_type.erase(str_sync_type.size()-1);
                  }

                  //SYNC_PUSH
                  //if(callExpr->getNumArgs() == 8) {
                  if(str_sync_type == "sync_push") {
                    vector<string> Temp_vec_PUSH;
                    for(unsigned i = 0; i < callExpr->getNumArgs(); ++i){
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();
                      if(temp_str.front() == '"'){
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if(temp_str.front() == '&'){
                        temp_str.erase(0,1);
                      }

                      llvm::outs() << " ------- > " << temp_str << "\n";
                      Temp_vec_PUSH.push_back(temp_str);

                    }
                    Write_set_PUSH WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PUSH[1];
                    WR_entry.NODE_TYPE = Temp_vec_PUSH[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PUSH[3];
                    WR_entry.FIELD_NAME = Temp_vec_PUSH[4];
                    WR_entry.VAL_TYPE = Temp_vec_PUSH[5];
                    WR_entry.REDUCE_OP_EXPR = Temp_vec_PUSH[6];
                    WR_entry.RESET_VAL_EXPR = Temp_vec_PUSH[7];

                    write_set_vec_PUSH.push_back(WR_entry);
                  }

                  //SYNC_PULL
                  else if(str_sync_type == "sync_pull") {
                    vector<string> Temp_vec_PULL;
                    for(unsigned i = 0; i < callExpr->getNumArgs(); ++i){
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(i)->printPretty(s_sub, 0, Policy);

                      string temp_str = s_sub.str();
                      if(temp_str.front() == '"'){
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if(temp_str.front() == '&'){
                        temp_str.erase(0,1);
                      }

                      Temp_vec_PULL.push_back(temp_str);

                    }
                    Write_set_PULL WR_entry;
                    WR_entry.GRAPH_NAME = Temp_vec_PULL[1];
                    WR_entry.NODE_TYPE = Temp_vec_PULL[2];
                    WR_entry.FIELD_TYPE = Temp_vec_PULL[3];
                    WR_entry.FIELD_NAME = Temp_vec_PULL[4];
                    WR_entry.VAL_TYPE = Temp_vec_PULL[5];

                    write_set_vec_PULL.push_back(WR_entry);
                  }
                 }
                }
              }
            }

            stringstream SSSyncer;
            unsigned counter = 0;
            for(auto i : write_set_vec_PUSH) {
              SSSyncer << " struct Syncer_" << counter << " {\n"
                << "\tstatic " << i.VAL_TYPE <<" extract( const " << i.NODE_TYPE << " node)" << "{ return " << "node." << i.FIELD_NAME <<  "; }\n"
                <<"\tstatic void reduce (" << i.NODE_TYPE << " node, " << i.VAL_TYPE << " y) "
                << i.REDUCE_OP_EXPR << "\n"
                //<<"\tstatic void reset (" << i.NODE_TYPE << " node ) { node." << i.FIELD_NAME << " = " << i.RESET_VAL_EXPR << "; }\n" 
                <<"\tstatic void reset (" << i.NODE_TYPE << " node )" << i.RESET_VAL_EXPR << "\n"
                <<"\ttypedef " << i.VAL_TYPE << " ValTy;\n"
                <<"};\n";


              rewriter.InsertText(ST_main, SSSyncer.str(), true, true);
              SSSyncer.str(string());
              SSSyncer.clear();
              ++counter;
            }

            // Addding structure for pull sync
            stringstream SSSyncer_pull;
            counter = 0;
            for(auto i : write_set_vec_PULL) {
              SSSyncer_pull << " struct SyncerPull_" << counter << " {\n"
                << "\tstatic " << i.VAL_TYPE <<" extract( const " << i.NODE_TYPE << " node)" << "{ return " << "node." << i.FIELD_NAME <<  "; }\n"
                <<"\tstatic void setVal (" << i.NODE_TYPE << " node, " << i.VAL_TYPE << " y) "
                << "{node." << i.FIELD_NAME << " = y; }\n"
                <<"\ttypedef " << i.VAL_TYPE << " ValTy;\n"
                <<"};\n";

              rewriter.InsertText(ST_main, SSSyncer_pull.str(), true, true);
              SSSyncer_pull.str(string());
              SSSyncer_pull.clear();
              ++counter;
            }

            // Adding Syn calls for write set
            stringstream SSAfter;
            SourceLocation ST = callFS->getSourceRange().getEnd().getLocWithOffset(2);
            for (unsigned i = 0; i < write_set_vec_PUSH.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              //SSAfter <<"\n" <<write_set_vec_PUSH[i].GRAPH_NAME<< ".sync_push<Syncer_" << i << ">" <<"();\n";
              SSAfter <<"\n" << "_graph.sync_push<Syncer_" << i << ">" <<"();\n";
              rewriter.InsertText(ST, SSAfter.str(), true, true);
            }
            //For sync Pull
            for (unsigned i = 0; i < write_set_vec_PULL.size(); i++) {
              SSAfter.str(string());
              SSAfter.clear();
              SSAfter <<"\n" << "_graph.sync_pull<SyncerPull_" << i << ">" <<"();\n";
              rewriter.InsertText(ST, SSAfter.str(), true, true);
            }
          }
        }
      };



  class GaloisFunctionsConsumer : public ASTConsumer {
    private:
    CompilerInstance &Instance;
    std::set<std::string> ParsedTemplates;
    GaloisFunctionsVisitor* Visitor;
    MatchFinder Matchers;
    ForStmtHandler forStmtHandler;
    //NameSpaceHandler namespaceHandler;
    FunctionCallHandler functionCallHandler;
  public:
    GaloisFunctionsConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates, Rewriter &R): Instance(Instance), ParsedTemplates(ParsedTemplates), Visitor(new GaloisFunctionsVisitor(&Instance)), functionCallHandler(R) {
      Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("Galois::do_all")))).bind("galoisLoop"), &functionCallHandler);
    }

    virtual void HandleTranslationUnit(ASTContext &Context){
      Matchers.matchAST(Context);
    }
  };

  class GaloisFunctionsAction : public PluginASTAction {
  private:
    std::set<std::string> ParsedTemplates;
    Rewriter TheRewriter;
  protected:

    void EndSourceFileAction() override {
      TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
      if (!TheRewriter.overwriteChangedFiles())
      {
        llvm::outs() << "Successfully saved changes\n";
      }

    }
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
      TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
      return llvm::make_unique<GaloisFunctionsConsumer>(CI, ParsedTemplates, TheRewriter);

    }

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
      return true;
    }

  };


}

static FrontendPluginRegistry::Add<GaloisFunctionsAction> X("galois-fns", "find galois function names");
