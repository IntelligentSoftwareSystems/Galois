/*
 * Author : Gurbinder Gill (gurbinder533@gmail.com)
 */

#include <sstream>
#include <climits>
#include <vector>
#include <map>
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
      //Dec->dump();
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

    virtual bool VisitCallExpr(CallExpr* call) {
      if (call) {
        auto t = call->getType().getTypePtrOrNull();
        if (t) {
          auto func = call->getDirectCallee();
          if (func)
            if (func->getNameAsString() == "getData"){
              llvm::outs() << func->getNameInfo().getName().getAsString() << "\n";
              call->dump();
            }
        }
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

  struct getData_entry {
    string NAME;
    string TYPE;
    bool LVALUE;
  };

  class InfoClass {
    public:
      map<string, vector<getData_entry>> getData_map;
  };


  class CallExprHandler : public MatchFinder::MatchCallback {
    private:
      ASTContext* astContext;
      InfoClass *info;
    public:
      CallExprHandler(CompilerInstance*  CI, InfoClass* _info):astContext(&(CI->getASTContext())), info(_info){}

      virtual void run(const MatchFinder::MatchResult &Results) {

        auto call = Results.Nodes.getNodeAs<clang::CallExpr>("calleeName");
        auto record = Results.Nodes.getNodeAs<clang::CXXRecordDecl>("callerInStruct");

        if (call && record) {
          string STRUCT = record->getNameAsString();
          llvm::outs() << STRUCT << "\n\n";
          string NAME = "";
          string TYPE = "";
          bool LVALUE = 0;

          auto func = call->getDirectCallee();
          if (func) {
            string functionName = func->getNameAsString();
            if (functionName == "getData") {
              //record->dump();
              //call->dump();
              TYPE = call->getCallReturnType(*astContext).getAsString();
              for (auto argument : call->arguments()) {
                //argument->dump();
                auto implCast = argument->IgnoreCasts();
                //implCast->dump();
                auto declref = dyn_cast<DeclRefExpr>(implCast);
                if (declref) {
                  LVALUE = declref->isLValue();
                  NAME = declref->getNameInfo().getAsString();
                }
              }
              //call->dumpColor();

              //Storing information.
              getData_entry entry;
              entry.NAME = NAME;
              entry.TYPE = TYPE;
              entry.LVALUE = LVALUE;

              info->getData_map[STRUCT].push_back(entry);
            }
            else if (functionName == "getEdgeDst") {
              llvm::outs() << "Found getEdgeDst \n";
              auto forStatement = Results.Nodes.getNodeAs<clang::ForStmt>("forLoopName");
              forStatement->dump();
            }
          }
        }
      }
  };

  class FindOperatorHandler : public MatchFinder::MatchCallback {
    private:
      ASTContext* astContext;
    public:
      FindOperatorHandler(CompilerInstance *CI):astContext(&(CI->getASTContext())){}
      virtual void run(const MatchFinder::MatchResult &Results) {
        llvm::outs() << "I found one operator\n";

        const CXXRecordDecl* recDecl = Results.Nodes.getNodeAs<clang::CXXRecordDecl>("GaloisOP");

        if(recDecl){
          //recDecl->dump();

          for(auto method : recDecl->methods()){
            if (method->getNameAsString() == "operator()") {
              llvm::outs() << method->getNameAsString() << "\n";
              //method->dump();
              auto funcDecl = dyn_cast<clang::FunctionDecl>(method);
              llvm::outs() << funcDecl->param_size() << "\n";
              if (funcDecl->hasBody() && (funcDecl->param_size() == 1)){
                auto paramValdecl = funcDecl->getParamDecl(0); // This is GNode src to operator. 
                auto stmt = funcDecl->getBody();
                if (stmt){
                  //stmt->dumpColor();

                  for(auto stmtChild : stmt->children()) {
                    stmtChild->dump();
                  }

                }
              }
            }
          }
        }
      }
  };
  class FunctionCallHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
    public:
      FunctionCallHandler(Rewriter &rewriter) :  rewriter(rewriter){}
      virtual void run(const MatchFinder::MatchResult &Results) {
        const CallExpr* callFS = Results.Nodes.getNodeAs<clang::CallExpr>("galoisLoop");
        llvm::outs() << "It is coming here\n" << callFS->getNumArgs() << "\n";

        //llvm::outs() << callFS << "\n";
                     //<< Results.Nodes.getNodeAs<clang::DeclRefExpr>("galoisLoop")->getQualifier()->getAsNamespace(); //->getIdentifier()->getName() << "\n";


        if(callFS){

              SourceLocation ST_main = callFS->getSourceRange().getBegin();

              llvm::outs() << "Galois::do_All loop found\n";
              //callFS->dump();

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

              for(int i = 0, j = callFS->getNumArgs(); i < j; ++i){
                string str_arg;
                llvm::raw_string_ostream s(str_arg);
                callFS->getArg(i)->printPretty(s, 0, Policy);
                //llvm::outs() << "arg : " << s.str() << "\n\n";

                const CallExpr* callExpr = dyn_cast<CallExpr>(callFS->getArg(i));
                if (callExpr) {
                  const FunctionDecl* func = callExpr->getDirectCallee();
                  if (func->getNameInfo().getName().getAsString() == "read_set"){
                    llvm::outs() << "Inside arg read_set: " << i <<  s.str() << "\n\n";
                    callExpr->dump();

                    for(unsigned k = 0; k < callExpr->getNumArgs(); ++k ) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(k)->printPretty(s_sub, 0, Policy);
                      //llvm::outs() << callExpr->getArg(k)->getType()->dump() << "\n";
                      llvm::outs() << "Dumping READ set TYPE\n\n";
                      callExpr->getArg(k)->getType().getNonReferenceType().dump();
                      /*
                       * To Find the GraphNode name to be used in Syncer() struct
                       */
                      auto ptype = callExpr->getArg(k)->getType().getTypePtrOrNull(); // dyn_cast<PointerType>(callExpr->getArg(k)->getType());
                      if(ptype->isPointerType()){
                        //ptype->getPointeeType().dump();
                        auto templatePtr = dyn_cast<TemplateSpecializationType>(ptype->getPointeeType().getTypePtrOrNull());
                        if (templatePtr) {
                          templatePtr->getArg(0).getAsType().dump();
                          //templatePtr->getArg(0).getAsType().getAsString();

                          //llvm::outs() << "\t\tFINALYY : " << templatePtr->getArg(0).getAsType().getBaseTypeIdentifier()->getName() << "\n";
                          GraphNode = templatePtr->getArg(0).getAsType().getBaseTypeIdentifier()->getName();
                          //llvm::outs() << "\t\tFINALYY : " << (dyn_cast<RecordType>(templatePtr->getArg(0).getAsType().getTypePtrOrNull()))->getDecl()->getAsString();
                          //llvm::outs() << "\t\tFINALYY : " << s_sub.str() << "\n";
                          //llvm::outs() << "TEMPLATE ARGUEMENTS : " << templatePtr->getNumArgs() << "\n";
                          //llvm::outs() << "Ptype : " << ptype->isPointerType() << "\n";
                        }
                      }
                      //callExpr->getArg(k)->desugar();

                      //const PointerType* pointerTy = dyn_cast<PointerType>(callExpr->getArg(k))
                      llvm::outs() << "TYPE = > " << callExpr->getArg(k)->getType().getAsString()<<"\n";

                      /*
                       * Removing Quotes. Find better solution.
                       */
                      string temp_str = s_sub.str();
                      if(temp_str.front() == '"'){
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }

                      string argType = callExpr->getArg(k)->getType().getAsString();
                      read_set_vec.push_back(make_pair(argType, temp_str));
                      llvm::outs() << "\t Sub arg : " << s_sub.str() << "\n\n";

                    }
                  }
                  else if (func->getNameInfo().getName().getAsString() == "write_set"){
                    llvm::outs() << "Inside arg write_set: " << i <<  s.str() << "\n\n";
                    callExpr->dump();

                    for(unsigned k = 0; k < callExpr->getNumArgs(); ++k ) {
                      string str_sub_arg;
                      llvm::raw_string_ostream s_sub(str_sub_arg);
                      callExpr->getArg(k)->printPretty(s_sub, 0, Policy);
                      llvm::outs() << "Dumping WRITE set TYPE\n\n"; 
                      callExpr->getArg(k)->getType()->dump();
                      /*
                       * To Find the GraphNode name to be used in Syncer() struct
                       */
                      auto ptype = callExpr->getArg(k)->getType().getTypePtrOrNull()->getUnqualifiedDesugaredType(); // dyn_cast<PointerType>(callExpr->getArg(k)->getType());
                      //ptype->dump();
                      if(ptype->isPointerType() && (k == 0)){
                        ptype->getPointeeType().getTypePtrOrNull()->getUnqualifiedDesugaredType()->dump();
                        //auto templatePtr = dyn_cast<TemplateSpecializationType>(ptype->getPointeeType().getTypePtrOrNull()->getUnqualifiedDesugaredType());
                        auto templatePtr = ptype->getPointeeType().getTypePtrOrNull()->getAs<TemplateSpecializationType>();
                        if (templatePtr) {
                          templatePtr->getArg(0).getAsType().dump();
                          string GraphNodeTy = templatePtr->getArg(0).getAsType().getBaseTypeIdentifier()->getName();
                          write_setGNode_vec.push_back(GraphNodeTy);
                        }
                      }
                      string argType = callExpr->getArg(k)->getType().getAsString();
                      /*
                       * Removing Quotes. Find better solution.
                       */
                      string temp_str = s_sub.str();
                      if(temp_str.front() == '"'){
                        temp_str.erase(0,1);
                        temp_str.erase(temp_str.size()-1);
                      }
                      if(temp_str.front() == '&'){
                        temp_str.erase(0,1);
                      }
                      write_set_vec.push_back(make_pair(argType,temp_str));
                      write_setNum++;
                      llvm::outs() << "\t Sub arg : " << s_sub.str() << "\n\n";

                    }
                  }
                }
                llvm::outs() <<"\n\n";
              }

              // Adding Syncer struct for synchronization.
              stringstream SSSyncer;
              for (unsigned i = 0; i < write_setGNode_vec.size(); ++i) {
                SSSyncer << " struct Syncer_" << i << " {\n"
                      << "\tstatic " << write_set_vec[4*i+3].first <<" extract( const " << write_setGNode_vec[i] <<"& node)" << "{ return " << "node." << write_set_vec[4*i+1].second <<  "; }\n"
                      <<"\tstatic void reduce (" << write_setGNode_vec[i] << "& node, " << write_set_vec[4*i+3].first << " y) { node." << write_set_vec[4*i+1].second << " = " << write_set_vec[4*i+2].second << " ( node." << write_set_vec[4*i+1].second << ", y);}\n" 
                      <<"\tstatic void reset (" << write_setGNode_vec[i] << "& node ) { node." << write_set_vec[4*i+1].second << " = " << write_set_vec[4*i+3].second << "; }\n" 
                      <<"\ttypedef " << write_set_vec[4*i+3].first << " ValTy;\n"
                      <<"};\n";

                rewriter.InsertText(ST_main, SSSyncer.str(), true, true);
                SSSyncer.str(string());
                SSSyncer.clear();
              }

              //Add text before
              stringstream SSBefore;
              SSBefore << "/* Galois:: do_all before  */\n";
              SourceLocation ST = callFS->getSourceRange().getBegin();
              rewriter.InsertText(ST, SSBefore.str(), true, true);

              // Adding Syn calls for read set
             /* for (unsigned i = 0; i < read_set_vec.size(); i+=2) {
                SSBefore.str(string());
                SSBefore.clear();
                SSBefore << "Syn(" << read_set_vec[i].second << " , " << read_set_vec[i+1].second << ")\n";
                rewriter.InsertText(ST, SSBefore.str(), true, true);
              }*/

              //Add text after
              stringstream SSAfter;
              SSAfter << "\n/* Galois:: do_all After */\n";
              ST = callFS->getSourceRange().getEnd().getLocWithOffset(2);
              rewriter.InsertText(ST, SSAfter.str(), true, true);

              // Adding Syn calls for write set
              for (unsigned i = 0; i < write_setGNode_vec.size(); i++) {
                SSAfter.str(string());
                SSAfter.clear();
                SSAfter << write_set_vec[i*4].second << ".sync_push<Syncer_" << i << ">" <<"();\n";
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
    MatchFinder Matchers, Matchers_op;
    FunctionCallHandler functionCallHandler;
    FindOperatorHandler findOperator;
    CallExprHandler callExprHandler;
    InfoClass info;
  public:
    GaloisFunctionsConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates, Rewriter &R): Instance(Instance), ParsedTemplates(ParsedTemplates), Visitor(new GaloisFunctionsVisitor(&Instance)), functionCallHandler(R), findOperator(&Instance), callExprHandler(&Instance, &info) {
      //Matchers.addMatcher(unless(isExpansionInSystemHeader(callExpr(callee(functionDecl(hasName("for_each")))).bind("galoisLoop"))), &namespaceHandler);
      //Matchers.addMatcher(callExpr(callee(functionDecl(hasName("Galois::do_all"))),unless(isExpansionInSystemHeader())).bind("galoisLoop"), &functionCallHandler);

      
      //Matchers.addMatcher(recordDecl(hasName("initializeGraph")).bind("op"), &findOperator);
      //Matchers.addMatcher(recordDecl(has(methodDecl(hasName("operator()")), hasParameter(0, hasType(varDecl())))).bind("op"), &findOperator);
      //Matchers.addMatcher(methodDecl(returns(asString("void"))).bind("op"), &findOperator); //****working
      //Matchers.addMatcher(recordDecl(hasMethod(hasAnyParameter(hasName("x")))).bind("op"), &findOperator); // *** working
      //Matchers.addMatcher(recordDecl(hasMethod(hasParameter(1, parmVarDecl(hasType(isInteger()))))).bind("op"), &findOperator); // *** working

      /**************** Found it ********************/
      //Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData"))), hasAncestor(binaryOperator(hasOperatorName("=")).bind("assignment"))).bind("getData"), &callExprHandler); //*** WORKS
      //Matchers.addMatcher(recordDecl(hasDescendant(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData")))).bind("getData"))).bind("getDataInStruct"), &callExprHandler); //**** works

      /** For matching the nodes in AST with getData function call ***/
      Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData"))), hasAncestor(recordDecl().bind("callerInStruct"))).bind("calleeName"), &callExprHandler);

      /** For matching the nodes in AST with getEdgeDst function call ***/
      StatementMatcher Edge_beginCallMatcher = memberCallExpr(argumentCountIs(1), callee(methodDecl(hasName("edge_begin"))));
      DeclarationMatcher InitDeclMatcher = varDecl(hasInitializer(anything())).bind("ForInitVarName");
      DeclarationMatcher EndDeclMatcher = varDecl(hasInitializer(anything())).bind("ForEndVarName");
      StatementMatcher Edge_endCallMatcher = memberCallExpr(argumentCountIs(1), callee(methodDecl(hasName("edge_end"))));

      StatementMatcher IteratorBoundMatcher = expr(anyOf(ignoringParenImpCasts(declRefExpr(to(
                                                  varDecl()))),
                                                      ignoringParenImpCasts(
                                                            expr(Edge_endCallMatcher)),
                                                      materializeTemporaryExpr(ignoringParenImpCasts(
                                                          expr(Edge_endCallMatcher)))));

      StatementMatcher IteratorComparisonMatcher = expr(ignoringParenImpCasts(declRefExpr(to(
                                                      varDecl()))));

      StatementMatcher OverloadedNEQMatcher = operatorCallExpr(
                                                hasOverloadedOperatorName("!="),
                                                argumentCountIs(2),
                                                hasArgument(0, IteratorComparisonMatcher),
                                                hasArgument(1,IteratorBoundMatcher));

      static const TypeMatcher AnyType = anything();
      StatementMatcher EdgeForLoopMatcher = forStmt(hasLoopInit(anyOf(
                                                  declStmt(declCountIs(2),
                                                              containsDeclaration(0, InitDeclMatcher),
                                                              containsDeclaration(1, EndDeclMatcher)),
                                                  declStmt(hasSingleDecl(InitDeclMatcher)))),
                                              hasCondition(anyOf(
                                                  binaryOperator(hasOperatorName("!="),
                                                                  hasLHS(IteratorComparisonMatcher),
                                                                  hasRHS(IteratorBoundMatcher)),
                                                  OverloadedNEQMatcher)),
                                              hasIncrement(anyOf(
                                                            unaryOperator(hasOperatorName("++"),
                                                                              hasUnaryOperand(declRefExpr(to(
                                                                                    varDecl(hasType(pointsTo(AnyType))))))),
                                                            operatorCallExpr(
                                                                hasOverloadedOperatorName("++"),
                                                                hasArgument(0, declRefExpr(to(
                                                                      varDecl()))))))).bind("forLoopName");


      Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getEdgeDst"))), hasAncestor(recordDecl().bind("callerInStruct")), hasAncestor(EdgeForLoopMatcher)).bind("calleeName"), &callExprHandler);

      //Matchers_op.addMatcher(recordDecl(isExpansionInMainFile(), hasMethod(hasName("operator()")), hasMethod(returns(asString("void"))), hasMethod(hasParameter(0, parmVarDecl(hasType(isInteger()))))).bind("GaloisOP"), &findOperator); // *** working

      //Matchers.addMatcher(methodDecl(hasAnyParameter(hasName("x"))).bind("op"), &findOperator); //***** works
      //Matchers.addMatcher(declRefExpr(to(functionDecl(hasName("do_all"))),unless(isExpansionInSystemHeader())).bind("galoisLoop"), &functionCallHandler);
      //Matchers.addMatcher(recordDecl(decl().bind("id"), hasName("::Galois")), &recordHandler);
      //Matchers.addMatcher(nestedNameSpecifier(specifiesNamespace(hasName("for_each"))).bind("galoisLoop"), &namespaceHandler);
      //Matchers.addMatcher(forStmt(nestedNameSpecifier(specifiesNamespace(hasName("Galois"))).bind("galoisLoop")), &namespaceHandler);
      //Matchers.addMatcher(elaboratedType(hasQualifier(hasPrefix(specifiesNamespace(hasName("Galois"))))).bind("galoisLoop"), &namespaceHandler);
      //Matchers.addMatcher(forStmt( hasLoopInit(declStmt(hasSingleDecl(varDecl(
                      //hasInitializer(integerLiteral(equals(0)))))))).bind("forLoop"), &forStmtHandler);
    }

    virtual void HandleTranslationUnit(ASTContext &Context){
      //Visitor->TraverseDecl(Context.getTranslationUnitDecl());
      Matchers.matchAST(Context);
      llvm::outs() << " MAP SIZE : " << info.getData_map.size() << "\n";
      Matchers_op.matchAST(Context);
    }
  };

  class GaloisFunctionsAction : public PluginASTAction {
  private:
    std::set<std::string> ParsedTemplates;
    Rewriter TheRewriter;
  protected:

    void EndSourceFileAction() override {
      /*
      TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
      if (!TheRewriter.overwriteChangedFiles())
      {
        llvm::outs() << "Successfully saved changes\n";
      }
      */
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

static FrontendPluginRegistry::Add<GaloisFunctionsAction> X("galois-analysis", "find galois function names");
