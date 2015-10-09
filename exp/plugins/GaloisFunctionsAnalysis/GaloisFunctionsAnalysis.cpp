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

  /* Convert double to string with specified number of places after the decimal
   *    and left padding. */
  template<typename ty>
  std::string prd(const ty x, const int width) {
      stringstream ss;
      ss << fixed << right;
      ss.fill(' ');        // fill space around displayed #
      ss.width(width);     // set  width around displayed #
      ss << x;
      return ss.str();
  }



/*! Center-aligns string within a field of width w. Pads with blank spaces
 *     to enforce alignment. */
std::string center(const string s, const int w) {
  stringstream ss, spaces;
  int padding = w - s.size();                 // count excess room to pad
  for(int i=0; i<padding/2; ++i)
    spaces << " ";
  ss << spaces.str() << s << spaces.str();
  if(padding>0 && padding%2!=0)               // if odd #, add 1 space
    ss << " ";
  return ss.str();
}
std::string center(const bool s, const int w) {
  stringstream ss, spaces;
  int padding = w - 1;                 // count excess room to pad
  for(int i=0; i<padding/2; ++i)
    spaces << " ";
  ss << spaces.str() << s << spaces.str();
  if(padding>0 && padding%2!=0)               // if odd #, add 1 space
    ss << " ";
  return ss.str();
}




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
    string VAR_NAME;
    string VAR_TYPE;
    string SRC_NAME;
    string RW_STATUS;
    bool IS_REFERENCE;
    bool IS_REFERENCED;
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
        auto binaryOP = Results.Nodes.getNodeAs<clang::Stmt>("getDataAssignment");

        auto lhs_ref = Results.Nodes.getNodeAs<clang::Stmt>("LHSRefVariable");
        auto lhs_nonref = Results.Nodes.getNodeAs<clang::Stmt>("LHSNonRefVariable");
        auto vardecl = Results.Nodes.getNodeAs<clang::Stmt>("variableDecl_getData");
        auto NonRefVardecl = Results.Nodes.getNodeAs<clang::Stmt>("nonRefVariableDecl_getData");

        //record->dump();
        if (call && record) {

          getData_entry DataEntry_obj;
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
              if (vardecl) {
                llvm::outs() << "Reference Variable\n";
                auto varName = Results.Nodes.getNodeAs<clang::VarDecl>("referencedVarName");
                //varName->dump();
                DataEntry_obj.VAR_NAME = varName->getNameAsString();
                DataEntry_obj.VAR_TYPE = varName->getType().getAsString();
                DataEntry_obj.IS_REFERENCED = varName->isReferenced();
                DataEntry_obj.IS_REFERENCE = true;

                llvm::outs() << " IS REFERENCED = > " << varName->isReferenced() << "\n";
                llvm::outs() << varName->getNameAsString() << "\n";
                llvm::outs() << varName->getType().getAsString() << "\n";
              }
              else if (NonRefVardecl) {
                llvm::outs() << "Non Reference variable \n";
                auto varName = Results.Nodes.getNodeAs<clang::VarDecl>("nonReferencedVarName");
                //varName->dump();

                DataEntry_obj.VAR_NAME = varName->getNameAsString();
                DataEntry_obj.VAR_TYPE = varName->getType().getAsString();
                DataEntry_obj.IS_REFERENCED = varName->isReferenced();
                DataEntry_obj.IS_REFERENCE = false;

                llvm::outs() << " IS REFERENCED = > " << varName->isReferenced() << "\n";
                llvm::outs() << varName->getNameAsString() << "\n";
                llvm::outs() << varName->getType().getAsString() << "\n";
              }
             /* if (lhs_ref) {
                llvm::outs() << "REFERENCE VARIABLE FOUND\n";
                lhs_ref->dump();
              }
              if (lhs_nonref) {
                llvm::outs() << "NON REFERENCE VARIABLE FOUND\n";
                //lhs_nonref->dump();
              }
              */
              else if (binaryOP)
              {
                llvm::outs() << "Assignment Found\n";
                //binaryOP->dump();

                auto lhs = Results.Nodes.getNodeAs<clang::Stmt>("LHSNonRefVariable");
                if(lhs)
                  llvm::outs() << "NON REFERENCE VARIABLE FOUND\n";
                //lhs->dump();

              }

              TYPE = call->getCallReturnType(*astContext).getAsString();
              for (auto argument : call->arguments()) {
                //argument->dump();
                auto implCast = argument->IgnoreCasts();
                //implCast->dump();
                auto declref = dyn_cast<DeclRefExpr>(implCast);
                if (declref) {
                  //LVALUE = declref->isLValue();
                  DataEntry_obj.SRC_NAME = declref->getNameInfo().getAsString();
                  llvm::outs() << " I NEED THIS : " << NAME << "\n";
                }
              }
              //call->dumpColor();

              //Storing information.
              info->getData_map[STRUCT].push_back(DataEntry_obj);
            }
            else if (functionName == "getEdgeDst") {
              llvm::outs() << "Found getEdgeDst \n";
              auto forStatement = Results.Nodes.getNodeAs<clang::ForStmt>("forLoopName");
              //forStatement->dump();
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
      static const TypeMatcher AnyType = anything();
      StatementMatcher GetDataMatcher = callExpr(argumentCountIs(1), callee(methodDecl(hasName("getData"))));
      StatementMatcher LHSRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl(hasType(references(AnyType))))))).bind("LHSRefVariable");

      StatementMatcher LHSNonRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl())))).bind("LHSNonRefVariable");

      Matchers.addMatcher(callExpr(
                                          isExpansionInMainFile(),
                                          callee(functionDecl(hasName("getData"))),
                                          hasAncestor(recordDecl().bind("callerInStruct")),
                                          anyOf(
                                                hasAncestor(binaryOperator(hasOperatorName("="),
                                                                              hasLHS(anyOf(
                                                                                        LHSRefVariable,
                                                                                        LHSNonRefVariable))).bind("getDataAssignment")
                                                              ),
                                                hasAncestor(declStmt(hasSingleDecl(varDecl(hasType(references(AnyType))).bind("referencedVarName"))).bind("variableDecl_getData")),
                                                hasAncestor(declStmt(hasSingleDecl(varDecl(unless(hasType(references(AnyType)))).bind("nonReferencedVarName"))).bind("nonRefVariableDecl_getData"))
                                            )
                                     ).bind("calleeName"), &callExprHandler);

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
      for (auto i : info.getData_map)
      {
        llvm::outs() << i.first << " has " << i.second.size() << " references of getData \n";

        int width = 20;
        llvm::outs() << center(string("VAR_NAME"), width) << "|"
                     <<  center(string("VAR_TYPE"), width) << "|"
                     << center(string("SRC_NAME"), width) << "|"
                     << center(string("IS_REFERENCE"), width) << "|"
                     << center(string("IS_REFERENCED"), width) << "||"
                     << center(string("RW_STATUS"), width) << "\n";
        llvm::outs() << std::string(width*6 + 2*6, '-') << "\n";
        for( auto j : i.second) {
          llvm::outs() << center(j.VAR_NAME,  width) << "|"
                       << center(j.VAR_TYPE,  width) << "|"
                       << center(j.SRC_NAME,  width) << "|"
                       << center(j.IS_REFERENCE,  width) << "|"
                       << center(j.IS_REFERENCED,  width) << "|"
                       << center(j.RW_STATUS,  width) << "\n";
        }
      }
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
