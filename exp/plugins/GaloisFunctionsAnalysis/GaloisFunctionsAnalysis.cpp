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

static const TypeMatcher AnyType = anything();
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

/** Function to split a string with a given delimiter **/
void split(const string& s, char delim, std::vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
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


  struct getData_entry {
    string VAR_NAME;
    string VAR_TYPE;
    string SRC_NAME;
    string RW_STATUS;
    bool IS_REFERENCE;
    bool IS_REFERENCED;
    string GRAPH_NAME;
  };

  struct Graph_entry {
    string GRAPH_NAME;
    string NODE_TYPE;
    string EDGE_TYPE;
  };

  struct NodeField_entry {
    string VAR_NAME;
    string VAR_TYPE;
    string FIELD_NAME;
    string NODE_NAME;
    string NODE_TYPE;
    string RW_STATUS;
    bool IS_REFERENCE;
    bool IS_REFERENCED;
    string GRAPH_NAME;
    string RESET_VALTYPE;
    string RESET_VAL;
  };

  struct ReductionOps_entry {
    string NODE_TYPE;
    string FIELD_NAME;
    string FIELD_TYPE;
    string VAL_TYPE;
    string OPERATION_EXPR;
    string RESETVAL_EXPR;
    string GRAPH_NAME;
    string SYNC_TYPE;
  };

  class InfoClass {
    public:
      map<string, vector<Graph_entry>> graphs_map;
      map<string, vector<getData_entry>> getData_map;
      map<string, vector<getData_entry>> edgeData_map;
      map<string, vector<NodeField_entry>> fieldData_map;
      map<string, vector<ReductionOps_entry>> reductionOps_map;
  };


class TypedefHandler :  public MatchFinder::MatchCallback {
  InfoClass* info;
  public:
    TypedefHandler(InfoClass* _info):info(_info){}
    virtual void run(const MatchFinder::MatchResult & Results) {
      auto typedefDecl = Results.Nodes.getNodeAs<clang::TypeDecl>("typedefDecl");

      //llvm::outs() << "GlobalID : " << typedefDecl->getGlobalID() << "\n";
      string Graph_name = typedefDecl->getNameAsString();
      llvm::outs() << "--->" << Graph_name << "\n";
      auto templatePtr = typedefDecl->getTypeForDecl()->getAs<TemplateSpecializationType>();
      if (templatePtr) {
        Graph_entry g_entry;
        /**We only need NopeType and EdgeType**/
        //templatePtr->getArg(0).getAsType().dump();
        llvm::outs() << " ARGUMETNS : " << templatePtr->getNumArgs() << "\n";
        g_entry.NODE_TYPE = templatePtr->getArg(0).getAsType().getBaseTypeIdentifier()->getName();
        templatePtr->getArg(1).getAsType().dump();
        g_entry.EDGE_TYPE = templatePtr->getArg(1).getAsType().getAsString();


        info->graphs_map[Graph_name].push_back(g_entry);
        llvm::outs() << "node is :" << g_entry.NODE_TYPE << ", edge is : " << g_entry.EDGE_TYPE << "\n";
      }
    }
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

        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        //record->dump();
        if (call && record) {

          getData_entry DataEntry_obj;
          string STRUCT = record->getNameAsString();
          llvm::outs() << STRUCT << "\n\n";
          /* if (STRUCT == "InitializeGraph") {
             record->dump();
             }*/
          string NAME = "";
          string TYPE = "";

          /** To get the graph name on which this getData is called **/
          auto memExpr_graph = Results.Nodes.getNodeAs<clang::Stmt>("getData_memExpr_graph");
          string str_field;
          llvm::raw_string_ostream s(str_field);
          /** Returns something like this->graphName **/
          memExpr_graph->printPretty(s, 0, Policy);
          DataEntry_obj.GRAPH_NAME = s.str();

          auto func = call->getDirectCallee();
          if (func) {
            string functionName = func->getNameAsString();
            if (functionName == "getData") {

              auto binaryOP = Results.Nodes.getNodeAs<clang::Stmt>("getDataAssignment");
              auto refVardecl_stmt = Results.Nodes.getNodeAs<clang::Stmt>("refVariableDecl_getData");
              auto nonRefVardecl_stmt = Results.Nodes.getNodeAs<clang::Stmt>("nonRefVariableDecl_getData");


              auto varDecl_stmt_getData = (refVardecl_stmt)?refVardecl_stmt:nonRefVardecl_stmt;
              DataEntry_obj.IS_REFERENCE = (refVardecl_stmt)?true:false;

              //record->dump();
              //call->dump();
              if (varDecl_stmt_getData) {

                auto varDecl_getData = Results.Nodes.getNodeAs<clang::VarDecl>("getData_varName");
                DataEntry_obj.VAR_NAME = varDecl_getData->getNameAsString();
                DataEntry_obj.VAR_TYPE = varDecl_getData->getType().getAsString();
                DataEntry_obj.IS_REFERENCED = varDecl_getData->isReferenced();

                llvm::outs() << " IS REFERENCED = > " << varDecl_getData->isReferenced() << "\n";
                llvm::outs() << varDecl_getData->getNameAsString() << "\n";
                llvm::outs() << varDecl_getData->getType().getAsString() << "\n";
              }
              else if (binaryOP)
              {
                llvm::outs() << "Assignment Found\n";
                //binaryOP->dump();

                auto lhs_ref = Results.Nodes.getNodeAs<clang::Stmt>("LHSRefVariable");
                auto lhs_nonref = Results.Nodes.getNodeAs<clang::Stmt>("LHSNonRefVariable");
                if(lhs_nonref)
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
                  DataEntry_obj.SRC_NAME = declref->getNameInfo().getAsString();
                  llvm::outs() << " I NEED THIS : " << NAME << "\n";
                }
              }
              //call->dumpColor();

              //Storing information.
              info->getData_map[STRUCT].push_back(DataEntry_obj);
            }
            else if (functionName == "getEdgeDst") {
              /** This is inside for loop which has getData as well. **/
              llvm::outs() << "Found getEdgeDst \n";
              auto forStatement = Results.Nodes.getNodeAs<clang::ForStmt>("forLoopName");
              auto varDecl_edgeDst = Results.Nodes.getNodeAs<clang::VarDecl>("varDeclName");
              //varDecl_edgeDst->dump();
              string varName_edgeDst = varDecl_edgeDst->getNameAsString();
              llvm::outs() << varName_edgeDst << "\n";

              auto binaryOP = Results.Nodes.getNodeAs<clang::Stmt>("getDataAssignment");
              auto refVardecl_stmt = Results.Nodes.getNodeAs<clang::Stmt>("refVariableDecl_getData");
              auto nonRefVardecl_stmt = Results.Nodes.getNodeAs<clang::Stmt>("nonRefVariableDecl_getData");

              auto varDecl_stmt_getData = (refVardecl_stmt)?refVardecl_stmt:nonRefVardecl_stmt;
              DataEntry_obj.IS_REFERENCE = (refVardecl_stmt)?true:false;

              auto call_getData = Results.Nodes.getNodeAs<clang::CallExpr>("calleeName_getEdgeDst");

              string argument_getData;
              for (auto argument : call_getData->arguments()) {
                auto implCast = argument->IgnoreCasts();
                auto declref = dyn_cast<DeclRefExpr>(implCast);
                if (declref) {
                  argument_getData = declref->getNameInfo().getAsString();
                  DataEntry_obj.SRC_NAME = argument_getData;
                  llvm::outs() << " Argument to getData : " << argument_getData << "\n";
                }
              }

              /**To make sure that return value from getEdgeDst is being used in getData i.e edges are being used**/
              if (varName_edgeDst == argument_getData)
              {


                if(varDecl_stmt_getData) {
                  auto varDecl_getData = Results.Nodes.getNodeAs<clang::VarDecl>("getData_varName");
                  llvm::outs() << "varname getData : "  << varDecl_getData->getNameAsString() << "\n";
                  DataEntry_obj.VAR_NAME = varDecl_getData->getNameAsString();
                  DataEntry_obj.VAR_TYPE = varDecl_getData->getType().getAsString();
                  DataEntry_obj.IS_REFERENCED = varDecl_getData->isReferenced();

                }
                //call->dumpColor();

                info->edgeData_map[STRUCT].push_back(DataEntry_obj);

              }
              else {
                llvm::outs() << " Variable Declaration with getEdgeDst != variable used in getData!!\n";
                //llvm::abort();
              }
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
    InfoClass* info;
  public:
    FunctionCallHandler(Rewriter &rewriter, InfoClass* _info ) :  rewriter(rewriter), info(_info){}
    virtual void run(const MatchFinder::MatchResult &Results) {
      const CallExpr* callFS = Results.Nodes.getNodeAs<clang::CallExpr>("galoisLoop");


      if(callFS){
        auto callRecordDecl = Results.Nodes.getNodeAs<clang::CXXRecordDecl>("do_all_recordDecl");
        string struct_name = callRecordDecl->getNameAsString();

        SourceLocation ST_main = callFS->getSourceRange().getBegin();
        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        //Add text before
        //stringstream SSBefore;
        //SSBefore << "/* Galois:: do_all before  */\n";
        //SourceLocation ST = callFS->getSourceRange().getBegin();
        //rewriter.InsertText(ST, "hello", true, true);

        //Add text after
        for(auto i : info->reductionOps_map){
          if((i.first == struct_name) &&  (i.second.size() > 0)){
            for(auto j : i.second){
              stringstream SSAfter;
              if(j.SYNC_TYPE == "sync_push")
                SSAfter << ", Galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \"" << j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\" , \"" << j.FIELD_NAME << "\", \"" << j.VAL_TYPE << "\" , \"" << j.OPERATION_EXPR << "\",  \"" << j.RESETVAL_EXPR << "\")";
              else if(j.SYNC_TYPE == "sync_pull")
                SSAfter << ", Galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \""<< j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\", \"" << j.FIELD_NAME << "\" , \"" << j.VAL_TYPE << "\")";

              SourceLocation ST = callFS->getSourceRange().getEnd().getLocWithOffset(0);
              rewriter.InsertText(ST, SSAfter.str(), true, true);
            }
          }
        }
      }
    }
};

class FindingFieldHandler : public MatchFinder::MatchCallback {
  private:
    ASTContext* astContext;
    InfoClass *info;
    public:
      FindingFieldHandler(CompilerInstance*  CI, InfoClass* _info):astContext(&(CI->getASTContext())), info(_info){}
      virtual void run(const MatchFinder::MatchResult &Results) {

        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        for(auto i : info->getData_map) {
          for(auto j : i.second) {
            //llvm::outs() << j.VAR_NAME << "\n";
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME + "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;
            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;


            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);
              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);


              /**Figure out variable type to set the reset value **/
              auto memExpr = Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);

              //memExpr->dump();
              if(memExpr){
                auto pType = memExpr->getType();
                auto templatePtr = pType->getAs<TemplateSpecializationType>();
                string Ty;
                /** It is complex type e.g atomic<int>**/
                if (templatePtr) {
                  templatePtr->getArg(0).getAsType().dump();
                  Ty = templatePtr->getArg(0).getAsType().getAsString();
                  //llvm::outs() << "TYPE FOUND ---->" << Ty << "\n";
                }
                /** It is a simple builtIn type **/
                else {
                  Ty = pType->getLocallyUnqualifiedSingleStepDesugaredType().getAsString();
                }

                ReductionOps_entry reduceOP_entry;
                NodeField_entry field_entry;
                field_entry.RESET_VALTYPE = reduceOP_entry.VAL_TYPE = Ty;
                field_entry.NODE_TYPE = j.VAR_TYPE;
                reduceOP_entry.NODE_TYPE = j.VAR_TYPE;
                reduceOP_entry.FIELD_TYPE = j.VAR_TYPE;

                auto memExpr_stmt = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                string str_field;
                llvm::raw_string_ostream s(str_field);
                /** Returns something like snode.field_name.operator **/
                memExpr_stmt->printPretty(s, 0, Policy);

                /** Split string at delims to get the field name **/
                char delim = '.';
                vector<string> elems;
                split(s.str(), delim, elems);
                //llvm::outs() << " ===> " << elems[1] << "\n";
                field_entry.NODE_NAME = elems[0];

                field_entry.FIELD_NAME = reduceOP_entry.FIELD_NAME = elems[1];
                field_entry.GRAPH_NAME = reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME;
                if(!field && (assignplusOP || plusOP || assignmentOP || atomicAdd_op)) {
                  reduceOP_entry.SYNC_TYPE = "sync_pull";
                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                }

                if(assignplusOP) {
                  //assignplusOP->dump();
                  field_entry.VAR_NAME = "+";
                  field_entry.VAR_TYPE = "+";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";

                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
                else if(plusOP) {
                  field_entry.VAR_NAME = "+=";
                  field_entry.VAR_TYPE = "+=";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";

                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;

                }
                /** If this is a an expression like nodeData.fieldName = value **/
                else if(assignmentOP) {

                  field_entry.VAR_NAME = "DirectAssignment";
                  field_entry.VAR_TYPE = "DirectAssignment";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";

                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
                else if(atomicAdd_op){
                  atomicAdd_op->dump();
                  field_entry.VAR_NAME = "AtomicAdd";
                  field_entry.VAR_TYPE = "AtomicAdd";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";

                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;

                }
                else if(field){

                  field_entry.VAR_NAME = field->getNameAsString();
                  field_entry.VAR_TYPE = field->getType().getAsString();
                  field_entry.IS_REFERENCED = field->isReferenced();
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";
                  info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
              }
            }
          }
        }
      }
};

class FindingFieldInsideForLoopHandler : public MatchFinder::MatchCallback {
  private:
      ASTContext* astContext;
      InfoClass *info;
    public:
      FindingFieldInsideForLoopHandler(CompilerInstance*  CI, InfoClass* _info):astContext(&(CI->getASTContext())), info(_info){}
      virtual void run(const MatchFinder::MatchResult &Results) {

        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        
        for(auto i : info->edgeData_map) {
          for(auto j : i.second) {
            //llvm::outs() << j.VAR_NAME << "\n";
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME + "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_ifMin = "ifMin_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinRHS = "ifMinRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignment = "Cond_equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignmentRHS = "Cond_equalOpRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSmemExpr = "Cond_RHSmemExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSVarDecl = "Cond_RHSVarDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_whileCAS = "whileCAS_" + j.VAR_NAME + "_" + i.first;
            string str_whileCAS_RHS = "whileCAS_RHS" + j.VAR_NAME + "_" + i.first;

            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;

            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);

              auto ifMinOp = Results.Nodes.getNodeAs<clang::Stmt>(str_ifMin);
              auto whileCAS_op = Results.Nodes.getNodeAs<clang::Stmt>(str_whileCAS);

              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);

              /**Figure out variable type to set the reset value **/
              auto memExpr = Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);

              //memExpr->dump();
              if(memExpr){
                auto pType = memExpr->getType();
                auto templatePtr = pType->getAs<TemplateSpecializationType>();
                string Ty;
                /** It is complex type e.g atomic<int>**/
                if (templatePtr) {
                  templatePtr->getArg(0).getAsType().dump();
                  Ty = templatePtr->getArg(0).getAsType().getAsString();
                  //llvm::outs() << "TYPE FOUND ---->" << Ty << "\n";
                }
                /** It is a simple builtIn type **/
                else {
                  Ty = pType->getLocallyUnqualifiedSingleStepDesugaredType().getAsString();
                }

                NodeField_entry field_entry;
                ReductionOps_entry reduceOP_entry;
                field_entry.NODE_TYPE = j.VAR_TYPE;
                reduceOP_entry.NODE_TYPE = j.VAR_TYPE;
                reduceOP_entry.FIELD_TYPE = j.VAR_TYPE;

                field_entry.VAR_TYPE = field_entry.RESET_VALTYPE = reduceOP_entry.VAL_TYPE = Ty;

                auto memExpr_stmt = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                string str_field;
                llvm::raw_string_ostream s(str_field);
                /** Returns something like snode.field_name.operator **/
                memExpr_stmt->printPretty(s, 0, Policy);

                /** Split string at delims to get the field name **/
                char delim = '.';
                vector<string> elems;
                split(s.str(), delim, elems);

                field_entry.NODE_NAME = elems[0];
                field_entry.FIELD_NAME = reduceOP_entry.FIELD_NAME = elems[1];
                field_entry.GRAPH_NAME = reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME;
                reduceOP_entry.SYNC_TYPE = "sync_push";



                if(assignplusOP) {
                  //assignplusOP->dump();
                  field_entry.VAR_NAME = "+";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";

                  string reduceOP = "node." + field_entry.FIELD_NAME + "= node." + field_entry.FIELD_NAME + " + y ;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
                else if(plusOP) {
                  field_entry.VAR_NAME = "+=";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";

                  string reduceOP = "node." + field_entry.FIELD_NAME + "+= y;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;

                }
                /** If this is a an expression like nodeData.fieldName = value **/
                else if(assignmentOP) {

                  field_entry.VAR_NAME = "DirectAssignment";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";


                  string reduceOP = "node." + field_entry.FIELD_NAME + "= y;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
                else if(ifMinOp){
                  //ifMinOp->dump();
                  auto ifMinRHS = Results.Nodes.getNodeAs<clang::Stmt>(str_ifMinRHS);
                  auto condAssignLHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  auto condAssignRHS = Results.Nodes.getNodeAs<clang::Stmt>(str_Cond_RHSmemExpr);
                  auto varAssignRHS = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
                  auto varCondRHS = Results.Nodes.getNodeAs<clang::VarDecl>(str_Cond_RHSVarDecl);
                  //ifMinRHS->dump();
                  //condAssignLHS->dump();

                  // It is min operation.
                  if(varCondRHS == varAssignRHS) {
                    //string reduceOP = "if(node." + field_entry.FIELD_NAME + " > y) { node." + field_entry.FIELD_NAME + " = y;}";
                    string reduceOP = "{  Galois::min(node." + field_entry.FIELD_NAME + ", y);}";
                    reduceOP_entry.OPERATION_EXPR = reduceOP;
                    string resetVal = "std::numeric_limits<" + field_entry.RESET_VALTYPE + ">::max()";
                    reduceOP_entry.RESETVAL_EXPR = resetVal;
                    varAssignRHS->dump();

                    info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  }
                  break;

                }
                else if(whileCAS_op){
                  auto whileCAS_LHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  //whileCAS_LHS->dump();
                  string reduceOP = "{Galois::min(node." + field_entry.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "std::numeric_limits<" + field_entry.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //whileCAS_LHS->dump();
                  break;
                }

                else if(atomicAdd_op){
                  string reduceOP = "{ Galois::atomicAdd(node." + field_entry.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(field){

                  field_entry.VAR_NAME = field->getNameAsString();
                  field_entry.VAR_TYPE = field->getType().getAsString();
                  field_entry.IS_REFERENCED = field->isReferenced();
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";
                  info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
              }
            }
          }
        }
      }
};

class FieldUsedInForLoop : public MatchFinder::MatchCallback {
  private:
      ASTContext* astContext;
      InfoClass *info;
    public:
      FieldUsedInForLoop(CompilerInstance*  CI, InfoClass* _info):astContext(&(CI->getASTContext())), info(_info){}
      virtual void run(const MatchFinder::MatchResult &Results) {

        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        
        for(auto i : info->fieldData_map) {
          for(auto j : i.second) {
            //llvm::outs() << j.VAR_NAME << "\n";
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME + "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_ifMin = "ifMin_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinLHS = "ifMinLHS_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinRHS = "ifMinRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignment = "Cond_equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignmentRHS = "Cond_equalOpRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSmemExpr = "Cond_RHSmemExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSVarDecl = "Cond_RHSVarDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_whileCAS = "whileCAS_" + j.VAR_NAME + "_" + i.first;
            string str_whileCAS_RHS = "whileCAS_RHS" + j.VAR_NAME + "_" + i.first;

            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;


            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);

              auto ifMinOp = Results.Nodes.getNodeAs<clang::Stmt>(str_ifMin);
              auto whileCAS_op = Results.Nodes.getNodeAs<clang::Stmt>(str_whileCAS);
              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);

              /**Figure out variable type to set the reset value **/
              auto memExpr = Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);

                ReductionOps_entry reduceOP_entry;
                reduceOP_entry.NODE_TYPE = j.NODE_TYPE;


                reduceOP_entry.FIELD_NAME = j.FIELD_NAME ;
                reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME ;
                reduceOP_entry.SYNC_TYPE = "sync_push";

                reduceOP_entry.FIELD_TYPE = j.VAR_TYPE;
                reduceOP_entry.VAL_TYPE = j.RESET_VALTYPE;


                if(assignplusOP) {
                  string reduceOP = "node." + j.FIELD_NAME + "= node." + j.FIELD_NAME + " + y ;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(plusOP) {
                  string reduceOP = "node." + j.FIELD_NAME + "+= y;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }
                /** If this is a an expression like nodeData.fieldName = value **/
                else if(assignmentOP) {

                  string reduceOP = "node." + j.FIELD_NAME + "= y;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(ifMinOp){
                  ifMinOp->dump();
                  auto lhs = Results.Nodes.getNodeAs<clang::DeclRefExpr>(str_ifMinLHS);
                  lhs->dump();
                  // It is min operation.
                  //string reduceOP = "if(node." + j.FIELD_NAME + " > y) { node." + j.FIELD_NAME + " = y;}";
                  string reduceOP = "{ Galois::min(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "std::numeric_limits<" + j.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }

                else if(whileCAS_op){
                  whileCAS_op->dump();
                  auto whileCAS_LHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  string reduceOP = "{Galois::min(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "std::numeric_limits<" + j.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  whileCAS_LHS->dump();
                  break;
                }
               else if(atomicAdd_op){
                  string reduceOP = "{ Galois::atomicAdd(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }

            }
          }
        }
      }
};


/*COMMAN MATCHER STATEMENTS  :*************** To match Galois stype getData calls  *************************************/
      StatementMatcher GetDataMatcher = callExpr(argumentCountIs(1), callee(methodDecl(hasName("getData"))));
      StatementMatcher LHSRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl(hasType(references(AnyType))))))).bind("LHSRefVariable");

      StatementMatcher LHSNonRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl())))).bind("LHSNonRefVariable");

      /****************************************************************************************************************/


      /*COMMAN MATCHER STATEMENTS  :*************** To match Galois stype for loops  *************************************/
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


      StatementMatcher getDataCallExprMatcher_insideFor = callExpr(
                                                          isExpansionInMainFile(),
                                                          callee(functionDecl(hasName("getData"))),
                                                          hasAncestor(recordDecl().bind("callerInStruct")),
                                                          anyOf(
                                                                hasAncestor(binaryOperator(hasOperatorName("="),
                                                                                              hasLHS(anyOf(
                                                                                                        LHSRefVariable,
                                                                                                        LHSNonRefVariable))).bind("getDataAssignment")
                                                                              ),
                                                                hasAncestor(declStmt(hasSingleDecl(varDecl(hasType(references(AnyType))).bind("getData_varName"))).bind("refVariableDecl_getData")),
                                                                hasAncestor(declStmt(hasSingleDecl(varDecl(unless(hasType(references(AnyType)))).bind("getData_varName"))).bind("nonRefVariableDecl_getData"))
                                                            ),
                                                          hasDescendant(memberExpr(hasDescendant(memberExpr().bind("getData_memExpr_graph"))))
                                                     ).bind("calleeName_getEdgeDst");

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
                                                                      varDecl())))))),
                                              hasDescendant(getDataCallExprMatcher_insideFor)
                                              ).bind("forLoopName");

      /****************************************************************************************************************/




class GaloisFunctionsConsumer : public ASTConsumer {
    private:
    CompilerInstance &Instance;
    std::set<std::string> ParsedTemplates;
    GaloisFunctionsVisitor* Visitor;
    MatchFinder Matchers, Matchers_doall, Matchers_op, Matchers_typedef, Matchers_gen, Matchers_gen_field;
    FunctionCallHandler functionCallHandler;
    FindOperatorHandler findOperator;
    CallExprHandler callExprHandler;
    TypedefHandler typedefHandler;
    FindingFieldHandler f_handler;
    FindingFieldInsideForLoopHandler insideForLoop_handler;
    FieldUsedInForLoop insideForLoopField_handler;
    InfoClass info;
  public:
    GaloisFunctionsConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates, Rewriter &R): Instance(Instance), ParsedTemplates(ParsedTemplates), Visitor(new GaloisFunctionsVisitor(&Instance)), functionCallHandler(R, &info), findOperator(&Instance), callExprHandler(&Instance, &info), typedefHandler(&info), f_handler(&Instance, &info), insideForLoop_handler(&Instance, &info), insideForLoopField_handler(&Instance, &info) {

     /**************** Additional matchers ********************/
      //Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData"))), hasAncestor(binaryOperator(hasOperatorName("=")).bind("assignment"))).bind("getData"), &callExprHandler); //*** WORKS
      //Matchers.addMatcher(recordDecl(hasDescendant(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData")))).bind("getData"))).bind("getDataInStruct"), &callExprHandler); //**** works

      /*MATCHER 1 :*************** Matcher to get the type of node defined by user *************************************/
      Matchers_typedef.addMatcher(typedefDecl(isExpansionInMainFile()).bind("typedefDecl"), &typedefHandler);
      /****************************************************************************************************************/

      
      /*MATCHER 2 :************* For matching the nodes in AST with getData function call *****************************/
      StatementMatcher getDataCallExprMatcher = callExpr(
                                                          isExpansionInMainFile(),
                                                          callee(functionDecl(hasName("getData"))),
                                                          hasAncestor(recordDecl().bind("callerInStruct")),
                                                          anyOf(
                                                                hasAncestor(binaryOperator(hasOperatorName("="),
                                                                                              hasLHS(anyOf(
                                                                                                        LHSRefVariable,
                                                                                                        LHSNonRefVariable))).bind("getDataAssignment")
                                                                              ),
                                                                hasAncestor(declStmt(hasSingleDecl(varDecl(hasType(references(AnyType))).bind("getData_varName"))).bind("refVariableDecl_getData")),
                                                                hasAncestor(declStmt(hasSingleDecl(varDecl(unless(hasType(references(AnyType)))).bind("getData_varName"))).bind("nonRefVariableDecl_getData"))
                                                            ),
                                                          hasDescendant(memberExpr(hasDescendant(memberExpr().bind("getData_memExpr_graph")))),
                                                          unless(hasAncestor(EdgeForLoopMatcher))
                                                     ).bind("calleeName");
      Matchers.addMatcher(getDataCallExprMatcher, &callExprHandler);
      /****************************************************************************************************************/



      /*MATCHER 3 : ********************* For matching the nodes in AST with getEdgeDst function call and with getData inside it ******/

      /** We only care about for loops which have getEdgeDst and getData. **/
      Matchers.addMatcher(callExpr(
                                    isExpansionInMainFile(),
                                    callee(functionDecl(hasName("getEdgeDst"))),
                                    hasAncestor(recordDecl().bind("callerInStruct")),
                                    hasAncestor(EdgeForLoopMatcher),
                                    hasAncestor(declStmt(hasSingleDecl(varDecl(hasType(AnyType)).bind("varDeclName"))).bind("variableDecl_getEdgeDst"))
                                ).bind("calleeName"), &callExprHandler);


      /****************************************************************************************************************/

  }

    virtual void HandleTranslationUnit(ASTContext &Context){
      //Visitor->TraverseDecl(Context.getTranslationUnitDecl());
      Matchers_typedef.matchAST(Context);
      Matchers.matchAST(Context);
      llvm::outs() << " MAP SIZE : " << info.getData_map.size() << "\n";
      for (auto i : info.getData_map)
      {
        llvm::outs() <<"\n" <<i.first << " has " << i.second.size() << " references of getData \n";

        int width = 30;
        llvm::outs() << center(string("VAR_NAME"), width) << "|"
                     <<  center(string("VAR_TYPE"), width) << "|"
                     << center(string("SRC_NAME"), width) << "|"
                     << center(string("IS_REFERENCE"), width) << "|"
                     << center(string("IS_REFERENCED"), width) << "||"
                     << center(string("RW_STATUS"), width) << "||"
                     << center(string("GRAPH_NAME"), width) << "\n";
        llvm::outs() << std::string(width*7 + 2*7, '-') << "\n";
        for( auto j : i.second) {
          llvm::outs() << center(j.VAR_NAME,  width) << "|"
                       << center(j.VAR_TYPE,  width) << "|"
                       << center(j.SRC_NAME,  width) << "|"
                       << center(j.IS_REFERENCE,  width) << "|"
                       << center(j.IS_REFERENCED,  width) << "|"
                       << center(j.RW_STATUS,  width) << "|"
                       << center(j.GRAPH_NAME,  width) << "\n";
        }
      }

      llvm::outs() << "\n\n Printing for getData in forLoop for all edges\n\n";
      for (auto i : info.edgeData_map)
      {
        llvm::outs() << "\n" << i.first << " has " << i.second.size() << " references of getData \n";

        int width = 25;
        llvm::outs() << center(string("VAR_NAME"), width) << "|"
                     <<  center(string("VAR_TYPE"), width) << "|"
                     << center(string("SRC_NAME"), width) << "|"
                     << center(string("IS_REFERENCE"), width) << "|"
                     << center(string("IS_REFERENCED"), width) << "||"
                     << center(string("RW_STATUS"), width) << "||"
                     << center(string("GRAPH_NAME"), width) << "\n";
        llvm::outs() << std::string(width*7 + 2*7, '-') << "\n";
        for( auto j : i.second) {
          llvm::outs() << center(j.VAR_NAME,  width) << "|"
                       << center(j.VAR_TYPE,  width) << "|"
                       << center(j.SRC_NAME,  width) << "|"
                       << center(j.IS_REFERENCE,  width) << "|"
                       << center(j.RW_STATUS,  width) << "|"
                       << center(j.GRAPH_NAME,  width) << "\n";
        }
      }

      /*MATCHER 5: *********************Match to get fields of NodeData structure being modified  but not in the Galois all edges forLoop *******************/
      for (auto i : info.getData_map) {
        for(auto j : i.second) {
          if(j.IS_REFERENCED && j.IS_REFERENCE) {
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_minusOp = "minusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;

            /** Only match references types, ignore read only **/
            DeclarationMatcher f_1 = varDecl(isExpansionInMainFile(), hasInitializer(expr(anyOf(
                                                                                                hasDescendant(memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)),
                                                                                                memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)
                                                                                                ))),
                                                                      hasType(references(AnyType)),
                                                                      hasAncestor(recordDecl(hasName(i.first)))
                                                                  ).bind(str_varDecl);


            StatementMatcher LHS_memExpr = memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr);
            /** Order matters as the most generic ones should be at the last **/
            StatementMatcher f_2 = expr(isExpansionInMainFile(), anyOf(
                                                                      /** For builtInType : NodeData.field += val **/
                                                                      binaryOperator(hasOperatorName("+="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_plusOp),
                                                                      /** For builtInType : NodeData.field = NodeData.field + val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr)),
                                                                            hasRHS(binaryOperator(hasOperatorName("+"),
                                                                                                  hasLHS(ignoringParenImpCasts(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For builtInType : NodeData.field = val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_assignment),


                                                                      /** For ComplexType : NodeData.field += val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("+="),
                                                                              hasDescendant(LHS_memExpr)).bind(str_plusOp),
                                                                      /** For ComplexType : NodeData.field = NodeData.field + val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                              hasDescendant(binaryOperator(hasOperatorName("+"),
                                                                                                              hasLHS(hasDescendant(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For ComplexType : NodeData.field = val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                             hasDescendant(LHS_memExpr)).bind(str_assignment),

                                                                      /** Atomic Add **/
                                                                      callExpr(argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_memExpr))).bind(str_atomicAdd)
                                                                     ),
                                                                     hasAncestor(recordDecl(hasName(i.first)))/*,
                                                                     unless(hasAncestor(ifStmt(hasCondition(anything()))))*/);

            Matchers_gen.addMatcher(f_1, &f_handler);
            Matchers_gen.addMatcher(f_2, &f_handler);
          }
        }
      }

      /****************************************************************************************************************/

      /*MATCHER 5: *********************Match to get fields of NodeData structure being modified inside the Galois all edges forLoop *******************/
      for (auto i : info.edgeData_map) {
        for(auto j : i.second) {
          if(j.IS_REFERENCED && j.IS_REFERENCE) {
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_minusOp = "minusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_ifMin = "ifMin_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinRHS = "ifMinRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignment = "Cond_equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignmentRHS = "Cond_equalOpRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSmemExpr = "Cond_RHSmemExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSVarDecl = "Cond_RHSVarDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_whileCAS = "whileCAS_" + j.VAR_NAME + "_" + i.first;
            string str_whileCAS_RHS = "whileCAS_RHS" + j.VAR_NAME + "_" + i.first;

            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;

            /** Only match references types, ignore read only **/
            DeclarationMatcher f_1 = varDecl(isExpansionInMainFile(), hasInitializer(expr(anyOf(
                                                                                                hasDescendant(memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)),
                                                                                                memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)
                                                                                                ))),
                                                                      hasType(references(AnyType)),
                                                                      hasAncestor(recordDecl(hasName(i.first)))
                                                                  ).bind(str_varDecl);


            StatementMatcher LHS_memExpr = memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME))))), hasAncestor(recordDecl(hasName(i.first)))).bind(str_memExpr);
            StatementMatcher Cond_notMemExpr = memberExpr(hasDescendant(declRefExpr(to(varDecl(unless(hasName(j.VAR_NAME))).bind(str_varDecl)))), hasAncestor(recordDecl(hasName(i.first)))).bind(str_Cond_RHSmemExpr);
            StatementMatcher Cond_notDeclExpr = declRefExpr(to(varDecl(unless(hasName(j.VAR_NAME))).bind(str_varDecl)), hasAncestor(recordDecl(hasName(i.first)))).bind(str_Cond_RHSmemExpr);
            /** Order matters as the most generic ones should be at the last **/
            StatementMatcher f_2 = expr(isExpansionInMainFile(), anyOf(
                                                                      /** For builtInType : NodeData.field += val **/
                                                                      binaryOperator(hasOperatorName("+="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_plusOp),
                                                                      /** For builtInType : NodeData.field = NodeData.field + val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr)),
                                                                            hasRHS(binaryOperator(hasOperatorName("+"),
                                                                                                  hasLHS(ignoringParenImpCasts(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For builtInType : NodeData.field = val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_assignment),


                                                                      /** For ComplexType : NodeData.field += val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("+="),
                                                                              hasDescendant(LHS_memExpr)).bind(str_plusOp),
                                                                      /** For ComplexType : NodeData.field = NodeData.field + val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                              hasDescendant(binaryOperator(hasOperatorName("+"),
                                                                                                              hasLHS(hasDescendant(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For ComplexType : NodeData.field = val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                             hasDescendant(LHS_memExpr)).bind(str_assignment),
                                                                      callExpr(isExpansionInMainFile(), argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_memExpr))).bind(str_atomicAdd)
                                                                     ),
                                                                     hasAncestor(recordDecl(hasName(i.first))),
                                                                     /** Only if condition inside EdgeForLoop should be considered**/
                                                                     unless(hasAncestor(ifStmt(hasCondition(anything()), hasAncestor(EdgeForLoopMatcher)))));

            /** Conditional modification within forEdgeLoop **/
            StatementMatcher f_3 = ifStmt(isExpansionInMainFile(), hasCondition(allOf(binaryOperator(hasOperatorName(">"),
                                                                                      hasLHS(hasDescendant(LHS_memExpr)),
                                                                                      hasRHS(hasDescendant(declRefExpr(to(decl().bind(str_Cond_RHSVarDecl))).bind(str_ifMinRHS)))),
                                                                                      hasAncestor(EdgeForLoopMatcher)
                                                                                )), hasDescendant(compoundStmt(anyOf(hasDescendant(binaryOperator(hasOperatorName("="),
                                                                                                                                                  hasLHS(ignoringParenImpCasts(LHS_memExpr)),
                                                                                                                                                  hasRHS(hasDescendant(declRefExpr(to(decl(equalsBoundNode(str_Cond_RHSVarDecl)))))))),
                                                                                                                                    hasDescendant(operatorCallExpr(hasOverloadedOperatorName("="),
                                                                                                                                                  hasDescendant(LHS_memExpr),
                                                                                                                                                  hasDescendant(Cond_notDeclExpr)))
                                                                                                                                    )))).bind(str_ifMin);

            StatementMatcher f_4 = whileStmt(isExpansionInMainFile(), hasCondition(binaryOperator(hasOperatorName(">"),
                                                                                                  hasLHS(hasDescendant(LHS_memExpr)),
                                                                                                  hasRHS(ignoringParenImpCasts(declRefExpr(to(decl().bind(str_whileCAS_RHS))))))),
                                                                      hasBody(compoundStmt(hasDescendant(memberCallExpr(callee(methodDecl(matchesName(".compare_exchange_strong"))), hasAnyArgument(declRefExpr(to(decl(equalsBoundNode(str_whileCAS_RHS)))))))))).bind(str_whileCAS);

            //StatementMatcher f_5 = callExpr(isExpansionInMainFile(), argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_memExpr))).bind(str_atomicAdd);
            Matchers_gen.addMatcher(f_1, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_2, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_3, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_4, &insideForLoop_handler);
            //Matchers_gen.addMatcher(f_5, &insideForLoop_handler);
          }
        }
      }

      /****************************************************************************************************************/

      Matchers_gen.matchAST(Context);

      for(auto i : info.fieldData_map){
        for(auto j : i.second) {
          if(j.IS_REFERENCED && j.IS_REFERENCE) {
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_minusOp = "minusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_ifMin = "ifMin_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinLHS = "ifMinLHS_" + j.VAR_NAME+ "_" + i.first;
            string str_ifMinRHS = "ifMinRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignment = "Cond_equalOp_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_assignmentRHS = "Cond_equalOpRHS_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSmemExpr = "Cond_RHSmemExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_Cond_RHSVarDecl = "Cond_RHSVarDecl_" + j.VAR_NAME+ "_" + i.first;

            string str_whileCAS = "whileCAS_" + j.VAR_NAME + "_" + i.first;
            string str_whileCAS_RHS = "whileCAS_RHS" + j.VAR_NAME + "_" + i.first;

            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;

            /** Only match references types, ignore read only **/
            DeclarationMatcher f_1 = varDecl(isExpansionInMainFile(), hasInitializer(expr(anyOf(
                                                                                                hasDescendant(memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)),
                                                                                                memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)
                                                                                                ))),
                                                                      hasType(references(AnyType)),
                                                                      hasAncestor(recordDecl(hasName(i.first)))
                                                                  ).bind(str_varDecl);



            StatementMatcher LHS_memExpr = memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME))))), hasAncestor(recordDecl(hasName(i.first)))).bind(str_memExpr);
            StatementMatcher LHS_declRefExpr = declRefExpr(to(varDecl(hasName(j.VAR_NAME)))).bind(str_ifMinLHS);
            StatementMatcher Cond_notDeclExpr = declRefExpr(to(varDecl(unless(hasName(j.VAR_NAME))).bind(str_varDecl)), hasAncestor(recordDecl(hasName(i.first)))).bind(str_Cond_RHSmemExpr);
            /** Order matters as the most generic ones should be at the last **/
            StatementMatcher f_2 = expr(isExpansionInMainFile(), anyOf(
                                                                      /** For builtInType : NodeData.field += val **/
                                                                      binaryOperator(hasOperatorName("+="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_plusOp),
                                                                      /** For builtInType : NodeData.field = NodeData.field + val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr)),
                                                                            hasRHS(binaryOperator(hasOperatorName("+"),
                                                                                                  hasLHS(ignoringParenImpCasts(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For builtInType : NodeData.field = val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_memExpr))).bind(str_assignment),


                                                                      /** For ComplexType : NodeData.field += val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("+="),
                                                                              hasDescendant(LHS_memExpr)).bind(str_plusOp),
                                                                      /** For ComplexType : NodeData.field = NodeData.field + val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                              hasDescendant(binaryOperator(hasOperatorName("+"),
                                                                                                              hasLHS(hasDescendant(LHS_memExpr))))).bind(str_assign_plus),
                                                                      /** For ComplexType : NodeData.field = val **/
                                                                      operatorCallExpr(hasOverloadedOperatorName("="),
                                                                             hasDescendant(LHS_memExpr)).bind(str_assignment),

                                                                      callExpr(isExpansionInMainFile(), argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_declRefExpr))).bind(str_atomicAdd)
                                                                     ),
                                                                     hasAncestor(recordDecl(hasName(i.first))),
                                                                     /** Only if condition inside EdgeForLoop should be considered**/
                                                                     unless(hasAncestor(ifStmt(hasCondition(anything()), hasAncestor(EdgeForLoopMatcher)))));

            /** Conditional modification within forEdgeLoop **/
            StatementMatcher f_3 = ifStmt(isExpansionInMainFile(), hasCondition(binaryOperator(hasOperatorName(">"),
                                                                                      hasLHS(anyOf(hasDescendant(LHS_memExpr), hasDescendant(LHS_declRefExpr))),
                                                                                      hasRHS(hasDescendant(declRefExpr(to(decl().bind(str_Cond_RHSVarDecl))).bind(str_ifMinRHS))))
                                                                                ), anyOf(hasDescendant(compoundStmt(anyOf(hasDescendant(binaryOperator(hasOperatorName("="),
                                                                                                                                       hasLHS(hasDescendant(LHS_declRefExpr)),
                                                                                                                                       hasRHS(hasDescendant(declRefExpr(to(decl(equalsBoundNode(str_Cond_RHSVarDecl)))))))),
                                                                                                                    hasDescendant(operatorCallExpr(hasOverloadedOperatorName("="),
                                                                                                                                  hasDescendant(LHS_declRefExpr),
                                                                                                                                  hasDescendant(declRefExpr(to(decl(equalsBoundNode(str_Cond_RHSVarDecl)))))))))),
                                                                                            hasDescendant(binaryOperator(hasOperatorName("="),
                                                                                                        hasLHS(LHS_declRefExpr),
                                                                                                        hasRHS(ignoringParenImpCasts(declRefExpr(to(decl(equalsBoundNode(str_Cond_RHSVarDecl))))))))
                                                                                                    )
                                                                                                ).bind(str_ifMin);

            StatementMatcher f_4 = whileStmt(isExpansionInMainFile(), hasCondition(binaryOperator(hasOperatorName(">"),
                                                                                                  hasLHS(hasDescendant(LHS_memExpr)),
                                                                                                  hasRHS(ignoringParenImpCasts(declRefExpr(to(decl().bind(str_whileCAS_RHS))))))),
                                                                      hasBody(compoundStmt(hasDescendant(memberCallExpr(callee(methodDecl(matchesName(".compare_exchange_strong"))), hasAnyArgument(declRefExpr(to(decl(equalsBoundNode(str_whileCAS_RHS)))))))))).bind(str_whileCAS);

            //StatementMatcher f_5 = callExpr(isExpansionInMainFile(), argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd")))[>, hasAnyArgument(LHS_memExpr)<])).bind(str_atomicAdd);

            Matchers_gen_field.addMatcher(f_1, &insideForLoopField_handler);
            Matchers_gen_field.addMatcher(f_2, &insideForLoopField_handler);
            Matchers_gen_field.addMatcher(f_3, &insideForLoopField_handler);
            Matchers_gen_field.addMatcher(f_4, &insideForLoopField_handler);
            //Matchers_gen_field.addMatcher(f_5, &insideForLoopField_handler);


        }
      }
    }

      Matchers_gen_field.matchAST(Context);

      llvm::outs() << "\n\n Printing for all fields found \n\n";
      for (auto i : info.fieldData_map)
      {
        llvm::outs() << "\n" << i.first << " has " << i.second.size() << " Fields \n";

        int width = 25;
        llvm::outs() << center(string("VAR_NAME"), width) << "|"
                     <<  center(string("VAR_TYPE"), width) << "|"
                     << center(string("FIELD_NAME"), width) << "|"
                     << center(string("NODE_NAME"), width) << "|"
                     << center(string("IS_REFERENCE"), width) << "|"
                     << center(string("IS_REFERENCED"), width) << "||"
                     << center(string("RW_STATUS"), width) << "||"
                     << center(string("RESET_VALTYPE"), width) << "||"
                     << center(string("RESET_VAL"), width) << "||"
                     << center(string("GRAPH_NAME"), width) << "\n";
        llvm::outs() << std::string(width*10 + 2*10, '-') << "\n";
        for( auto j : i.second) {
          llvm::outs() << center(j.VAR_NAME,  width) << "|"
                       << center(j.VAR_TYPE,  width) << "|"
                       << center(j.FIELD_NAME,  width) << "|"
                       << center(j.NODE_NAME,  width) << "|"
                       << center(j.IS_REFERENCE,  width) << "|"
                       << center(j.IS_REFERENCED,  width) << "|"
                       << center(j.RW_STATUS,  width) << "|"
                       << center(j.RESET_VALTYPE,  width) << "|"
                       << center(j.RESET_VAL,  width) << "|"
                       << center(j.GRAPH_NAME,  width) << "\n";
        }
      }


      /** PRINTING FINAL REDUCTION OPERATIONS **/

      llvm::outs() << "\n\n"; 
      for (auto i : info.reductionOps_map){
        llvm::outs() << "FOR >>>>> " << i.first << "\n";
        for(auto j : i.second) {
          string write_set = "write_set( " + j.NODE_TYPE + " , " + j.FIELD_NAME + " , " + j.OPERATION_EXPR + " , " +  j.RESETVAL_EXPR + ")";
          llvm::outs() << write_set << "\n";
        }
      }


      for (auto i : info.reductionOps_map){
        if(i.second.size() > 0) {
          Matchers_doall.addMatcher(callExpr(callee(functionDecl(hasName("Galois::do_all"))),unless(isExpansionInSystemHeader()), hasAncestor(recordDecl(hasName(i.first)).bind("do_all_recordDecl"))).bind("galoisLoop"), &functionCallHandler);
        }
      }

      Matchers_doall.matchAST(Context);
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

static FrontendPluginRegistry::Add<GaloisFunctionsAction> X("galois-analysis", "find galois function names");
