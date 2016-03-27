/*
 * Author : Gurbinder Gill (gurbinder533@gmail.com)
 */

#include <sstream>
#include <climits>
#include <vector>
#include <map>
#include <regex>
#include <algorithm>
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


  /**************************** Helper Functions: start *****************************/
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

/** Check for a duplicate entry in the map **/
template <typename EntryTy>
bool syncPull_reduction_exists(EntryTy entry, vector<EntryTy> entry_vec){
  if(entry_vec.size() == 0){
    return false;
  }
  for(auto it : entry_vec){
    if((it.SYNC_TYPE == "syncPull") && (it.FIELD_NAME == entry.FIELD_NAME) && (it.FIELD_TYPE == entry.FIELD_TYPE) && (it.NODE_TYPE == entry.NODE_TYPE) && (it.GRAPH_NAME == entry.GRAPH_NAME))
      return false;
  }
  return true;
}

/**************************** Helper Functions: End *****************************/


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
    string SYNC_TYPE;
    bool IS_VEC;
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

  struct FirstIter_struct_entry{
    //pair: type, varName
    vector<pair<string,string>> MEMBER_FIELD_VEC;
    string OPERATOR_ARG;
    string OPERATOR_BODY;
  };

  class InfoClass {
    public:
      map<string, vector<Graph_entry>> graphs_map;
      map<string, vector<getData_entry>> getData_map;
      map<string, vector<getData_entry>> edgeData_map;
      map<string, vector<NodeField_entry>> fieldData_map;
      map<string, vector<ReductionOps_entry>> reductionOps_map;
      map<string, vector<FirstIter_struct_entry>> FirstItr_struct_map;
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
      //typedefDecl->dump();
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

                //llvm::outs() << " IS REFERENCED = > " << varDecl_getData->isReferenced() << "\n";
                //llvm::outs() << varDecl_getData->getNameAsString() << "\n";
                llvm::outs() << "THIS IS TYPE : " << varDecl_getData->getType().getAsString() << "\n";
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


/** Helping Functions **/
bool is_vector_type(string inputStr, string& Type, string& Num_ele) {
  /** Check if type is suppose to be a vector **/
  //llvm::outs()<< "inside is_vec : " <<  inputStr << "\n ";
  //std::regex re("([(a-z)]+)[[:space:]&]*(\[[0-9][[:d:]]*\])");
  std::regex re("(.*)(\[[0-9][[:d:]]*\])");
  std::smatch sm;
  if(std::regex_match(inputStr, sm, re)) {
    Type = "std::vector<" + sm[1].str() + ">";
    llvm::outs() << "INSIDE ALSO : " << Type << "\n";
    Num_ele = sm[2].str();
    Num_ele.erase(0,1);
    Num_ele.erase(Num_ele.size() -1);
    return true;
  }
  return false;
}


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

        auto forVector = Results.Nodes.getNodeAs<clang::CXXBindTemporaryExpr>("forVector");
        //forVector->dump();
        for(auto i : info->getData_map) {
          for(auto j : i.second) {
            //llvm::outs() << j.VAR_NAME << "\n";
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment = "equalOp_" + j.VAR_NAME + "_" + i.first;
            string str_plusOp = "plusEqualOp_" + j.VAR_NAME+ "_" + i.first;
            string str_assign_plus = "assignplusOp_" + j.VAR_NAME+ "_" + i.first;
            string str_varDecl = "varDecl_" + j.VAR_NAME+ "_" + i.first;
            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;
            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;

            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);
              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);

              /** Vector operations **/
              auto plusOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp_vec);
              auto assignmentOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment_vec);


              /**Figure out variable type to set the reset value **/
              auto memExpr = Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);
              //memExpr->dump();
              if(memExpr){
                //memExpr->dump();
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

                /** Check if type is suppose to be a vector **/
                string Type = Ty;
                if (plusOP_vec || assignmentOP_vec){
                  Type = "std::vector<" + Ty + ">";
                }
                string Num_ele = "0";
                bool is_vector = is_vector_type(Ty, Type, Num_ele);
                /*
                std::regex re("([(a-z)]+)[[:space:]&]*(\[[0-9][[:d:]]*\])");
                std::smatch sm;
                if(std::regex_match(Ty, sm, re)) {
                   Type = "std::vector<" + sm[1].str() + ">";
                   Num_ele = sm[2].str();
                   Num_ele.erase(0,1);
                   Num_ele.erase(Num_ele.size() -1);
                }*/

                ReductionOps_entry reduceOP_entry;
                NodeField_entry field_entry;
                field_entry.RESET_VALTYPE = reduceOP_entry.VAL_TYPE = Type;
                field_entry.NODE_TYPE = j.VAR_TYPE;
                field_entry.IS_VEC = is_vector;
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
                if(assignmentOP_vec) {
                  //assignmentOP_vec->dump();
                }
                if(!field && (assignplusOP || plusOP || assignmentOP || atomicAdd_op || plusOP_vec || assignmentOP_vec)) {
                  reduceOP_entry.SYNC_TYPE = "sync_pull";
                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                /*
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
                // If this is a an expression like nodeData.fieldName = value
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
                */
                else if(field){

                  field_entry.VAR_NAME = field->getNameAsString();
                  field_entry.VAR_TYPE = field_entry.RESET_VALTYPE = field->getType().getNonReferenceType().getAsString();
                  llvm::outs() << " VAR TYPE in getdata not in loop :" << field_entry.VAR_TYPE << "\n";
                  field_entry.IS_REFERENCED = field->isReferenced();
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";
                  field_entry.SYNC_TYPE = "sync_pull";
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

            /** For vector operations **/
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;

            /** Sync pull variables **/
            string str_syncPull_var = "syncPullVar_" + j.VAR_NAME + "_" + i.first;

            

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

              /** Vector operations **/
              auto plusOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp_vec);
              auto assignmentOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment_vec);

              /** Sync Pull variables **/
              auto syncPull_var = Results.Nodes.getNodeAs<clang::VarDecl>(str_syncPull_var);

              

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

                /** Check if type is suppose to be a vector **/
                string Type = Ty;
                if (plusOP_vec || assignmentOP_vec){
                  Type = "std::vector<" + Ty + ">";
                }
                string Num_ele = "0";
                bool is_vector = is_vector_type(Ty, Type, Num_ele);


                NodeField_entry field_entry;
                ReductionOps_entry reduceOP_entry;
                field_entry.NODE_TYPE = j.VAR_TYPE;
                field_entry.IS_VEC = is_vector;
                reduceOP_entry.NODE_TYPE = j.VAR_TYPE;
                reduceOP_entry.FIELD_TYPE = j.VAR_TYPE;

                field_entry.VAR_TYPE = field_entry.RESET_VALTYPE = reduceOP_entry.VAL_TYPE = Type;

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

                if(syncPull_var){
                  //syncPull_var->dumpColor();
                  if(syncPull_var->isReferenced()){
                    reduceOP_entry.SYNC_TYPE = "sync_pull";
                    /** check for duplicate **/
                    if(!syncPull_reduction_exists(reduceOP_entry, info->reductionOps_map[i.first])){
                      info->reductionOps_map[i.first].push_back(reduceOP_entry);
                    }
                  }
                  break;
                }

                if(assignplusOP) {
                  //assignplusOP->dump();
                  field_entry.VAR_NAME = "+";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";

                  string reduceOP = "{node." + field_entry.FIELD_NAME + "= node." + field_entry.FIELD_NAME + " + y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + field_entry.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //info->fieldData_map[i.first].push_back(field_entry);
                  break;
                }
                else if(plusOP) {
                  field_entry.VAR_NAME = "+=";
                  field_entry.IS_REFERENCED = true;
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";

                  string reduceOP, resetValExpr;

                  if(is_vector){
                    reduceOP = "{Galois::pairWiseAvg_vec(node." + field_entry.FIELD_NAME + ", y); }";
                    resetValExpr = "{Galois::resetVec(node." + field_entry.FIELD_NAME + "); }";
                  }
                  else {
                    reduceOP = "{node." + field_entry.FIELD_NAME + "+= y;}";
                    resetValExpr = "{node." + field_entry.FIELD_NAME + " = 0 ; }";
                  }

                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;
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


                  string reduceOP = "{node." + field_entry.FIELD_NAME + " = y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  string resetValExpr = "{node." + field_entry.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

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
                    string resetValExpr = "{node." + field_entry.FIELD_NAME + " = std::numeric_limits<" + field_entry.RESET_VALTYPE + ">::max()";
                    reduceOP_entry.RESETVAL_EXPR = resetValExpr;
                    //varAssignRHS->dump();

                    info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  }
                  break;

                }
                else if(whileCAS_op){
                  auto whileCAS_LHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  //whileCAS_LHS->dump();
                  string reduceOP = "{Galois::min(node." + field_entry.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + field_entry.FIELD_NAME + " = std::numeric_limits<" + field_entry.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  //whileCAS_LHS->dump();
                  break;
                }

                else if(atomicAdd_op){
                  //llvm::outs() << "NOT  INSIDE  FIELD \n";
                  //atomicAdd_op->dump();
                  string reduceOP = "{ Galois::atomicAdd(node." + field_entry.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + field_entry.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(assignmentOP_vec){
                  string reduceOP = "{node." + field_entry.FIELD_NAME + "= y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + field_entry.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(plusOP_vec){
                  string reduceOP, resetValExpr;
                  reduceOP = "{Galois::pairWiseAvg_vec(node." + field_entry.FIELD_NAME + ", y); }";
                  resetValExpr = "{Galois::resetVec(node." + field_entry.FIELD_NAME + "); }";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;
                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }
                else if(field){

                  field_entry.VAR_NAME = field->getNameAsString();
                  field_entry.VAR_TYPE = field->getType().getNonReferenceType().getAsString();
                  llvm::outs() << " VAR TYPE in getdata IN loop :" << field_entry.VAR_TYPE << "\n";
                  field_entry.IS_REFERENCED = field->isReferenced();
                  field_entry.IS_REFERENCE = true;
                  field_entry.RESET_VAL = "0";
                  field_entry.RESET_VALTYPE = field_entry.VAR_TYPE;
                  field_entry.SYNC_TYPE = "sync_push";
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

            /** For vector operations **/
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;

            
            auto vectorOp = Results.Nodes.getNodeAs<clang::Stmt>("vectorOp");
            if(vectorOp){
              vectorOp->dump();
              return;
            }
           

            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);

              auto ifMinOp = Results.Nodes.getNodeAs<clang::Stmt>(str_ifMin);
              auto whileCAS_op = Results.Nodes.getNodeAs<clang::Stmt>(str_whileCAS);
              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);

              auto assignmentOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment_vec);
              auto plusOP_vec = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp_vec);

              /**Figure out variable type to set the reset value **/
              auto memExpr = Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);
              /** Check if type is suppose to be a vector **/
              string Type = j.VAR_TYPE;
              if (plusOP_vec || assignmentOP_vec){
                Type = "std::vector<" + j.VAR_TYPE + ">";
              }
              ReductionOps_entry reduceOP_entry;
              reduceOP_entry.NODE_TYPE = j.NODE_TYPE;


              reduceOP_entry.FIELD_NAME = j.FIELD_NAME ;
              reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME ;
              reduceOP_entry.SYNC_TYPE =  j.SYNC_TYPE; //"sync_push";

              reduceOP_entry.FIELD_TYPE = Type;
              reduceOP_entry.VAL_TYPE = j.RESET_VALTYPE;


              if(assignplusOP) {
                string reduceOP = "{ node." + j.FIELD_NAME + "= node." + j.FIELD_NAME + " + y ;}";
                reduceOP_entry.OPERATION_EXPR = reduceOP;
                string resetValExpr = "{node." + j.FIELD_NAME + " = 0 ; }";
                reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                info->reductionOps_map[i.first].push_back(reduceOP_entry);
                break;
              }
              else if(plusOP) {

                  /*
                  string reduceOP = "node." + j.FIELD_NAME + "+= y;";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetVal = "0";
                  reduceOP_entry.RESETVAL_EXPR = resetVal;
                  */
                  string reduceOP, resetValExpr;

                  if(j.IS_VEC){
                    reduceOP = "{Galois::pairWiseAvg_vec(node." + j.FIELD_NAME + ", y); }";
                    resetValExpr = "{Galois::resetVec(node." + j.FIELD_NAME + "); }";
                  }
                  else {
                    reduceOP = "{node." + j.FIELD_NAME + "+= y;}";
                    resetValExpr = "{node." + j.FIELD_NAME + "= 0 ; }";
                  }

                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;
                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }
                /** If this is a an expression like nodeData.fieldName = value **/
                else if(assignmentOP) {

                  string reduceOP = "{node." + j.FIELD_NAME + "= y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + j.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

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

                  string resetValExpr = "{node." + j.FIELD_NAME + " = std::numeric_limits<" + j.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }

                else if(whileCAS_op){
                  whileCAS_op->dump();
                  auto whileCAS_LHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  string reduceOP = "{Galois::min(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  string resetValExpr = "{node." + j.FIELD_NAME + " = std::numeric_limits<" + j.RESET_VALTYPE + ">::max()";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  whileCAS_LHS->dump();
                  break;
                }
                else if(atomicAdd_op){
                  llvm::outs() << "INSIDE  FIELD \n";
                  atomicAdd_op->dump();
                  string reduceOP = "{ Galois::atomicAdd(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  string resetValExpr = "{node." + j.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(assignmentOP_vec){
                  string reduceOP = "{node." + j.FIELD_NAME + "= y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  string resetValExpr = "{node." + j.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(plusOP_vec){
                  string reduceOP, resetValExpr;
                  reduceOP = "{Galois::pairWiseAvg_vec(node." + j.FIELD_NAME + ", y); }";
                  resetValExpr = "{Galois::resetVec(node." + j.FIELD_NAME + "); }";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;
                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }

            }
          }
        }
      }
};



/** To split string on Dot operator and return num numbered argument **/
template<typename Ty>
string split_dot(Ty memExpr, unsigned num ){
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  string str_field;
  llvm::raw_string_ostream s(str_field);
  /** Returns something like snode.field_name.operator **/
  memExpr->printPretty(s, 0, Policy);

  /** Split string at delims to get the field name **/
  char delim = '.';
  vector<string> elems;
  split(s.str(), delim, elems);

  if(num == 0)
    return elems[0];
  if(num == 1)
    return elems[1];
}

template<typename Ty>
string printPretty(Ty memExpr){
  clang::LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  clang::PrintingPolicy Policy(LangOpts);

  string str_field;
  llvm::raw_string_ostream s(str_field);
  memExpr->printPretty(s, 0, Policy);
  return s.str();
}


string global_var_from_local(string localVar){
  std::regex re ("^local_(.*)");
  std::smatch sm;
  if(std::regex_match(localVar, sm, re)){
    return sm[1].str();
  }
  else {
    return "";
  }
}

/***Global constants ***/
string galois_distributed_accumulator_type = "Galois::DGAccumulator<int> ";
string galois_distributed_accumulator_name = "DGAccumulator_accum";


string makeFunctorFirstIter(string orig_functor_name, FirstIter_struct_entry entry, vector<ReductionOps_entry>& redOp_vec){
  string functor = "struct FirstItr_" + orig_functor_name + "{\n";

  /** Adding member functions **/
  for(auto memField : entry.MEMBER_FIELD_VEC){
    functor += (memField.first + " " + memField.second + ";\n");
  }

  /** Adding constructors **/
  string constructor = "FirstItr_" + orig_functor_name + "(";
  string initList = "";
  string initList_call = "";
  unsigned num = entry.MEMBER_FIELD_VEC.size();
  for(auto memField : entry.MEMBER_FIELD_VEC){
    --num;
    if(num > 0){
      constructor += ( memField.first + " _" + memField.second + ",");
      initList += (memField.second + "(_" + memField.second + "),");
      string global_var = global_var_from_local(memField.second);
      if(!global_var.empty())
        initList_call += (global_var + ",");
    }
    else {
      constructor += ( memField.first + " _" + memField.second + ")");
      initList += (memField.second + "(_" + memField.second + "){}");
      string  global_var = global_var_from_local(memField.second);
      if(!global_var.empty())
        initList_call += (global_var);
    }
  }

  constructor += (":" + initList);
  functor += (constructor + "\n");

  /** Insert write sets **/
  stringstream SS_write_set;
  for(auto j : redOp_vec){
    if(j.SYNC_TYPE == "sync_push")
      SS_write_set << ", Galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \"" << j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\" , \"" << j.FIELD_NAME << "\", \"" << j.VAL_TYPE << "\" , \"" << j.OPERATION_EXPR << "\",  \"" << j.RESETVAL_EXPR << "\")";
    else if(j.SYNC_TYPE == "sync_pull")
      SS_write_set << ", Galois::write_set(\"" << j.SYNC_TYPE << "\", \"" << j.GRAPH_NAME << "\", \""<< j.NODE_TYPE << "\", \"" << j.FIELD_TYPE << "\", \"" << j.FIELD_NAME << "\" , \"" << j.VAL_TYPE << "\")";
  }
  /** Adding static go function **/
  //TODO: Assuming graph type is Graph
  string static_go = "void static go(Graph& _graph) {\n";
  static_go += "Galois::for_each(_graph.begin(), _graph.end(), FirstItr_" + orig_functor_name + "{" + initList_call + "&_graph" + "}" + SS_write_set.str() + ");\n}\n";
  functor += static_go;

  string operator_func = entry.OPERATOR_BODY + "\n";
  functor += operator_func;


  functor += "\n};\n";

  return functor;
}

class LoopTransformHandler : public MatchFinder::MatchCallback {
  private:
    Rewriter &rewriter;
    InfoClass* info;
  public:
    LoopTransformHandler(Rewriter &rewriter, InfoClass* _info ) :  rewriter(rewriter), info(_info){}
    virtual void run(const MatchFinder::MatchResult &Results) {

        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);
      for(auto i : info->edgeData_map) {
        for(auto j : i.second) {
          string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;

          if(j.IS_REFERENCED && j.IS_REFERENCE){
            string str_ifGreater_2loopTrans = "ifGreater_2loopTrans_" + j.VAR_NAME + "_" + i.first;
            string str_if_RHS = "if_RHS_" + j.VAR_NAME + "_" + i.first;
            string str_main_struct = "main_struct_" + i.first;
            string str_forLoop_2LT = "forLoop_2LT_" + i.first;
            string str_method_operator  = "methodDecl_" + i.first;
            string str_sdata = "sdata_" + j.VAR_NAME + "_" + i.first;
            string str_for_each = "for_each_" + j.VAR_NAME + "_" + i.first;

            auto ifGreater_2loopTrans = Results.Nodes.getNodeAs<clang::Stmt>(str_ifGreater_2loopTrans);
            auto if_RHS_memExpr = Results.Nodes.getNodeAs<clang::Stmt>(str_if_RHS);
            auto if_LHS_memExpr = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
            if(ifGreater_2loopTrans){
              FirstIter_struct_entry first_itr_entry;
              SourceLocation if_loc = ifGreater_2loopTrans->getSourceRange().getBegin();

              //TODO: Assuming if statement has parenthesis. NOT NECESSARY!!
              /**1: remove if statement (order matters) **/
              unsigned len_rm = ifGreater_2loopTrans->getSourceRange().getEnd().getRawEncoding() -  if_loc.getRawEncoding() + 1;
              rewriter.RemoveText(if_loc, len_rm);

              /**2: extract operator body without if statement (order matters) **/
              auto method_operator = Results.Nodes.getNodeAs<clang::CXXMethodDecl>(str_method_operator);
              string operator_body = rewriter.getRewrittenText(method_operator->getSourceRange());
              first_itr_entry.OPERATOR_BODY = operator_body;
              
              /**Adding Distributed Galois accumulator **/
              SourceLocation method_operator_begin = method_operator->getSourceRange().getBegin();
              rewriter.InsertTextBefore(method_operator_begin, ("static " + galois_distributed_accumulator_type + galois_distributed_accumulator_name + ";\n"));


              /**3: add new if statement (order matters) **/
              /** Adding new condtional outside for loop **/
              auto forLoop_2LT = Results.Nodes.getNodeAs<clang::Stmt>(str_forLoop_2LT);
              auto sdata_declStmt = Results.Nodes.getNodeAs<clang::Stmt>(str_sdata);
              SourceLocation for_loc_begin = forLoop_2LT->getSourceRange().getBegin();

              /**Add galois accumulator += (work is done) **/
              string work_done = "\n" + galois_distributed_accumulator_name + "+= 1;\n";
              rewriter.InsertTextBefore(for_loc_begin, work_done);

              //TODO: change sdata_loc getLocWithOffset from 2 to total length of statement.
              SourceLocation sdata_loc = sdata_declStmt->getSourceRange().getEnd().getLocWithOffset(2);
              string new_condition = "\nif( " + info->getData_map[i.first][0].VAR_NAME + "." + split_dot(if_LHS_memExpr, 1) + " > " + split_dot(if_RHS_memExpr, 0) + "){\n";
              rewriter.InsertText(sdata_loc, new_condition, true, true);

              auto method_operator_loc = method_operator->getSourceRange().getEnd();
              string extra_paren = "    }\n";
              rewriter.InsertText(method_operator_loc, extra_paren, true, true);


              /**4: Get main struct arguments (order matters) **/
              auto main_struct = Results.Nodes.getNodeAs<clang::CXXRecordDecl>(str_main_struct);
              for(auto field_itr = main_struct->field_begin(); field_itr != main_struct->field_end(); ++field_itr){
                auto mem_field = std::make_pair(field_itr->getType().getAsString(), field_itr->getNameAsString());
                first_itr_entry.MEMBER_FIELD_VEC.push_back(mem_field);
              }

              /**5: Store the first argument of the operator call **/
              //TODO: Assuming there is (src, context) in arguments
              auto OP_parm_itr = method_operator->param_begin();
              string operator_arg = (*OP_parm_itr)->getType().getAsString() + "  " + (*OP_parm_itr)->getNameAsString();
              first_itr_entry.OPERATOR_ARG = operator_arg;

              info->FirstItr_struct_map[i.first].push_back(first_itr_entry);

              /**6: Put functor for first iteration. All nodes need to be processed in the first iteration. **/
              string firstItr_func = makeFunctorFirstIter(i.first, first_itr_entry, info->reductionOps_map[i.first]);
              SourceLocation main_struct_loc_begin = main_struct->getSourceRange().getBegin();
              rewriter.InsertText(main_struct_loc_begin, firstItr_func, true, true);

              /**7: call functor for first iteration in the main functor and do while of for_each ***/
              auto for_each_node = Results.Nodes.getNodeAs<clang::Stmt>(str_for_each);
              SourceLocation for_each_loc_begin = for_each_node->getSourceRange().getBegin();
              SourceLocation for_each_loc_end = for_each_node->getSourceRange().getEnd().getLocWithOffset(2);
              //TODO:: get _graph from go struct IMPORTANT
              string firstItr_func_call = "FirstItr_" + i.first + "::go(_graph);\n";
              string do_while = firstItr_func_call + "\n do { \n" + galois_distributed_accumulator_name + ".reset();\n";
              rewriter.InsertText(for_each_loc_begin, do_while, true, true);

              string while_conditional = "\n}while("+ galois_distributed_accumulator_name + ".reduce());\n";
              rewriter.InsertText(for_each_loc_end, while_conditional, true, true);

              /**8. Adding definition for static accumulator field name **/
              string galois_accumulator_def = "\n" + galois_distributed_accumulator_type + " " + i.first + "::" + galois_distributed_accumulator_name + ";\n";
              SourceLocation main_struct_loc_end = main_struct->getSourceRange().getEnd().getLocWithOffset(2);
              rewriter.InsertText(main_struct_loc_end, galois_accumulator_def, true, true);
              break;
            }
          }
        }
      }
    }
};



/*COMMAN MATCHER STATEMENTS  :*************** To match Galois style getData calls  *************************************/
      StatementMatcher GetDataMatcher = callExpr(argumentCountIs(1), callee(methodDecl(hasName("getData"))));
      StatementMatcher LHSRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl(hasType(references(AnyType))))))).bind("LHSRefVariable");

      StatementMatcher LHSNonRefVariable = expr(ignoringParenImpCasts(declRefExpr(to(
                                        varDecl())))).bind("LHSNonRefVariable");

      /****************************************************************************************************************/


      /*COMMAN MATCHER STATEMENTS  :*************** To match Galois style for loops  *************************************/
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



      /** To construct various StatementMatchers for 2 loop transforms. **/
      template<typename Ty>
      StatementMatcher makeStatement_EdgeForStmt(Ty DM_update, string str_loopName){
        StatementMatcher returnMatcher = forStmt(hasLoopInit(anyOf(
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
                                              hasDescendant(getDataCallExprMatcher_insideFor),
                                              hasDescendant(DM_update)
                                              ).bind(str_loopName);
        return returnMatcher;
      }

class GaloisFunctionsConsumer : public ASTConsumer {
    private:
    CompilerInstance &Instance;
    std::set<std::string> ParsedTemplates;
    GaloisFunctionsVisitor* Visitor;
    MatchFinder Matchers, Matchers_doall, Matchers_op, Matchers_typedef, Matchers_gen, Matchers_gen_field, Matchers_2LT;
    FunctionCallHandler functionCallHandler;
    FindOperatorHandler findOperator;
    CallExprHandler callExprHandler;
    TypedefHandler typedefHandler;
    FindingFieldHandler f_handler;
    FindingFieldInsideForLoopHandler insideForLoop_handler;
    FieldUsedInForLoop insideForLoopField_handler;
    LoopTransformHandler loopTransform_handler;
    InfoClass info;
  public:
    GaloisFunctionsConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates, Rewriter &R): Instance(Instance), ParsedTemplates(ParsedTemplates), Visitor(new GaloisFunctionsVisitor(&Instance)), functionCallHandler(R, &info), findOperator(&Instance), callExprHandler(&Instance, &info), typedefHandler(&info), f_handler(&Instance, &info), insideForLoop_handler(&Instance, &info), insideForLoopField_handler(&Instance, &info), loopTransform_handler(R, &info) {

     /**************** Additional matchers ********************/
      //Matchers.addMatcher(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData"))), hasAncestor(binaryOperator(hasOperatorName("=")).bind("assignment"))).bind("getData"), &callExprHandler); //*** WORKS
      //Matchers.addMatcher(recordDecl(hasDescendant(callExpr(isExpansionInMainFile(), callee(functionDecl(hasName("getData")))).bind("getData"))).bind("getDataInStruct"), &callExprHandler); //**** works

      /*MATCHER 1 :*************** Matcher to get the type of node defined by user *************************************/
      //Matchers_typedef.addMatcher(typedefDecl(isExpansionInMainFile()).bind("typedefDecl"), &typedefHandler);
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
#if 0
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
            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;

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

                                                                      binaryOperator(hasOperatorName("="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))).bind(str_assignment_vec),
                                                                      binaryOperator(hasOperatorName("+="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))).bind(str_plusOp_vec),
                                                                      /** Atomic Add **/
                                                                      callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicAdd"))))), hasAnyArgument(LHS_memExpr)).bind(str_atomicAdd),
                                                                      //callExpr(argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_memExpr))).bind(str_atomicAdd),

                                                                      binaryOperator(hasOperatorName("="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub"))).bind(str_assignment),
                                                                      binaryOperator(hasOperatorName("+="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub"))).bind(str_plusOp),

                                                                     /** For vectors **/
                                                                      binaryOperator(hasOperatorName("+="), hasDescendant(operatorCallExpr().bind("forVector"))).bind("OpforVector")

                                                                     ),
                                                                     hasAncestor(recordDecl(hasName(i.first)))/*,
                                                                     unless(hasAncestor(ifStmt(hasCondition(anything()))))*/);

            Matchers_gen.addMatcher(f_1, &f_handler);
            Matchers_gen.addMatcher(f_2, &f_handler);
          }
        }
      }
#endif
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

            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME + "_" + i.first;
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME + "_" + i.first;

            string str_syncPull_var = "syncPullVar_" + j.VAR_NAME + "_" + i.first;


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

            /** REDUCTIONS : Only need syncPush : Order matters as the most generic ones should be at the last **/
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

                                                                      /** for vector operations NodeData.field[i] += val **/
                                                                      binaryOperator(hasOperatorName("="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))).bind(str_assignment_vec),
                                                                      binaryOperator(hasOperatorName("+="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))).bind(str_plusOp_vec),

                                                                      /** Atomic Add **/
                                                                      callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicAdd"))))), hasAnyArgument(LHS_memExpr)).bind(str_atomicAdd),

                                                                      binaryOperator(hasOperatorName("="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub"))).bind(str_assignment),
                                                                      binaryOperator(hasOperatorName("+="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub"))).bind(str_plusOp)
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


            /** USE but !REDUCTIONS : NodeData.field is used, therefore needs syncPull **/
            DeclarationMatcher f_syncPull_1 = varDecl(isExpansionInMainFile(), hasInitializer(expr(anyOf(
                                                                                                hasDescendant(memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)),
                                                                                                memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME)))))).bind(str_memExpr)
                                                                                                ),unless(f_2))),
                                                                      hasAncestor(recordDecl(hasName(i.first)))
                                                                  ).bind(str_syncPull_var);


            //StatementMatcher f_5 = callExpr(isExpansionInMainFile(), argumentCountIs(2), (callee(functionDecl(hasName("atomicAdd"))), hasAnyArgument(LHS_memExpr))).bind(str_atomicAdd);
            Matchers_gen.addMatcher(f_1, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_2, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_3, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_4, &insideForLoop_handler);
            Matchers_gen.addMatcher(f_syncPull_1, &insideForLoop_handler);
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

            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;

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
            StatementMatcher LHS_declRefExprVector = declRefExpr(to(varDecl(hasName(j.VAR_NAME), hasType(asString(j.VAR_TYPE)))));
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

                                                                      //callExpr(hasAnyArgument(declRefExpr(to(varDecl(hasName(j.VAR_NAME)[>, hasType(asString("class std::vector"))<]))))).bind("vectorOp"),
                                                                      /** For vectors: field[i] += val **/
                                                                      binaryOperator(hasOperatorName("+="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_declRefExpr)))).bind(str_plusOp_vec),
                                                                      binaryOperator(hasOperatorName("="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_declRefExpr)))).bind(str_assignment_vec),

                                                                       /** Galois::atomicAdd(field, val) **/
                                                                      callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicAdd"))))), hasAnyArgument(LHS_declRefExpr)).bind(str_atomicAdd),

                                                                      /** For arrays: field[i] += val **/
                                                                      binaryOperator(hasOperatorName("=") , hasLHS(arraySubscriptExpr(hasDescendant(LHS_declRefExpr)).bind("arraySub"))).bind(str_assignment),
                                                                      binaryOperator(hasOperatorName("+=") , hasLHS(arraySubscriptExpr(hasDescendant(LHS_declRefExpr)).bind("arraySub"))).bind(str_plusOp),

                                                                      /** For builtInType : field = val **/
                                                                      binaryOperator(hasOperatorName("="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_declRefExpr))).bind(str_assignment),

                                                                      /** For builtInType : field += val **/
                                                                      binaryOperator(hasOperatorName("+="),
                                                                            hasLHS(ignoringParenImpCasts(LHS_declRefExpr))).bind(str_plusOp)
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

            //StatementMatcher f_5 = forStmt(isExpansionInMainFile()).bind("arraySub");
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
          Matchers_doall.addMatcher(callExpr(callee(functionDecl(anyOf(hasName("Galois::do_all"), hasName("Galois::for_each")))),unless(isExpansionInSystemHeader()), hasAncestor(recordDecl(hasName(i.first)).bind("do_all_recordDecl"))).bind("galoisLoop"), &functionCallHandler);
        }
      }

      Matchers_doall.matchAST(Context);


      /*MATCHER 7: ********************Matchers for 2 loop transformations *******************/
      for (auto i : info.edgeData_map) {
        for(auto j : i.second) {
          if(j.IS_REFERENCED && j.IS_REFERENCE) {
            string str_memExpr = "memExpr_" + j.VAR_NAME+ "_" + i.first;
            string str_ifGreater_2loopTrans = "ifGreater_2loopTrans_" + j.VAR_NAME + "_" + i.first;
            string str_if_RHS = "if_RHS_" + j.VAR_NAME + "_" + i.first;
            string str_main_struct = "main_struct_" + i.first;
            string str_forLoop_2LT = "forLoop_2LT_" + i.first;
            string str_method_operator  = "methodDecl_" + i.first;
            string str_sdata = "sdata_" + j.VAR_NAME + "_" + i.first;
            string str_for_each = "for_each_" + j.VAR_NAME + "_" + i.first;

            StatementMatcher LHS_memExpr = memberExpr(hasDescendant(declRefExpr(to(varDecl(hasName(j.VAR_NAME))))), hasAncestor(recordDecl(hasName(i.first)))).bind(str_memExpr);

            /** For 2 loop Transforms: if nodeData.field > TOLERANCE **/
            StatementMatcher callExpr_atomicAdd = callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicAdd"))))), hasAnyArgument(LHS_memExpr));
            //StatementMatcher EdgeForLoopMatcher_withAtomicAdd = makeStatement_EdgeForStmt(callExpr_atomicAdd, str_forLoop_2LT);
            /**Assumptions: 1. first argument is always src in operator() method **/
            StatementMatcher f_2loopTrans_1 = ifStmt(isExpansionInMainFile(), hasCondition(allOf(binaryOperator(hasOperatorName(">"), hasLHS(hasDescendant(LHS_memExpr)), hasRHS(hasDescendant(memberExpr().bind(str_if_RHS)))),
                                                                                                hasAncestor(makeStatement_EdgeForStmt(callExpr_atomicAdd, str_forLoop_2LT)),
                                                                                                hasAncestor(recordDecl(hasName(i.first), hasDescendant(methodDecl(hasName("operator()"), hasAncestor(recordDecl(hasDescendant(callExpr(callee(functionDecl(hasName("for_each"))) , unless(hasDescendant(declRefExpr(to(functionDecl(hasName("workList_version")))))) ).bind(str_for_each)))) , hasParameter(0,decl().bind("src_arg")), hasDescendant(declStmt(hasDescendant(declRefExpr(to(varDecl(equalsBoundNode("src_arg")))))).bind(str_sdata))).bind(str_method_operator))).bind(str_main_struct))))).bind(str_ifGreater_2loopTrans);

            Matchers_2LT.addMatcher(f_2loopTrans_1, &loopTransform_handler);

          }
        }
      }

      Matchers_2LT.matchAST(Context);
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
