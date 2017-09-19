/** clang analysis plugin for Galois programs: AST handler -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2017, The University of Texas at Austin. All rights reserved.
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
 *
 * @section Description
 *
 * Handler for fields inside ForEach edge loop.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */


#ifndef _PLUGIN_ANALYSIS_MAIN_CALL_EXPR_H
#define _PLUGIN_ANALYSIS_MAIN_CALL_EXPR_H

using namespace clang;
using namespace clang::ast_matchers;
using namespace std;

namespace{

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

}//namespace

#endif//_PLUGIN_ANALYSIS_MAIN_CALL_EXPR_H
