/** clang analysis plugin for Galois programs: AST handler -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
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
 * @section Description: Handler for fields outside ForEach edge loop.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */


#ifndef _PLUGIN_ANALYSIS_FIELD_OUTSIDE_FORLOOP_H
#define _PLUGIN_ANALYSIS_FIELD_OUTSIDE_FORLOOP_H

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

namespace {
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
            /** For making the read set at the source **/
            string str_varDecl_nonRef = "varDeclNonRef_" + j.VAR_NAME+ "_" + i.first;
            string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;
            string str_atomicMin = "atomicMin_" + j.VAR_NAME + "_" + i.first;
            string str_min = "min_" + j.VAR_NAME + "_" + i.first;


            string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_assignment_vec = "equalOpVec_" + j.VAR_NAME+ "_" + i.first;
            string str_ifCompare = "ifCompare_" + j.VAR_NAME+ "_" + i.first;

            if(j.IS_REFERENCED && j.IS_REFERENCE){
              auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto field_nonRef = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl_nonRef);
              auto assignmentOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
              auto plusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
              auto assignplusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);
              auto atomicAdd_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);
              auto atomicMin_op = Results.Nodes.getNodeAs<clang::Stmt>(str_atomicMin);
              auto min_op = Results.Nodes.getNodeAs<clang::Stmt>(str_min);
              auto ifCompare = Results.Nodes.getNodeAs<clang::Stmt>(str_ifCompare);

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
                reduceOP_entry.NODE_NAME = field_entry.NODE_NAME = elems[0];

                field_entry.FIELD_NAME = reduceOP_entry.FIELD_NAME = elems[1];
                field_entry.GRAPH_NAME = reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME;
                /** non ref field for read entry **/
                if(field_nonRef || ifCompare){
                  SyncFlag_entry syncFlags_entry;
                  syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
                  syncFlags_entry.RW = "read";
                  syncFlags_entry.AT = "source";
                  if(!syncFlags_entry_exists(syncFlags_entry, info->syncFlags_map[i.first])){
                    info->syncFlags_map[i.first].push_back(syncFlags_entry);
                  }
                  break;
                }

                if(assignmentOP_vec) {
                  //assignmentOP_vec->dump();
                }
                if(!field && (assignplusOP || plusOP || assignmentOP || atomicAdd_op || plusOP_vec || assignmentOP_vec) || atomicMin_op || min_op) {
                  reduceOP_entry.SYNC_TYPE = "sync_pull_maybe";

                  if(assignplusOP){
                    reduceOP_entry.OPERATION_EXPR = "add";
                    reduceOP_entry.RESETVAL_EXPR = "0";
                  }
                  else if(plusOP){
                    reduceOP_entry.OPERATION_EXPR = "add";
                    reduceOP_entry.RESETVAL_EXPR = "0";
                  }
                  else if(assignmentOP){
                    reduceOP_entry.OPERATION_EXPR = "set";
                  }
                  else if(atomicAdd_op){
                    reduceOP_entry.OPERATION_EXPR = "add";
                    reduceOP_entry.RESETVAL_EXPR = "0";
                  }
                  else if(atomicMin_op){
                    reduceOP_entry.OPERATION_EXPR = "min";
                  }
                  else if(min_op){
                    reduceOP_entry.OPERATION_EXPR = "min";
                  }
                  else if(assignmentOP_vec){
                    reduceOP_entry.OPERATION_EXPR = "set";
                  }
                  else if(plusOP_vec){
                    string reduceOP, resetValExpr;
                    // FIXME: do not pass entire string; pass only the command like add_vec
                    reduceOP = "{Galois::pairWiseAvg_vec(node." + field_entry.FIELD_NAME + ", y); }";
                    resetValExpr = "{Galois::resetVec(node." + field_entry.FIELD_NAME + "); }";
                    reduceOP_entry.OPERATION_EXPR = reduceOP;
                    reduceOP_entry.RESETVAL_EXPR = resetValExpr;
                  }

                  if(!syncPull_reduction_exists(reduceOP_entry, info->reductionOps_map[i.first])){
                    info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  }

                  /** Add to the write set **/
                  SyncFlag_entry syncFlags_entry;
                  syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
                  syncFlags_entry.RW = "write";
                  syncFlags_entry.AT = "source";
                  if(!syncFlags_entry_exists(syncFlags_entry, info->syncFlags_map[i.first])){
                    info->syncFlags_map[i.first].push_back(syncFlags_entry);
                  }

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
                  //llvm::outs() << " VAR TYPE in getdata not in loop :" << field_entry.VAR_TYPE << "\n";
                  field_entry.IS_REFERENCED = field->isReferenced();
                  field_entry.IS_REFERENCE = true;
                  field_entry.GRAPH_NAME = j.GRAPH_NAME;
                  field_entry.RESET_VAL = "0";
                  field_entry.SYNC_TYPE = "sync_pull_maybe";
                  info->fieldData_map[i.first].push_back(field_entry);

                  /** Add to the read set **/
                  SyncFlag_entry syncFlags_entry;
                  syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
                  syncFlags_entry.RW = "read";
                  syncFlags_entry.AT = "source";
                  if(!syncFlags_entry_exists(syncFlags_entry, info->syncFlags_map[i.first])){
                    info->syncFlags_map[i.first].push_back(syncFlags_entry);
                  }

                  break;
                }
              }
            }
          }
        }
      }
};
}//namespace

#endif// _PLUGIN_ANALYSIS_FIELD_OUTSIDE_FORLOOP_H
