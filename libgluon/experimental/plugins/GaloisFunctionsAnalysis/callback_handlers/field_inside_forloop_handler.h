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

#ifndef _PLUGIN_ANALYSIS_FIELD_INSIDE_FORLOOP_H
#define _PLUGIN_ANALYSIS_FIELD_INSIDE_FORLOOP_H

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

class FindingFieldInsideForLoopHandler : public MatchFinder::MatchCallback {
private:
  ASTContext* astContext;
  InfoClass* info;
  Rewriter& rewriter;

public:
  FindingFieldInsideForLoopHandler(CompilerInstance* CI, InfoClass* _info,
                                   Rewriter& rewriter)
      : astContext(&(CI->getASTContext())), info(_info), rewriter(rewriter) {}
  virtual void run(const MatchFinder::MatchResult& Results) {

    clang::LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    clang::PrintingPolicy Policy(LangOpts);

    for (auto i : info->edgeData_map) {
      for (auto j : i.second) {
        llvm::outs() << "Fields inside edge loop: " << j.VAR_NAME << "\n";
        string str_memExpr     = "memExpr_" + j.VAR_NAME + "_" + i.first;
        string str_assignment  = "equalOp_" + j.VAR_NAME + "_" + i.first;
        string str_plusOp      = "plusEqualOp_" + j.VAR_NAME + "_" + i.first;
        string str_minusOp     = "minusEqualOp_" + j.VAR_NAME + "_" + i.first;
        string str_assign_plus = "assignplusOp_" + j.VAR_NAME + "_" + i.first;
        string str_varDecl     = "varDecl_" + j.VAR_NAME + "_" + i.first;
        string str_varDecl_nonRef =
            "varDeclNonRef_" + j.VAR_NAME + "_" + i.first;

        string str_ifMin    = "ifMin_" + j.VAR_NAME + "_" + i.first;
        string str_ifMinRHS = "ifMinRHS_" + j.VAR_NAME + "_" + i.first;
        string str_Cond_assignment =
            "Cond_equalOp_" + j.VAR_NAME + "_" + i.first;
        string str_Cond_assignmentRHS =
            "Cond_equalOpRHS_" + j.VAR_NAME + "_" + i.first;
        string str_Cond_RHSmemExpr =
            "Cond_RHSmemExpr_" + j.VAR_NAME + "_" + i.first;
        string str_Cond_RHSVarDecl =
            "Cond_RHSVarDecl_" + j.VAR_NAME + "_" + i.first;
        string str_ifCompare = "ifCompare_" + j.VAR_NAME + "_" + i.first;

        string str_whileCAS     = "whileCAS_" + j.VAR_NAME + "_" + i.first;
        string str_whileCAS_RHS = "whileCAS_RHS" + j.VAR_NAME + "_" + i.first;

        string str_atomicAdd = "atomicAdd_" + j.VAR_NAME + "_" + i.first;
        string str_atomicMin = "atomicMin_" + j.VAR_NAME + "_" + i.first;
        string str_min       = "min_" + j.VAR_NAME + "_" + i.first;

        /** For vector operations **/
        string str_assignment_vec = "equalOpVec_" + j.VAR_NAME + "_" + i.first;
        string str_plusOp_vec = "plusEqualOpVec_" + j.VAR_NAME + "_" + i.first;

        if (j.IS_REFERENCED && j.IS_REFERENCE) {
          auto field = Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
          auto field_nonRef =
              Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl_nonRef);
          auto assignmentOP =
              Results.Nodes.getNodeAs<clang::Stmt>(str_assignment);
          auto plusOP  = Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp);
          auto minusOP = Results.Nodes.getNodeAs<clang::Stmt>(str_minusOp);
          auto assignplusOP =
              Results.Nodes.getNodeAs<clang::Stmt>(str_assign_plus);

          auto ifMinOp   = Results.Nodes.getNodeAs<clang::Stmt>(str_ifMin);
          auto ifCompare = Results.Nodes.getNodeAs<clang::Stmt>(str_ifCompare);
          auto whileCAS_op = Results.Nodes.getNodeAs<clang::Stmt>(str_whileCAS);

          auto atomicAdd_op =
              Results.Nodes.getNodeAs<clang::Stmt>(str_atomicAdd);
          auto atomicMin_op =
              Results.Nodes.getNodeAs<clang::Stmt>(str_atomicMin);
          auto min_op = Results.Nodes.getNodeAs<clang::Stmt>(str_min);

          /**Figure out variable type to set the reset value **/
          auto memExpr =
              Results.Nodes.getNodeAs<clang::MemberExpr>(str_memExpr);

          /** Vector operations **/
          auto plusOP_vec =
              Results.Nodes.getNodeAs<clang::Stmt>(str_plusOp_vec);
          auto assignmentOP_vec =
              Results.Nodes.getNodeAs<clang::Stmt>(str_assignment_vec);

          if (memExpr) {
            // memExpr->dumpColor();
            auto pType       = memExpr->getType();
            auto templatePtr = pType->getAs<TemplateSpecializationType>();
            string Ty;
            /** It is complex type e.g atomic<int>**/
            if (templatePtr) {
              templatePtr->getArg(0).getAsType().dump();
              Ty = templatePtr->getArg(0).getAsType().getAsString();
              // llvm::outs() << "TYPE FOUND ---->" << Ty << "\n";
            }
            /** It is a simple builtIn type **/
            else {
              Ty = pType->getLocallyUnqualifiedSingleStepDesugaredType()
                       .getAsString();
            }

            /** Check if type is suppose to be a vector **/
            string Type = Ty;
            if (plusOP_vec || assignmentOP_vec) {
              Type = "std::vector<" + Ty + ">";
            }
            string Num_ele = "0";
            bool is_vector = is_vector_type(Ty, Type, Num_ele);

            NodeField_entry field_entry;
            ReductionOps_entry reduceOP_entry;
            field_entry.NODE_TYPE     = j.VAR_TYPE;
            field_entry.IS_VEC        = is_vector;
            reduceOP_entry.NODE_TYPE  = j.VAR_TYPE;
            reduceOP_entry.FIELD_TYPE = j.VAR_TYPE;

            field_entry.VAR_TYPE        = field_entry.RESET_VALTYPE =
                reduceOP_entry.VAL_TYPE = Type;

            auto memExpr_stmt =
                Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
            string str_field;
            llvm::raw_string_ostream s(str_field);
            /** Returns something like snode.field_name.operator **/
            memExpr_stmt->printPretty(s, 0, Policy);

            /** Split string at delims to get the field name **/
            char delim = '.';
            vector<string> elems;
            split(s.str(), delim, elems);

            field_entry.NODE_NAME = reduceOP_entry.NODE_NAME = elems[0];
            field_entry.FIELD_NAME = reduceOP_entry.FIELD_NAME = elems[1];
            field_entry.GRAPH_NAME = reduceOP_entry.GRAPH_NAME = j.GRAPH_NAME;
            reduceOP_entry.SYNC_TYPE                           = "sync_push";

            if (field_nonRef || ifCompare) {
              SyncFlag_entry syncFlags_entry;
              syncFlags_entry.VAR_NAME   = j.VAR_NAME;
              syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
              syncFlags_entry.RW         = "read";
              syncFlags_entry.AT         = "readDestination";
              if (!syncFlags_entry_exists(syncFlags_entry,
                                          info->syncFlags_map[i.first])) {
                info->syncFlags_map[i.first].push_back(syncFlags_entry);
              }
              break;
            }

            if (minusOP) {

              string str_minusOP;
              llvm::raw_string_ostream s(str_minusOP);
              /** Returns something like snode.field_name.operator **/
              minusOP->printPretty(s, 0, Policy);
              str_minusOP = s.str();
              minusOP->dumpColor();
              llvm::outs() << "____________________________> : " << s.str()
                           << "\n";

              SourceLocation minusOP_loc = minusOP->getSourceRange().getBegin();
              unsigned len_rm =
                  minusOP->getSourceRange().getEnd().getRawEncoding() -
                  minusOP_loc.getRawEncoding() + 1;
              rewriter.RemoveText(minusOP_loc, len_rm);

              std::string new_field_name =
                  std::string("contrib_" + field_entry.FIELD_NAME);

              findAndReplace(str_minusOP, field_entry.FIELD_NAME,
                             new_field_name);
              findAndReplace(str_minusOP, "-=", "+=");

              // TODO:: Assuming reset type for field is zero.
              rewriter.InsertText(minusOP_loc, str_minusOP, true, true);

              auto varDecl_getData_src =
                  Results.Nodes.getNodeAs<clang::VarDecl>(
                      "varDecl_getData_src");
              std::string str_varName_getData_src = "";
              if (varDecl_getData_src) {
                str_varName_getData_src =
                    varDecl_getData_src->getNameAsString();
              }
              auto edgeForLoop =
                  Results.Nodes.getNodeAs<clang::Stmt>("forLoopName");
              SourceLocation edgeForLoop_loc =
                  edgeForLoop->getSourceRange().getBegin();

              // Using new field to set the old field outside the for edge loop.
              // sdata.Field1 = sdata.contrib_Field1;
              std::string str_using_new_field = "\nif(" +
                                                str_varName_getData_src + "." +
                                                new_field_name + "){";
              str_using_new_field += "\n" + str_varName_getData_src + "." +
                                     field_entry.FIELD_NAME + " = " +
                                     str_varName_getData_src + "." +
                                     new_field_name + ";";
              std::string str_reset_new_field = std::string(
                  "\ngalois::reset(" + str_varName_getData_src + "." +
                  new_field_name + " , " + field_entry.RESET_VALTYPE + "(" +
                  std::to_string(0) + ")" + ");\n}\n");

              str_using_new_field += str_reset_new_field;
              rewriter.InsertText(edgeForLoop_loc, str_using_new_field, true,
                                  true);

              // Add an entry for new variable to be synced and split operator
              // if required.
              reduceOP_entry.SYNC_TYPE      = "sync_push";
              reduceOP_entry.OPERATION_EXPR = "add";
              reduceOP_entry.RESETVAL_EXPR  = "0";
              reduceOP_entry.FIELD_NAME     = new_field_name;
              if (!syncPull_reduction_exists(reduceOP_entry,
                                             info->reductionOps_map[i.first])) {
                info->reductionOps_map[i.first].push_back(reduceOP_entry);
              }

              /** Add to the write set **/
              SyncFlag_entry syncFlags_entry;
              syncFlags_entry.FIELD_NAME = new_field_name;
              syncFlags_entry.RW         = "write";
              syncFlags_entry.AT         = "writeDestination";
              syncFlags_entry.IS_RESET   = true;
              if (!syncFlags_entry_exists(syncFlags_entry,
                                          info->syncFlags_map[i.first])) {
                info->syncFlags_map[i.first].push_back(syncFlags_entry);
              }

              // Add new variable to be constructed later.
              New_variable_entry new_variable_entry;
              new_variable_entry.FIELD_NAME = new_field_name;
              new_variable_entry.FIELD_TYPE = field_entry.RESET_VALTYPE;
              new_variable_entry.INIT_VAL   = "0";

              if (!newVariable_entry_exists(new_variable_entry,
                                            info->newVariables_map[i.first])) {
                info->newVariables_map[i.first].push_back(new_variable_entry);
              }
              break;
            }

            if (!field && (assignplusOP || plusOP || assignmentOP ||
                           atomicAdd_op || plusOP_vec || assignmentOP_vec) ||
                atomicMin_op || min_op || ifMinOp || whileCAS_op) {
              llvm::outs() << " syncFlags_entry : " << field_entry.FIELD_NAME
                           << "\n";
              SyncFlag_entry syncFlags_entry;
              syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
              syncFlags_entry.VAR_NAME   = j.VAR_NAME;
              syncFlags_entry.RW         = "write";
              syncFlags_entry.AT         = "writeDestination";
              syncFlags_entry.IS_RESET   = true;
              if (!syncFlags_entry_exists(syncFlags_entry,
                                          info->syncFlags_map[i.first])) {
                info->syncFlags_map[i.first].push_back(syncFlags_entry);
              }
            }

            if (assignplusOP) {
              // assignplusOP->dump();
              field_entry.VAR_NAME      = "+";
              field_entry.IS_REFERENCED = true;
              field_entry.IS_REFERENCE  = true;
              field_entry.RESET_VAL     = "0";

              string reduceOP               = "add";
              reduceOP_entry.OPERATION_EXPR = reduceOP;
              string resetValExpr           = "0";
              reduceOP_entry.RESETVAL_EXPR  = resetValExpr;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              // info->fieldData_map[i.first].push_back(field_entry);
              break;
            } else if (plusOP) {
              field_entry.VAR_NAME      = "+=";
              field_entry.IS_REFERENCED = true;
              field_entry.IS_REFERENCE  = true;
              field_entry.RESET_VAL     = "0";

              string reduceOP, resetValExpr;

              if (is_vector) {
                // FIXME: do not pass entire string; pass only the command like
                // add_vec
                reduceOP = "{galois::pairWiseAvg_vec(node." +
                           field_entry.FIELD_NAME + ", y); }";
                resetValExpr =
                    "{galois::resetVec(node." + field_entry.FIELD_NAME + "); }";
              } else {
                reduceOP     = "add";
                resetValExpr = "0";
              }

              reduceOP_entry.OPERATION_EXPR = reduceOP;
              reduceOP_entry.RESETVAL_EXPR  = resetValExpr;
              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              // info->fieldData_map[i.first].push_back(field_entry);
              break;

            }
            /** If this is a an expression like nodeData.fieldName = value **/
            else if (assignmentOP) {

              field_entry.VAR_NAME      = "DirectAssignment";
              field_entry.IS_REFERENCED = true;
              field_entry.IS_REFERENCE  = true;

              string reduceOP               = "set";
              reduceOP_entry.OPERATION_EXPR = reduceOP;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              // info->fieldData_map[i.first].push_back(field_entry);
              break;
            } else if (ifMinOp) {
              // ifMinOp->dump();
              auto ifMinRHS =
                  Results.Nodes.getNodeAs<clang::Stmt>(str_ifMinRHS);
              auto condAssignLHS =
                  Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
              auto condAssignRHS =
                  Results.Nodes.getNodeAs<clang::Stmt>(str_Cond_RHSmemExpr);
              auto varAssignRHS =
                  Results.Nodes.getNodeAs<clang::VarDecl>(str_varDecl);
              auto varCondRHS =
                  Results.Nodes.getNodeAs<clang::VarDecl>(str_Cond_RHSVarDecl);
              // ifMinRHS->dump();
              // condAssignLHS->dump();

              // It is min operation.
              if (varCondRHS == varAssignRHS) {
                string reduceOP               = "min";
                reduceOP_entry.OPERATION_EXPR = reduceOP;
                // varAssignRHS->dump();

                info->reductionOps_map[i.first].push_back(reduceOP_entry);
              }
              break;

            } else if (whileCAS_op) {
              auto whileCAS_LHS =
                  Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
              // whileCAS_LHS->dump();
              string reduceOP               = "min";
              reduceOP_entry.OPERATION_EXPR = reduceOP;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              // whileCAS_LHS->dump();
              break;
            }

            else if (atomicAdd_op) {
              llvm::outs()
                  << "*************************************************\n";
              llvm::outs() << i.first << "\n" << field_entry.FIELD_NAME << "\n";
              atomicAdd_op->dumpColor();

              string reduceOP               = "add";
              reduceOP_entry.OPERATION_EXPR = reduceOP;
              string resetValExpr           = "0";
              reduceOP_entry.RESETVAL_EXPR  = resetValExpr;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              break;
            }

            else if (atomicMin_op) {
              string reduceOP               = "min";
              reduceOP_entry.OPERATION_EXPR = reduceOP;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);

              /** Also needs sync pull for operator to terminate! **/
#if 0
                  ReductionOps_entry reduceOP_entry_pull = reduceOP_entry;
                  reduceOP_entry_pull.SYNC_TYPE = "sync_pull";
                  info->reductionOps_map[i.first].push_back(reduceOP_entry_pull);
#endif

              break;
            } else if (min_op) {
              string reduceOP               = "min";
              reduceOP_entry.OPERATION_EXPR = reduceOP;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);

              break;
            }

            else if (assignmentOP_vec) {
              string reduceOP               = "set";
              reduceOP_entry.OPERATION_EXPR = reduceOP;

              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              break;
            } else if (plusOP_vec) {
              string reduceOP, resetValExpr;
              // FIXME: do not pass entire string; pass only the command like
              // add_vec
              reduceOP = "{galois::pairWiseAvg_vec(node." +
                         field_entry.FIELD_NAME + ", y); }";
              resetValExpr =
                  "{galois::resetVec(node." + field_entry.FIELD_NAME + "); }";
              reduceOP_entry.OPERATION_EXPR = reduceOP;
              reduceOP_entry.RESETVAL_EXPR  = resetValExpr;
              info->reductionOps_map[i.first].push_back(reduceOP_entry);
              break;

            } else if (field) {

              field_entry.VAR_NAME = field->getNameAsString();
              field_entry.VAR_TYPE =
                  field->getType().getNonReferenceType().getAsString();
              llvm::outs() << " VAR TYPE in getdata IN loop :"
                           << field_entry.VAR_TYPE << "\n";
              field_entry.IS_REFERENCED = field->isReferenced();
              field_entry.IS_REFERENCE  = true;
              field_entry.RESET_VAL     = "0";
              field_entry.RESET_VALTYPE = field_entry.VAR_TYPE;
              field_entry.SYNC_TYPE     = "sync_push";
              info->fieldData_map[i.first].push_back(field_entry);

              /** Add to the read set **/
              SyncFlag_entry syncFlags_entry;
              syncFlags_entry.FIELD_NAME = field_entry.FIELD_NAME;
              syncFlags_entry.RW         = "read";
              syncFlags_entry.VAR_NAME   = j.VAR_NAME;
              syncFlags_entry.AT         = "readDestination";
              if (!syncFlags_entry_exists(syncFlags_entry,
                                          info->syncFlags_map[i.first])) {
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

#endif //_PLUGIN_ANALYSIS_FIELD_INSIDE_FORLOOP_H
