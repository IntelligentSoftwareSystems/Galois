#ifndef _PLUGIN_ANALYSIS_FIELD_USED_IN_FORLOOP_H
#define _PLUGIN_ANALYSIS_FIELD_USED_IN_FORLOOP_H

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

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
              reduceOP_entry.NODE_NAME = j.NODE_NAME;

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
                    reduceOP = "{galois::pairWiseAvg_vec(node." + j.FIELD_NAME + ", y); }";
                    resetValExpr = "{galois::resetVec(node." + j.FIELD_NAME + "); }";
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

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(ifMinOp){
                  ifMinOp->dump();
                  auto lhs = Results.Nodes.getNodeAs<clang::DeclRefExpr>(str_ifMinLHS);
                  lhs->dump();
                  // It is min operation.
                  //string reduceOP = "if(node." + j.FIELD_NAME + " > y) { node." + j.FIELD_NAME + " = y;}";
                  string reduceOP = "{ galois::min(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;

                }

                else if(whileCAS_op){
                  whileCAS_op->dump();
                  auto whileCAS_LHS = Results.Nodes.getNodeAs<clang::Stmt>(str_memExpr);
                  string reduceOP = "{galois::min(node." + j.FIELD_NAME + ", y);}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  whileCAS_LHS->dump();
                  break;
                }
                else if(atomicAdd_op){
                  llvm::outs() <<"--------------------> " << j.VAR_NAME << " : " << j.FIELD_NAME << "\n";
                  atomicAdd_op->dump();
                  //string reduceOP = "{ galois::atomicAdd(node." + j.FIELD_NAME + ", y);}";
                  string reduceOP = "pair_wise_add_array";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  string resetValExpr = "{node." + j.FIELD_NAME + " = 0 ; }";
                  reduceOP_entry.RESETVAL_EXPR = resetValExpr;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(assignmentOP_vec){
                  string reduceOP = "{node." + j.FIELD_NAME + "= y;}";
                  reduceOP_entry.OPERATION_EXPR = reduceOP;

                  info->reductionOps_map[i.first].push_back(reduceOP_entry);
                  break;
                }
                else if(plusOP_vec){
                  string reduceOP, resetValExpr;
                  reduceOP = "{galois::pairWiseAvg_vec(node." + j.FIELD_NAME + ", y); }";
                  resetValExpr = "{galois::resetVec(node." + j.FIELD_NAME + "); }";
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

#endif//_PLUGIN_ANALYSIS_FIELD_USED_IN_FORLOOP_H
