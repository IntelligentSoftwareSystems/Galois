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
 * @section Description: Handler for 2 loop transform.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */


#ifndef _PLUGIN_ANALYSIS_TWO_LOOP_TRANSFORM_H
#define _PLUGIN_ANALYSIS_TWO_LOOP_TRANSFORM_H

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

namespace{

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
          if(j.IS_REFERENCED && j.IS_REFERENCE){
            string str_ifGreater_2loopTrans = "ifGreater_2loopTrans_" + j.VAR_NAME + "_" + i.first;
            string str_main_struct = "main_struct_" + i.first;
            string str_forLoop_2LT = "forLoop_2LT_" + i.first;
            string str_method_operator  = "methodDecl_" + i.first;
            string str_sdata = "sdata_" + j.VAR_NAME + "_" + i.first;
            string str_for_each = "for_each_" + j.VAR_NAME + "_" + i.first;

            auto ifGreater_2loopTrans = Results.Nodes.getNodeAs<clang::IfStmt>(str_ifGreater_2loopTrans);
            if(ifGreater_2loopTrans){
              FirstIter_struct_entry first_itr_entry;
              SourceLocation if_loc = ifGreater_2loopTrans->getSourceRange().getBegin();

              //TODO: Assuming if statement has parenthesis. NOT NECESSARY!!
              /**1: remove if statement (order matters) **/
              unsigned len_rm = ifGreater_2loopTrans->getSourceRange().getEnd().getRawEncoding() -  if_loc.getRawEncoding() + 1;
              rewriter.RemoveText(if_loc, len_rm);

              /**2: extract operator body without if statement (order matters) **/
              auto method_operator = Results.Nodes.getNodeAs<clang::CXXMethodDecl>(str_method_operator);
              // remove UserContext : assuming it is 2nd argument
              assert(method_operator->getNumParams() == 2);
              auto OP_parm_itr = method_operator->param_begin();
              auto first_op_length = (*OP_parm_itr)->getNameAsString().length();
              auto first_end_loc = (*OP_parm_itr)->getSourceRange().getEnd();
              auto first_end = first_end_loc.getRawEncoding();
              auto second_end = method_operator->getParamDecl(1)->getSourceRange().getEnd().getRawEncoding();
              rewriter.RemoveText(first_end_loc.getLocWithOffset(first_op_length), second_end - first_end);
              string operator_body = rewriter.getRewrittenText(method_operator->getSourceRange());
              first_itr_entry.OPERATOR_BODY = operator_body;
              
              /**Adding Distributed Galois accumulator **/
              SourceLocation method_operator_begin = method_operator->getSourceRange().getBegin();
              rewriter.InsertTextBefore(method_operator_begin, ("static " + galois_distributed_accumulator_type + galois_distributed_accumulator_name + ";\n"));


              /**3: add new if statement (order matters) **/
              /** Adding new condtional outside for loop **/
              auto forLoop_2LT = Results.Nodes.getNodeAs<clang::Stmt>(str_forLoop_2LT);
              auto sdata_declStmt = Results.Nodes.getNodeAs<clang::Stmt>(str_sdata);
              SourceLocation for_loc_end = forLoop_2LT->getSourceRange().getEnd().getLocWithOffset(2);

              /**Add galois accumulator += (work is done) **/
              string work_done = "\n" + galois_distributed_accumulator_name + "+= 1;\n";
              rewriter.InsertTextAfter(for_loc_end, work_done);

              //TODO: change sdata_loc getLocWithOffset from 2 to total length of statement.
              SourceLocation sdata_loc = sdata_declStmt->getSourceRange().getEnd().getLocWithOffset(2);
              auto binary_op = dyn_cast<clang::BinaryOperator>(ifGreater_2loopTrans->getCond());
              string str_binary_op_arg;
              llvm::raw_string_ostream s_binary_op(str_binary_op_arg);
              binary_op->printPretty(s_binary_op, 0, Policy);
              string str_binary_op = s_binary_op.str();
              findAndReplace(str_binary_op, j.VAR_NAME, info->getData_map[i.first][0].VAR_NAME);
              string new_condition = "\nif(" + str_binary_op + "){\n";
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
              string operator_arg = (*OP_parm_itr)->getType().getAsString() + "  " + (*OP_parm_itr)->getNameAsString();
              first_itr_entry.OPERATOR_ARG = operator_arg;

              /**6: Get the range of the operator call **/
              auto for_each_node = Results.Nodes.getNodeAs<clang::Stmt>(str_for_each);
              string for_each_text_arg;
              llvm::raw_string_ostream s_for_each_text(for_each_text_arg);
              for_each_node->printPretty(s_for_each_text, 0, Policy);
              string for_each_text = s_for_each_text.str();
              size_t begin = for_each_text.find("(");
              size_t end = for_each_text.find(i.first, begin);
              string operator_range = for_each_text.substr(begin+1, end - begin - 1);
              first_itr_entry.OPERATOR_RANGE = operator_range;

              info->FirstItr_struct_map[i.first].push_back(first_itr_entry);

              /**6: Put functor for first iteration. All nodes need to be processed in the first iteration. **/
              string firstItr_func = makeFunctorFirstIter(i.first, first_itr_entry, info->reductionOps_map[i.first]);
              SourceLocation main_struct_loc_begin = main_struct->getSourceRange().getBegin();
              rewriter.InsertText(main_struct_loc_begin, firstItr_func, true, true);

              /**7: call functor for first iteration in the main functor and do while of for_each ***/
              SourceLocation for_each_loc_begin = for_each_node->getSourceRange().getBegin();
              SourceLocation for_each_loc_end = for_each_node->getSourceRange().getEnd().getLocWithOffset(2);
              //TODO:: get _graph from go struct IMPORTANT
              // change for_each to do_all
              string galois_foreach = "Galois::for_each(";
              string galois_doall = "Galois::do_all(_graph.begin(), _graph.end(), ";
              rewriter.ReplaceText(for_each_loc_begin, galois_foreach.length() + operator_range.length(), galois_doall);
              string num_run = ", Galois::numrun(_graph.get_run_identifier())";
              rewriter.InsertText(for_each_loc_end.getLocWithOffset(-2), num_run, true, true);
              string firstItr_func_call = "\nFirstItr_" + i.first + "::go(_graph);\n";
              string iteration = "\nunsigned _num_iterations = 1;\n";
              size_t found = operator_range.find(",");
              assert(found != string::npos);
              if (operator_range.find(",", found+1) == string::npos) // single-source
                iteration += "\nunsigned long _num_work_items = 1;\n";
              else
                iteration += "\nunsigned long _num_work_items = _graph.end() - _graph.begin();\n";
              string do_while = firstItr_func_call + iteration + "do { \n _graph.set_num_iter(_num_iterations);\n" +  galois_distributed_accumulator_name + ".reset();\n";
              rewriter.InsertText(for_each_loc_begin, do_while, true, true);

              string iteration_inc = "++_num_iterations;\n";
              iteration_inc += "_num_work_items += " + galois_distributed_accumulator_name + ".read();\n";
              string while_conditional = "}while("+ galois_distributed_accumulator_name + ".reduce());\n";
              string report_iteration = "Galois::Runtime::reportStat(\"(NULL)\", \"NUM_ITERATIONS_\" + std::to_string(_graph.get_run_num()), (unsigned long)_num_iterations, 0);\n";
              report_iteration += "Galois::Runtime::reportStat(\"(NULL)\", \"NUM_WORK_ITEMS_\" + std::to_string(_graph.get_run_num()), (unsigned long)_num_work_items, 0);\n";
              rewriter.InsertText(for_each_loc_end, iteration_inc + while_conditional + report_iteration, true, true);

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

}//namespace
#endif//_PLUGIN_ANALYSIS_TWO_LOOP_TRANSFORM_H
