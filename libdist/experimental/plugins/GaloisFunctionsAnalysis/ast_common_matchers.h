/** clang analysis plugin for Galois programs: AST Comman Matchers -*- C++ -*-
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
 * @section Description: Common AST matchers.
 *
 * @author Gurbinder Gill <gurbinder533@gmail.com>
 */


#ifndef _PLUGIN_ANALYSIS_AST_COMMON_MATCHERS_H
#define _PLUGIN_ANALYSIS_AST_COMMON_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

static const TypeMatcher AnyType = anything();
/*COMMAN MATCHER STATEMENTS  :*************** To match Galois style getData calls  *************************************/
      StatementMatcher GetDataMatcher = callExpr(argumentCountIs(1), callee(methodDecl(hasName("getData")))).bind("getData_callExpr");
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

      /***Supports both Old style for and new range based for  ******/
      StatementMatcher EdgeForLoopMatcher = anyOf(forStmt(
                                                 hasLoopInit(anyOf(
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
                                              ).bind("forLoopName"),
                                              forRangeStmt(hasDescendant(getDataCallExprMatcher_insideFor)).bind("forLoopName")
                                              );
#if 0
      /***** New forRangeStmt *********/
      StatementMatcher EdgeForLoopMatcher = forRangeStmt(hasDescendant(getDataCallExprMatcher_insideFor)).bind("forLoopName"); 
#endif
      /****************************************************************************************************************/



      /** To construct various StatementMatcher for reduction operations **/
      template<typename Ty>
      StatementMatcher makeStatement_reductionOp(Ty LHS_memExpr, string i_first){
        /** REDUCTIONS : Only need syncPush : Order matters as the most generic ones should be at the last **/
        StatementMatcher f_2 = expr(isExpansionInMainFile(), anyOf(
                                                                  /** For builtInType : NodeData.field += val **/
                                                                  binaryOperator(hasOperatorName("+="),
                                                                        hasLHS(ignoringParenImpCasts(LHS_memExpr))),
                                                                  /** For builtInType : NodeData.field = NodeData.field + val **/
                                                                  binaryOperator(hasOperatorName("="),
                                                                        hasLHS(ignoringParenImpCasts(LHS_memExpr)),
                                                                        hasRHS(binaryOperator(hasOperatorName("+"),
                                                                                              hasLHS(ignoringParenImpCasts(LHS_memExpr))))),
                                                                  /** For builtInType : NodeData.field = val **/
                                                                  binaryOperator(hasOperatorName("="),
                                                                        hasLHS(ignoringParenImpCasts(LHS_memExpr))),


                                                                  /** For ComplexType : NodeData.field += val **/
                                                                  operatorCallExpr(hasOverloadedOperatorName("+="),
                                                                          hasDescendant(LHS_memExpr)),
                                                                  /** For ComplexType : NodeData.field = NodeData.field + val **/
                                                                  operatorCallExpr(hasOverloadedOperatorName("="),
                                                                          hasDescendant(binaryOperator(hasOperatorName("+"),
                                                                                                          hasLHS(hasDescendant(LHS_memExpr))))),
                                                                  /** For ComplexType : NodeData.field = val **/
                                                                  operatorCallExpr(hasOverloadedOperatorName("="),
                                                                         hasDescendant(LHS_memExpr)),

                                                                  /** for vector operations NodeData.field[i] += val **/
                                                                  binaryOperator(hasOperatorName("="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))),
                                                                  binaryOperator(hasOperatorName("+="), hasLHS(operatorCallExpr(hasDescendant(declRefExpr(to(methodDecl(hasName("operator[]"))))), hasDescendant(LHS_memExpr)))),

                                                                  /** Atomic Add **/
                                                                  callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicAdd"))))), hasAnyArgument(LHS_memExpr)),
                                                                  /** Atomic min **/
                                                                  callExpr(argumentCountIs(2), hasDescendant(declRefExpr(to(functionDecl(hasName("atomicMin"))))), hasAnyArgument(LHS_memExpr)),

                                                                  binaryOperator(hasOperatorName("="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub"))),
                                                                  binaryOperator(hasOperatorName("+="), hasDescendant(arraySubscriptExpr(hasDescendant(LHS_memExpr)).bind("arraySub")))
                                                                 ),
                                                                 hasAncestor(recordDecl(hasName(i_first))),
                                                                 /** Only if condition inside EdgeForLoop should be considered**/
                                                                 unless(hasAncestor(ifStmt(hasCondition(anything()), hasAncestor(EdgeForLoopMatcher)))));

        return f_2;
      }

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
#endif #_PLUGIN_ANALYSIS_AST_COMMON_MATCHERS_H
