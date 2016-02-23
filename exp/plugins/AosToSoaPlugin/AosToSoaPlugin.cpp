/*
 * Author : Rashid Kaleem (rashid.kaleem@gmail.com)
 */

#include <sstream>
#include <climits>
#include <vector>
#include <map>
#include <regex>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Basic/SourceManager.h"

/*
 *Matchers
 */

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

class ConvertStructureDefinition {
public:
//   static void convert(CXXRecordDecl * record){
//      for (auto f : record->fields()){
//         f->get
//      }
//   }
};
class InitHandler: public MatchFinder::MatchCallback {
public:
   Rewriter & rewriter;
   InitHandler(Rewriter & r) :
         rewriter(r) {

   }
   string toString(Stmt * S) {
      string s;
      llvm::raw_string_ostream raw_s(s);
      S->dump(raw_s);
      return raw_s.str();
   }
   char * get_src_ptr(const clang::SourceLocation & s) {
      return const_cast<char *>(rewriter.getSourceMgr().getCharacterData(s));
   }
   void print_expr(const clang::SourceLocation & b, const clang::SourceLocation & e) {
      for (char * p = get_src_ptr(b); p <= get_src_ptr(e); ++p) {
         llvm::outs() << *p;
      }
   }
   std::string get_string(const clang::SourceLocation & b, const clang::SourceLocation & e){
      char * b_ptr = get_src_ptr(b);
      char * e_ptr = get_src_ptr(e);
      std::string s (b_ptr, std::distance(b_ptr, e_ptr));
      return s;
   }
   void print_expr(const clang::SourceRange & c) {
      for (char * p = get_src_ptr(c.getBegin()); p <= get_src_ptr(c.getEnd()); ++p) {
         llvm::outs() << *p;
      }
   }
   virtual void run(const MatchFinder::MatchResult & results) {
      llvm::outs() << "InitHandler::matchfound\n";
      VarDecl * decl = const_cast<VarDecl *>(results.Nodes.getNodeAs<VarDecl>("decl"));
      decl->dump(llvm::outs());
      CXXNewExpr * initializerExpr = const_cast<CXXNewExpr*>(results.Nodes.getNodeAs<CXXNewExpr>("op"));
      std::string size_expression;
      {
         char * ptr = get_src_ptr(initializerExpr->getLocStart());
         char * end_ptr = get_src_ptr(initializerExpr->getLocEnd());
         std::string expr_string(ptr, std::distance(ptr, end_ptr));
         int b_idx = expr_string.find('[');
         int e_idx = expr_string.rfind(']');
         size_expression = expr_string.substr(b_idx + 1, e_idx - b_idx - 1);
         llvm::outs() << " Parsed size expression " << size_expression << "\n";
      }
      llvm::outs() << "Initializer expression :: \n";
      print_expr(initializerExpr->getStartLoc(), initializerExpr->getEndLoc());
      llvm::outs() << "\nDeclaration for  " << decl->getNameAsString() << "of type " << decl->getType()->getTypeClassName() << "\n";
      CXXRecordDecl * typeDecl = nullptr;
      std::string decl_string = get_string(decl->getLocStart(), decl->getLocEnd());
      if (decl->getType()->isPointerType()) {
         typeDecl = const_cast<CXXRecordDecl*>(decl->getType()->getPointeeCXXRecordDecl());
         decl_string[decl_string.find('*')]=' ';
      } else {
         //Should never happen since it should always be assigned to a pointer.
         typeDecl = const_cast<CXXRecordDecl*>(decl->getType()->getAsCXXRecordDecl());
         decl_string[decl_string.find('[')]=' ';
         decl_string[decl_string.find(']')]=' ';
      }
      rewriter.ReplaceText(decl->getSourceRange(), decl_string);

      vector<std::string> field_inits;
//      std::string array_size = initializerExpr->getArraySize()->getType().getAsString()
      for (auto f : typeDecl->fields()) {
         llvm::outs() << "Field declaration :: " << f->getNameAsString() << ", " << "" << "\n";
         QualType q = typeDecl->getASTContext().getPointerType(f->getType());
         QualType old_type = f->getType();
         f->setType(q);
         rewriter.ReplaceText(f->getTypeSourceInfo()->getTypeLoc().getSourceRange(), q.getAsString());
         char arr[1024];
         sprintf(arr, "\n%s.%s = new %s [ %s ];", decl->getNameAsString().c_str(), f->getNameAsString().c_str(), old_type.getAsString().c_str(), size_expression.c_str());
         field_inits.push_back(std::string(arr));
         llvm::outs() << arr << " \n";
//         rewriter.InsertText(f->getTypeSourceInfo()->getTypeLoc().getLocStart(), "?");
//         rewriter.InsertText(f->getTypeSourceInfo()->getTypeLoc().getEndLoc(), "*");
      }
      print_expr(decl->getLocStart(), decl->getLocEnd());

      {
         if (decl->getType().getTypePtr()->isPointerType()) {


         } else if (decl->getType().getTypePtr()->isArrayType()) {

         }

         llvm::outs() << "##"
               << QualType::getAsString(
                     decl->getType().getTypePtr()->getPointeeType().getTypePtr()->getUnqualifiedDesugaredType()->getCanonicalTypeInternal().getSplitDesugaredType()) << "##\n";
//         rewriter.ReplaceText(decl->getTypeSourceInfo()->getTypeLoc().getSourceRange(),
//               decl->getType().getTypePtr()->getPointeeType().getTypePtr()->getLocallyUnqualifiedSingleStepDesugaredType().getAsString().c_str());
//         rewriter.ReplaceText(decl->getTypeSourceInfo()->getTypeLoc().getSourceRange());
         rewriter.ReplaceText(initializerExpr->getSourceRange(), "\b;\n");
         for (auto s : field_inits) {
            rewriter.InsertTextAfter(decl->getLocEnd(), s.c_str());
         }
      }
      llvm::outs() << "Inithandler conversion complete\n";
   }
};
/////////////////////////////////////////////////////////////////////////
class SubscriptHandler: public MatchFinder::MatchCallback {
public:
   Rewriter &rewriter;
   std::vector<RecordDecl*> varDecls;
public:
   SubscriptHandler(Rewriter &rewriter) :
         rewriter(rewriter) {
   }
   string toString(Stmt * S) {
      string s;
      llvm::raw_string_ostream raw_s(s);
      S->dump(raw_s);
      return raw_s.str();
   }
   char * get_src_ptr(const clang::SourceLocation & s) {
      return const_cast<char *>(rewriter.getSourceMgr().getCharacterData(s));
   }
   void print_expr(const clang::SourceLocation & b, const clang::SourceLocation & e) {
      for (char * p = get_src_ptr(b); p <= get_src_ptr(e); ++p) {
         llvm::outs() << *p;
      }
   }
   void print_expr(const clang::SourceRange & c) {
      for (char * p = get_src_ptr(c.getBegin()); p <= get_src_ptr(c.getEnd()); ++p) {
         llvm::outs() << *p;
      }
   }
   void convert(const CXXRecordDecl * record) {
      for (auto f : record->fields()) {
         llvm::outs() << "###Field found :: ";
         print_expr(f->getSourceRange());
      }
   }
   /*
    *
    * */
   virtual void run(const MatchFinder::MatchResult &Results) {
      string new_text = "";
      string index_expression = "";
      MemberExpr * expr = const_cast<MemberExpr*>(Results.Nodes.getNodeAs<MemberExpr>("subscript"));
      const ValueDecl * decl = expr->getMemberDecl();
      const CXXRecordDecl * declExpr = expr->getBase()->getBestDynamicClassType();
      if (declExpr) {
         llvm::outs() << "DECL:: \n";
         print_expr(declExpr->getLocStart(), declExpr->getLocEnd());
         llvm::outs() << "\n";
         convert(declExpr);
      }
      llvm::outs() << "Declaration found :: " << decl->getNameAsString() << "\n";
      for (char * p = get_src_ptr(decl->getCanonicalDecl()->getLocStart()); p != get_src_ptr(decl->getCanonicalDecl()->getLocEnd()); ++p) {
         llvm::outs() << *p;
      }
      llvm::outs() << "\nEnd\n";
      ArraySubscriptExpr * idx_expr = const_cast<ArraySubscriptExpr*>(Results.Nodes.getNodeAs<ArraySubscriptExpr>("index"));
      string field_name = expr->getMemberNameInfo().getAsString();
      llvm::outs() << " Field name :: " << field_name << "\n";

      char * ptr_to_expr = get_src_ptr(idx_expr->getLocStart());
      char * ptr_to_lparen = get_src_ptr(idx_expr->getLocEnd());

      char * op_loc = get_src_ptr(expr->getOperatorLoc());
      char * expr_start = get_src_ptr(expr->getLocStart());
      char * expr_end = get_src_ptr(expr->getLocEnd());
      char * expr_mem_end = get_src_ptr(expr->getMemberLoc()) + expr->getMemberNameInfo().getAsString().length();
      {
         llvm::outs() << "EXPR:: ";
         char * p = expr_start;
         while (p != expr_mem_end) {
            if (p == op_loc) {
               llvm::outs() << "OPLOC";
            }
            llvm::outs() << *p;
            p++;
         }
         llvm::outs() << "\n";
      }

      {
         char * loc_star = get_src_ptr(idx_expr->getLocStart());
         char * loc_end = get_src_ptr(idx_expr->getLocEnd());
         llvm::outs() << "IndexExpression :: ";
         for (char * p = loc_star; p <= loc_end; ++p)
            llvm::outs() << *p;
         llvm::outs() << "\n";
         std::string s(loc_star, loc_end + 1);

         llvm::outs() << "IndexItself:: ";
         size_t last_idx = s.find(']');
         if (last_idx != string::npos) {
            for (auto p = s.find('[') + 1; p != last_idx; p++) {
               llvm::outs() << s[p];
            }
            index_expression.append(s.substr(s.find('[') + 1, last_idx - 1 - s.find('[')));
         }
         llvm::outs() << "\n" << index_expression << "\n";
      }
      {

      }

      llvm::outs() << " IndexExprBase:: ";
      char * p = ptr_to_expr;
      for (; p != ptr_to_lparen && *p != '['; p++) {
         llvm::outs() << *p;
      }
      new_text.append(ptr_to_expr, std::distance(ptr_to_expr, p));
      new_text.append(".");
      new_text.append(field_name);
      new_text.append("[");
      new_text.append(index_expression);
      new_text.append("]");
      llvm::outs() << "  \n";
      llvm::outs() << " Expr-LHS :: " << idx_expr->getLHS()->getExprLoc().printToString(rewriter.getSourceMgr()) << "\n";
      llvm::outs() << " Expr-RHS :: " << idx_expr->getRHS()->getExprLoc().printToString(rewriter.getSourceMgr()) << "\n";
      //const char * expr_char = rewriter.getSourceMgr().getCharacterData(expr->getLocStart());
      //llvm::outs()<<" Char Data :: " << expr_char << "\nEND\n";
      //expr->setBase(idx_expr->getBase());
      Expr * subscript_expr = idx_expr->getIdx();
      llvm::outs() << " MemberExpr Start :: " << expr->getLocStart().printToString(rewriter.getSourceMgr()) << " ,  End :: "
            << expr->getLocEnd().printToString(rewriter.getSourceMgr()) << "\n";

      llvm::outs() << " MemberExpr :: " << expr->getExprLoc().printToString(rewriter.getSourceMgr()) << "\n";
      llvm::outs() << " IndexExpr :: " << idx_expr->getExprLoc().printToString(rewriter.getSourceMgr()) << "\n";
      llvm::outs() << "Matched:: " << expr->getMemberNameInfo().getAsString() << "\n" << toString(expr) << "\n";
      llvm::outs() << "IndexExpr::" << toString(idx_expr) << "\n";
      SourceRange exprRange(expr->getLocStart(), expr->getLocEnd());
      rewriter.ReplaceText(exprRange, new_text);
//      rewriter.ReplaceText(subscript_expr->getExprLoc(), "");
   }
};
/* ************************************************************************************************
 *
 * *************************************************************************************************/

class ArrayofStructConsumer: public ASTConsumer {
private:
   CompilerInstance &Instance;
   std::set<std::string> ParsedTemplates;
   MatchFinder Matchers;
   Rewriter & R;
//   AoS2SoAHandler aso2soa_handler;

public:
   ArrayofStructConsumer(CompilerInstance &Instance, std::set<std::string> &ParsedTemplates, Rewriter &_R) :
         Instance(Instance), ParsedTemplates(ParsedTemplates), R(_R) {
      llvm::outs() << "=============================Created ArrayofStructConsumer===========================\n";

   }
   /********************************************************************************************************
    * 1) Find all the do_all calls in the main-file.
    * 2) For each do_all call, go over the class that defines the operator
    * 3) Add call to the generated operator implementation in OpenCL
    *
    ********************************************************************************************************/

   virtual void HandleTranslationUnit(ASTContext &Context) {
      llvm::outs() << "=============================InHandleTranslationUnit===========================\n";
      {
         MatchFinder memberCall;
         SubscriptHandler sub_handler(R);
         memberCall.addMatcher(memberExpr(isExpansionInMainFile(), hasObjectExpression(arraySubscriptExpr().bind("index"))).bind("subscript"), &sub_handler);
         //memberCall.matchAST(Context);
         /**/
         MatchFinder init_exprs;
         InitHandler init_handler(R);
         init_exprs.addMatcher(varDecl(isExpansionInMainFile(), hasInitializer(newExpr().bind("op"))).bind("decl"), &init_handler);
         init_exprs.matchAST(Context);
      }
   }

};
/********************************************************************************************************
 * The plugin AST action.
 *
 ********************************************************************************************************/
class OpenCLCodeGenAction: public PluginASTAction {
private:
   std::set<std::string> ParsedTemplates;
   Rewriter TheRewriter;
protected:

   void EndSourceFileAction() override {
      llvm::outs() << "EndSourceFileAction() :: Entered\n" << "ParseTemplates:: " << ParsedTemplates.size() << "\n";
      for (auto s : ParsedTemplates) {
         llvm::outs() << "PARSED-TEMPLATE:: " << s << "\n";
      }
      //Write the transformed AST to llvm::outs()
      //TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
      if (!TheRewriter.overwriteChangedFiles()) {
         llvm::outs() << "Successfully saved changes\n";
      }
      llvm::outs() << "EndSourceFileAction() :: Returning\n";
   }
   std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
      llvm::outs() << "CreateASTConsumer() :: Entered\n";
      TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
      llvm::outs() << "CreateASTConsumer() :: Returning\n";
//      return llvm::make_unique<OpenCLCodeGenConsumer>(CI, ParsedTemplates, TheRewriter);
      return llvm::make_unique<ArrayofStructConsumer>(CI, ParsedTemplates, TheRewriter);

   }

   bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
      return true;
   }

};

} // End anonymous namespace
/********************************************************************************************************
 * Register the plugin. The plugin can be invoked by
 * ./clang -cc1 -load ../lib/OpenCLCodeGen.so -plugin opencl-analysis <FILENAME>
 *
 ********************************************************************************************************/
static FrontendPluginRegistry::Add<OpenCLCodeGenAction> X("aos2soa", "Convert an array of structs to struct of arrays.");
