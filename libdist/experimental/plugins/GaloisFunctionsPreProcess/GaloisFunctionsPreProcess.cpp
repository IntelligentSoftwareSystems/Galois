/**
 * Author : Gurbinder Gill (gurbinder533@gmail.com)
 */
#include <sstream>
#include <climits>
#include <vector>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"

/**
 * Matchers
 */

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

/**
 * Rewriter
 */

#include "clang/Rewrite/Core/Rewriter.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

struct Global_entry {
  string GLOBAL_NAME;
  string GLOBAL_TYPE;
};

// TODO isn't this comparison op kind of dangerous?
struct compareForMap {
  bool operator()(const Global_entry& a, const Global_entry& b) const {
    return (a.GLOBAL_TYPE + a.GLOBAL_NAME).length() < 
             (b.GLOBAL_TYPE + b.GLOBAL_NAME).length();
  }
};

class GlobalInfo {
  public:
    vector<Global_entry> globals_vec;
    //map<string, Global_entry> globals_map;
    map<Global_entry, vector<string>, compareForMap> globals_map;
};

namespace {
  /**
   * Saves global vars found into a GlobalInfo object + push to globals
   * stack.
   */
  class FindGlobalsHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
      GlobalInfo* g_info;
    public:
      FindGlobalsHandler(Rewriter &rewriter, GlobalInfo* _g_info) :  
          rewriter(rewriter), g_info(_g_info) {}

      virtual void run(const MatchFinder::MatchResult &Results) {
        auto global_var = Results.Nodes.getNodeAs<clang::VarDecl>("globalVar");
        // get global var name + type and save it to GlobalInfo object
        // which is passed out of this matcher
        if (global_var) {
          Global_entry g_entry;
          //global_var->dumpColor();
          //llvm::outs() << "Name : " <<  global_var->getNameAsString() 
          //             << " , type : "  
          //             << global_var->getType().getNonReferenceType().getAsString() 
          //             << "\n";
          g_entry.GLOBAL_NAME = global_var->getNameAsString();
          g_entry.GLOBAL_TYPE = global_var->getType().getNonReferenceType().getAsString();
          g_info->globals_vec.push_back(g_entry);
        }
      }
  };

  /**
   * Find a structure name in a vector
   */
  bool isStructNamePresent(vector<string>& vec, string name) {
    if (vec.size() == 0) {
      return false;
    }

    for (auto& i : vec) {
      if (name == i) {
        return true;
      }
    }

    return false;
  }

  /**
   * Prepend local_ to global usages found and stored in the globals
   * info object.
   */
  class RemoveGlobalsHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
      GlobalInfo* globals_info;
    public:
      RemoveGlobalsHandler(Rewriter &rewriter, GlobalInfo* globals_info) : 
          rewriter(rewriter), globals_info(globals_info) {}

      virtual void run(const MatchFinder::MatchResult &Results) {
        // loop through globals found
        for (auto& i : globals_info->globals_vec) {
          string str_global_name = "global_" + i.GLOBAL_NAME;
          string str_global_use = "global_use_" + i.GLOBAL_NAME;
          string str_global_constructor = "global_recordDecl_constructor_" + i.GLOBAL_NAME;
          string str_global_struct = "global_recordDecl_" + i.GLOBAL_NAME;

          auto global_decl = Results.Nodes.getNodeAs<clang::DeclRefExpr>(
                                str_global_use
                              );

          if (global_decl) {
            auto global_use_struct = 
              Results.Nodes.getNodeAs<clang::CXXRecordDecl>(str_global_struct);
            auto global_recordDecl_constructor = 
              Results.Nodes.getNodeAs<clang::CXXConstructorDecl>(
                str_global_constructor
              );

            //global_decl->dumpColor();
            string struct_name =  global_use_struct->getNameAsString();
            SourceLocation rm_loc = global_decl->getSourceRange().getBegin();
            string local_global_name = "local_" + i.GLOBAL_NAME;
            // global -> local_global
            rewriter.ReplaceText(rm_loc, local_global_name);

            // save structure global is in to map
            if(!isStructNamePresent(globals_info->globals_map[i], struct_name)){
              globals_info->globals_map[i].push_back(struct_name);
            }
            // TODO why break?
            break;
          }
        }
      }
  };


  /**
   * Adds the local field to the constructor of any structure that uses
   * the global.
   */
  class EditConstructorHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
      GlobalInfo* globals_info;
    public:
      EditConstructorHandler(Rewriter &rewriter, GlobalInfo* globals_info) : 
            rewriter(rewriter), globals_info(globals_info) {}

      virtual void run(const MatchFinder::MatchResult &Results) {
        // loop through all globals
        for (auto& j : globals_info->globals_map) {
          // loop through all structures the global is in
          for (auto& i : j.second) {
            string str_struct_constructor = "struct_constructor_" + 
                                            j.first.GLOBAL_NAME + 
                                            j.first.GLOBAL_TYPE + i;
            string str_struct_field_decl = "struct_field_" + 
                                           j.first.GLOBAL_NAME + 
                                           j.first.GLOBAL_TYPE + 
                                           i;
            string str_struct_contr_parmVarDecl = "struct_constr_parmVarDecl_" +
                                                  j.first.GLOBAL_NAME + 
                                                  j.first.GLOBAL_TYPE + 
                                                  i;
            string str_ctorInitializer = "struct_ctorInitializer_" + 
                                         j.first.GLOBAL_NAME + 
                                         j.first.GLOBAL_TYPE + 
                                         i;

            auto struct_constructor = 
              Results.Nodes.getNodeAs<clang::CXXConstructorDecl>(
                str_struct_constructor
              );
            auto struct_field = 
              Results.Nodes.getNodeAs<clang::FieldDecl>(
                str_struct_field_decl
              );
            auto struct_contr_parmVarDecl = 
              Results.Nodes.getNodeAs<clang::ParmVarDecl>(
                str_struct_contr_parmVarDecl
              );
            auto struct_ctorInitializer = 
              Results.Nodes.getNodeAs<clang::CXXCtorInitializer>(
                str_ctorInitializer
              );

            // if the structure has a constructor
            if (struct_constructor) {
              // local field declaration to add to structure
              string new_local_field = j.first.GLOBAL_TYPE + " &local_" + 
                                       j.first.GLOBAL_NAME + ";\n  ";
              // add declaration of local field
              SourceLocation field_loc = 
                struct_field->getSourceRange().getBegin();
              rewriter.InsertTextAfter(field_loc, new_local_field);

              // text to add to constructor
              string new_local_field_constructor_parmVar = 
                j.first.GLOBAL_TYPE + " &_" + j.first.GLOBAL_NAME + ", ";
              string new_local_field_constructor_initializer = 
                "local_" + j.first.GLOBAL_NAME + "(_" + j.first.GLOBAL_NAME + "), ";

              // add local field text to structure constructor
              SourceLocation constructor_pamVar_loc = 
                struct_contr_parmVarDecl->getSourceRange().getBegin();
              SourceLocation constructor_initializer_loc = 
                struct_ctorInitializer->getSourceRange().getBegin();
              rewriter.InsertTextBefore(constructor_pamVar_loc, 
                                        new_local_field_constructor_parmVar);
              rewriter.InsertTextBefore(constructor_initializer_loc, 
                                        new_local_field_constructor_initializer);
              // TODO why break?
              break;
            }
          }
        }
      }
  };

  /**
   * Insert replaced global as an argument to the structure constructor.
   */
  class EditCallHandler : public MatchFinder::MatchCallback {
    private:
      Rewriter &rewriter;
      GlobalInfo* globals_info;
    public:
      EditCallHandler(Rewriter &rewriter, GlobalInfo* globals_info) : 
            rewriter(rewriter), globals_info(globals_info) {}

      virtual void run(const MatchFinder::MatchResult &Results) {
        // loop through all globals
        for (auto& j : globals_info->globals_map) {
          // loop through all structures global j is in
          for (auto& i : j.second) {
            string str_struct_constr_call = 
              "struct_constr_call_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE + i;
            string str_struct_constr_call_noUOp = 
              "struct_constr_call_noUOp_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE + i;


            auto struct_constr_declRef = 
              Results.Nodes.getNodeAs<clang::DeclRefExpr>(
                str_struct_constr_call
              );
            auto struct_constr_declRef_noUOp = 
              Results.Nodes.getNodeAs<clang::DeclRefExpr>(
                str_struct_constr_call_noUOp
              );

            if (struct_constr_declRef) {
              SourceLocation constr_declRef_loc = 
                struct_constr_declRef->
                  getSourceRange().getBegin().getLocWithOffset(-1);

              string global_var = j.first.GLOBAL_NAME + ", ";
              rewriter.InsertTextBefore(constr_declRef_loc, global_var);

              // TODO why break?
              break;
            } else if(struct_constr_declRef_noUOp) {
              SourceLocation constr_declRef_loc = 
                struct_constr_declRef_noUOp->getSourceRange().getBegin();

              string global_var = j.first.GLOBAL_NAME + ", ";
              rewriter.InsertTextBefore(constr_declRef_loc, global_var);

              // TODO why break?
              break;
            }
          }
        }
      }
  };

  /**
   * AST consumer responsible for passing over AST + running match handlers
   * above (register matchers)
   */
  class GaloisFunctionsPreProcessConsumer : public ASTConsumer {
    private:
      CompilerInstance &Instance;
      std::set<std::string> ParsedTemplates;
      MatchFinder Matchers, Matchers_removeGlobals, Matchers_EditConstructors, 
                  Matchers_editCalls;
      FindGlobalsHandler findGlobalsHandler;
      RemoveGlobalsHandler removeGlobalsHandler;
      EditConstructorHandler editConstructorHandler;
      EditCallHandler editCallHandler;
      GlobalInfo globals_info;
    public:
      GaloisFunctionsPreProcessConsumer(
            CompilerInstance &Instance, 
            std::set<std::string> ParsedTemplates, 
            Rewriter &R) : 
        Instance(Instance), ParsedTemplates(ParsedTemplates), 
        findGlobalsHandler(R, &globals_info), 
        removeGlobalsHandler(R, &globals_info), 
        editConstructorHandler(R, &globals_info), 
        editCallHandler(R, &globals_info) 
      {
        /** Find all global variables **/
        DeclarationMatcher Global_var_1 = 
          varDecl(
            isExpansionInMainFile(), 
            hasGlobalStorage(), 
            unless(anyOf(isPrivate(), isPublic(), isProtected()))
          ).bind("globalVar");

        Matchers.addMatcher(Global_var_1, &findGlobalsHandler);
      }

      virtual void HandleTranslationUnit(ASTContext &Context) {
        // find globals + save to globals_info
        Matchers.matchAST(Context);

        /*
        for(auto& i : globals_info.globals_vec) {
          llvm::outs() << i.GLOBAL_NAME << "  [Type  : " << 
                          i.GLOBAL_TYPE << "]\n";
        }
        */

        /** Find the usage of globals and corresponding structures. **/
        for (auto& i : globals_info.globals_vec) {
          string str_global_name = "global_" + i.GLOBAL_NAME;
          string str_global_use = "global_use_" + i.GLOBAL_NAME;
          string str_global_constructor = "global_recordDecl_constructor_" + 
                                          i.GLOBAL_NAME;
          string str_global_struct = "global_recordDecl_" + i.GLOBAL_NAME;

          //llvm::outs() << " Matcher for : " << str_global_name <<"\n";

          StatementMatcher expr_global_use = 
            expr(
              isExpansionInMainFile(), 
              declRefExpr(
                to(varDecl(hasName(i.GLOBAL_NAME)))
              ).bind(str_global_use),
              hasAncestor(
                recordDecl(
                  has(
                    constructorDecl(
                      hasAnyConstructorInitializer(anything())
                    ).bind(str_global_constructor)
                  )
                ).bind(str_global_struct)
              ), 
              hasAncestor(methodDecl(hasName("operator()"))) // used in operator
            );

          Matchers_removeGlobals.addMatcher(expr_global_use, 
                                            &removeGlobalsHandler);
        }

        // change global uses to local_global
        Matchers_removeGlobals.matchAST(Context);

        /*
        for(auto& j : globals_info.globals_map){
          llvm::outs() << "Global Name--> " << j.first.GLOBAL_NAME << "\n";
          for(auto& i : j.second){
            llvm::outs() << "\t --> " << i << "\n";
          }
        }
        */

        /** Edit constructors to include globals as arguments **/
        for (auto& j : globals_info.globals_map) {
          for (auto& i : j.second) {
            string str_struct_constructor = 
              "struct_constructor_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE + i;
            string str_struct_field_decl = 
              "struct_field_" + j.first.GLOBAL_NAME + j.first.GLOBAL_TYPE + i;
            string str_struct_contr_parmVarDecl = 
              "struct_constr_parmVarDecl_"+ j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE  + i;
            string str_ctorInitializer = 
              "struct_ctorInitializer_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE+ i;

            // TODO: Handle case where user has not given any constructor explicitly 
            DeclarationMatcher struct_constr_matcher = 
              recordDecl(
                isExpansionInMainFile(), 
                hasName(i), 
                has(
                  constructorDecl(
                    unless(isImplicit()), 
                    hasDescendant(
                      parmVarDecl().bind(str_struct_contr_parmVarDecl)
                    ),
                    hasAnyConstructorInitializer(
                      ctorInitializer().bind(str_ctorInitializer)
                    )
                  ).bind(str_struct_constructor)), 
                has(fieldDecl().bind(str_struct_field_decl))
              );

            Matchers_EditConstructors.addMatcher(struct_constr_matcher, 
                                                 &editConstructorHandler);
          }
        }

        // add local_global declaration to structure
        Matchers_EditConstructors.matchAST(Context);

        for (auto& j : globals_info.globals_map) {
          for (auto& i : j.second) {
            string str_struct_constr_call = 
              "struct_constr_call_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE + i;
            string str_struct_constr_call_noUOp = 
              "struct_constr_call_noUOp_" + j.first.GLOBAL_NAME + 
              j.first.GLOBAL_TYPE+ i;

            // finds constructor call of structure
            StatementMatcher callSite_matcher = 
              expr(
                isExpansionInMainFile(), 
                materializeTemporaryExpr(
                  isExpansionInMainFile(), 
                  hasDescendant(
                    constructExpr(
                      hasType(recordDecl(hasName(i))), 
                      anyOf(
                        hasDescendant(
                          unaryOperator(
                            anyOf(
                              hasOperatorName("&"), 
                              hasOperatorName("*")
                            ),
                            hasDescendant(
                              declRefExpr().bind(str_struct_constr_call)
                            )
                          )
                        ),
                        hasDescendant(
                          declRefExpr().bind(str_struct_constr_call_noUOp)
                        )
                      )
                    )
                  )
                )
              );

            Matchers_editCalls.addMatcher(callSite_matcher, &editCallHandler);
          }
        }

        // add global as constructor argument
        Matchers_editCalls.matchAST(Context);
      }
  };

  /**
   * Registers the PreProcess consumer + handles calls for writing changes
   */
  class GaloisFunctionsPreProcessAction : public PluginASTAction {
  private:
    std::set<std::string> ParsedTemplates;
    Rewriter TheRewriter;
  protected:
    void EndSourceFileAction() override {
      TheRewriter.getEditBuffer(
        TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs()
      );

      if (!TheRewriter.overwriteChangedFiles()) {
        llvm::outs() << "Successfully saved changes\n";
      }
    }

    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
                                                   llvm::StringRef) override {
      TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
      return llvm::make_unique<GaloisFunctionsPreProcessConsumer>(CI, 
                ParsedTemplates, TheRewriter);

    }

    bool ParseArgs(const CompilerInstance &CI, 
                   const std::vector<std::string> &args) override {
      return true;
    }
  };
}

// Registration of compiler pass plugin
static FrontendPluginRegistry::Add<GaloisFunctionsPreProcessAction> X(
  "galois-preProcess", "replace globals with local variables");
