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

/*
 * ClangUtils.h
 *
 *  Created on: Jan 27, 2016
 *      Author: rashid
 */

#ifndef SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_
#define SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include <fstream>
#include <time.h>

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

struct ASTUtility {
  Rewriter* rewriter;
  ASTUtility() : rewriter(nullptr) {}
  void init(Rewriter* r) { rewriter = r; }
  std::string toString(Stmt* S) {
    std::string s;
    llvm::raw_string_ostream raw_s(s);
    S->dump(raw_s);
    return raw_s.str();
  }
  char* get_src_ptr(const clang::SourceLocation& s) {
    return const_cast<char*>(rewriter->getSourceMgr().getCharacterData(s));
  }
  void print_expr(const clang::SourceLocation& b,
                  const clang::SourceLocation& e) {
    for (char* p = get_src_ptr(b); p <= get_src_ptr(e); ++p) {
      llvm::outs() << *p;
    }
  }
  std::string get_string(const clang::SourceLocation& b,
                         const clang::SourceLocation& e) {
    char* b_ptr = get_src_ptr(b);
    char* e_ptr = get_src_ptr(e);
    std::string s(b_ptr, std::distance(b_ptr, e_ptr) + 1);
    return s;
  }
  std::string get_string(const clang::SourceRange& e) {
    char* b_ptr = get_src_ptr(e.getBegin());
    char* e_ptr = get_src_ptr(e.getEnd());
    std::string s(b_ptr, std::distance(b_ptr, e_ptr) + 1);
    return s;
  }
  void print_expr(const clang::SourceRange& c) {
    for (char* p = get_src_ptr(c.getBegin()); p <= get_src_ptr(c.getEnd());
         ++p) {
      llvm::outs() << *p;
    }
  }
  static std::string get_timestamp() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d - %X", &tstruct);

    return buf;
  }
};

#endif /* SRC_PLUGINS_OPENCLCODEGEN_CLANGUTILS_H_ */
