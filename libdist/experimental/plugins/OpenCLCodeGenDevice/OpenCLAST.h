/*
 * GaloisAST.h
 *
 *  Created on: Jan 21, 2016
 *      Author: rashid
 */
#include "clang/AST/AST.h"
#include "clang/AST/Decl.h"
#include "ClangUtils.h"
//#include "GaloisCLGen.h"
#ifndef SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_
#define SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_

extern ASTUtility ast_utility;

namespace galois {
namespace GAST {
/****************************************************************************
 *****************************************************************************/
using namespace clang;
using namespace clang::ast_matchers;
using namespace llvm;
using namespace std;

struct CLNode {
  virtual ~CLNode() {}
  virtual string toString() = 0;
};
struct VarNode : public CLNode {
protected:
  std::string type;
  std::string name;

public:
  VarNode(std::string t, std::string n) : type(t), name(n) {
    llvm::outs() << "CreatedVarDecl :[" << type << "," << name << "]\n";
  }
  virtual ~VarNode() {}
  string toString() { return type + " " + name; }
};

struct DeclNode : public CLNode {
protected:
  VarNode* var;
  CLNode* initializer;

public:
  DeclNode(VarNode* v, CLNode* init) : var(v), initializer(init) {}
  virtual ~DeclNode() {}
  virtual string toString() {
    string res = var->toString();
    if (initializer)
      res += " = " + initializer->toString() + ";";
    return res;
  }
};

struct BinaryOp : public CLNode {
  std::string op;
  CLNode* lhs;
  CLNode* rhs;
  BinaryOp(std::string o, CLNode* l, CLNode* r) : op(o), lhs(l), rhs(r) {}
  virtual ~BinaryOp() {}
  virtual string toString() { return lhs->toString() + op + rhs->toString(); }
};

struct CallNode : public CLNode {
  std::string fname;
  std::vector<CLNode*> args;
  CallNode(string f, std::vector<CLNode*>& a) : fname(f), args(a) {
    llvm::outs() << "CreatedCalNode" << fname << "(";
    for (auto a : args) {
      llvm::outs() << a->toString() << " ,";
    }
    llvm::outs() << ")\n";
  }
  virtual ~CallNode() {}
  virtual string toString() {
    string ret = fname + "(";
    for (auto a : args) {
      ret += a->toString() + ",";
    }
    ret += ")";
    return ret;
  }
};

struct CLKernel : public CLNode {
  virtual string toString() { return "__kernel"; }
};
/****************************************************************************
 *****************************************************************************/
} // namespace GAST
} // namespace galois

#endif /* SRC_PLUGINS_OPENCLCODEGEN_GALOISAST_H_ */
