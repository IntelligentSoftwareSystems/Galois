/** Defines the basic functors for one and two input logic gates -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2011, The University of Texas at Austin. All rights reserved.
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
 * @author M. Amber Hassaan <ahassaan@ices.utexas.edu>
 */

#ifndef _LOGIC_FUNCTIONS_H_
#define _LOGIC_FUNCTIONS_H_

#include <functional>
#include <string>

#include "logicDefs.h"

/**
 * LogicFunc is a functor, serving as a common base type 
 * for one and two input functors.
 */
struct LogicFunc {
  virtual const std::string toString () const = 0;
};

/**
 * Interface of a functor for modeling the funciton of a one input one, output logic gate.
 * Each implementation of this interface is a different kind of one input gate, e.g. an
 * inverter, buffer etc
 */

struct OneInputFunc: public LogicFunc {
  virtual LogicVal operator () (const LogicVal& in) const = 0;
};

/**
 * Interface of a functor for modeling functionality a logic gate with two inputs and one output.
 * Each implementation of this interface describes a two input gate
 */
struct TwoInputFunc: public LogicFunc {
  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const = 0;
};

/**
 * Buffer
 */
struct BUF : public OneInputFunc, public std::unary_function<LogicVal, LogicVal> {
  LogicVal _buf_ (const LogicVal& in) const {
    return in;
  }

  virtual LogicVal operator () (const LogicVal& in) const {
    return _buf_ (in);
  }

  virtual const std::string toString () const { return "BUF"; }
};

/**
 * Inverter
 */
struct INV : public OneInputFunc, public std::unary_function<LogicVal, LogicVal> {
  LogicVal _not_ (const LogicVal& in) const {
    if (in == '0') {
      return '1';
    } else if (in == '1') {
      return '0';
    } else {
      return UNKNOWN_VALUE;
    }
  }

  virtual LogicVal operator () (const LogicVal& in) const {
    return _not_ (in);
  }

  virtual const std::string toString () const { return "INV"; }
};


/**
 * And with two inputs
 */

struct AND2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _and_ (const LogicVal& x, const LogicVal& y) const {
    if (x == '0' || y == '0') {
      return '0';

    } else if (x == '1' ) {
      return y;

    } else if (y == '1') {
      return x;

    } else {
      return UNKNOWN_VALUE;
    }

  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _and_ (x, y);
  }

  virtual const std::string toString () const { return "AND2"; }
};

/**
 * Nand with two inputs
 */
struct NAND2: public AND2 {
  LogicVal _nand_ (const LogicVal& x, const LogicVal& y) const {
    return INV()._not_ (AND2::_and_ (x, y));
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _nand_ (x, y);
  }

  virtual const std::string toString () const { return "NAND2"; }
};

/**
 * OR with two inputs
 */
struct OR2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _or_ (const LogicVal& x, const LogicVal& y) const {
    if (x == '1' || y == '1') {
      return '1';
    } else if (x == '0') {
      return y;
    } else if (y == '0') {
      return x;
    } else {
      return UNKNOWN_VALUE;
    }
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _or_ (x, y);
  }

  virtual const std::string toString () const { return "OR2"; }
};

/**
 * NOR with two inputs
 */
struct NOR2: public OR2 {
  LogicVal _nor_ (const LogicVal& x, const LogicVal& y) const {
    return INV()._not_ (OR2::_or_ (x, y));
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _nor_ (x, y);
  }

  virtual const std::string toString () const { return "NOR2"; }
};

/**
 * XOR with two inputs
 */
struct XOR2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _xor_ (const LogicVal& x, const LogicVal& y) const {
    if (x == UNKNOWN_VALUE || y == UNKNOWN_VALUE) {
      return UNKNOWN_VALUE;
    } else if (INV()._not_(x) == y) {
      return '1';
    } else if (x == y) {
      return '0';
    } else {
      return 'X';
    }
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _xor_ (x, y);
  }

  virtual const std::string toString () const { return "XOR2"; }
};

/**
 * XNOR with two inputs
 */
struct XNOR2: public XOR2 {
  LogicVal _xnor_ (const LogicVal& x, const LogicVal& y) const {
    return INV()._not_ (XOR2::_xor_ (x, y) );
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _xnor_ (x, y);
  }

  virtual const std::string toString () const { return "XNOR2"; }
};

#endif
