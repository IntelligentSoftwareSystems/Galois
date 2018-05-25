/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef DES_LOGIC_FUNCTIONS_H_
#define DES_LOGIC_FUNCTIONS_H_

#include <functional>
#include <string>

#include "logicDefs.h"


namespace des {

/**
 * LogicFunc is a functor, serving as a common base type 
 * for one and two input functors.
 */
struct LogicFunc {
  virtual const std::string str () const = 0;
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

  virtual const std::string str () const { return "BUF"; }
};

/**
 * Inverter
 */
struct INV : public OneInputFunc, public std::unary_function<LogicVal, LogicVal> {
  LogicVal _not_ (const LogicVal& in) const {
    if (in == LOGIC_ZERO) {
      return LOGIC_ONE;
    } else if (in == LOGIC_ONE) {
      return LOGIC_ZERO;
    } else {
      return LOGIC_UNKNOWN;
    }
  }

  virtual LogicVal operator () (const LogicVal& in) const {
    return _not_ (in);
  }

  virtual const std::string str () const { return "INV"; }
};


/**
 * And with two inputs
 */

struct AND2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _and_ (const LogicVal& x, const LogicVal& y) const {
    if (x == LOGIC_ZERO || y == LOGIC_ZERO) {
      return LOGIC_ZERO;

    } else if (x == LOGIC_ONE ) {
      return y;

    } else if (y == LOGIC_ONE) {
      return x;

    } else {
      return LOGIC_UNKNOWN;
    }

  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _and_ (x, y);
  }

  virtual const std::string str () const { return "AND2"; }
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

  virtual const std::string str () const { return "NAND2"; }
};

/**
 * OR with two inputs
 */
struct OR2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _or_ (const LogicVal& x, const LogicVal& y) const {
    if (x == LOGIC_ONE || y == LOGIC_ONE) {
      return LOGIC_ONE;
    } else if (x == LOGIC_ZERO) {
      return y;
    } else if (y == LOGIC_ZERO) {
      return x;
    } else {
      return LOGIC_UNKNOWN;
    }
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _or_ (x, y);
  }

  virtual const std::string str () const { return "OR2"; }
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

  virtual const std::string str () const { return "NOR2"; }
};

/**
 * XOR with two inputs
 */
struct XOR2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _xor_ (const LogicVal& x, const LogicVal& y) const {
    if (x == LOGIC_UNKNOWN || y == LOGIC_UNKNOWN) {
      return LOGIC_UNKNOWN;
    } else if (INV()._not_(x) == y) {
      return LOGIC_ONE;
    } else if (x == y) {
      return LOGIC_ZERO;
    } else {
      return LOGIC_UNKNOWN;
    }
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _xor_ (x, y);
  }

  virtual const std::string str () const { return "XOR2"; }
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

  virtual const std::string str () const { return "XNOR2"; }
};


} // namespace des

#endif
