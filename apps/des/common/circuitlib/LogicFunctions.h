#ifndef _LOGIC_FUNCTIONS_H_
#define _LOGIC_FUNCTIONS_H_

#include <functional>
#include <string>

#include "logicDefs.h"

struct LogicFunc {
  virtual const std::string toString () const = 0;
};

struct OneInputFunc: public LogicFunc {
  virtual LogicVal operator () (const LogicVal& in) const = 0;
};

struct TwoInputFunc: public LogicFunc {
  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const = 0;
};

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
 * Not.
 *
 * @param in the in
 * @return _not_(in)
 */
struct INV : public OneInputFunc, public std::unary_function<LogicVal, LogicVal> {
  LogicVal _not_ (const LogicVal& in) const {
    if (in == '0') {
      return '1';
    } else if (in == '1') {
      return '0';
    } else {
      return _X;
    }
  }

  virtual LogicVal operator () (const LogicVal& in) const {
    return _not_ (in);
  }

  virtual const std::string toString () const { return "INV"; }
};


/**
 * And.
 *
 * @param x the x
 * @param y the y
 * @return x & y
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
      return _X;
    }

  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _and_ (x, y);
  }

  virtual const std::string toString () const { return "AND2"; }
};

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
 * Or.
 *
 * @param x the x
 * @param y the y
 * @return x | y
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
      return _X;
    }
  }

  virtual LogicVal operator () (const LogicVal& x, const LogicVal& y) const {
    return _or_ (x, y);
  }

  virtual const std::string toString () const { return "OR2"; }
};

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
 * Xor.
 *
 * @param x the x
 * @param y the y
 * @return x ^ y
 */
struct XOR2: public TwoInputFunc, public std::binary_function<LogicVal, LogicVal, LogicVal> {
  LogicVal _xor_ (const LogicVal& x, const LogicVal& y) const {
    if (x == _X || y == _X) {
      return _X;
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
