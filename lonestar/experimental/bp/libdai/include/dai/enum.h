/*
 * This file belongs to the Galois project, a C++ library for exploiting
 * parallelism. The code is being released under the terms of the 3-Clause BSD
 * License (a copy is located in LICENSE.txt at the top-level directory).
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

/*  This file is part of libDAI - http://www.libdai.org/
 *
 *  Copyright (c) 2006-2011, The libDAI authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license that can be found
 * in the LICENSE file.
 */

/// \file
/// \brief Defines the DAI_ENUM macro, which can be used to define an \c enum
/// with additional functionality.

#ifndef __defined_libdai_enum_h
#define __defined_libdai_enum_h

#include <cstring>
#include <iostream>
#include <dai/exceptions.h>

/// Extends the C++ \c enum type by supporting input/output streaming and
/// conversion to and from <tt>const char*</tt> and \c size_t
/** For more details see the source code.
 *
 *  \par Example:
 *  \code
 *  DAI_ENUM(colors,RED,GREEN,BLUE)
 *  \endcode
 *  defines a class \a colors encapsulating an
 *  \code
 *  enum {RED, GREEN, BLUE};
 *  \endcode
 *  which offers additional functionality over the plain \c enum keyword.
 */
#define DAI_ENUM(x, val0, ...)                                                 \
  class x {                                                                    \
  public:                                                                      \
    enum value { val0, __VA_ARGS__ };                                          \
                                                                               \
    x() : v(val0) {}                                                           \
                                                                               \
    x(value w) : v(w) {}                                                       \
                                                                               \
    x(char const* w) {                                                         \
      static char const* labelstring = #val0 "," #__VA_ARGS__;                 \
      size_t pos_begin               = 0;                                      \
      size_t i                       = 0;                                      \
      for (size_t pos_end = 0;; pos_end++) {                                   \
        if ((labelstring[pos_end] == ',') || (labelstring[pos_end] == '\0')) { \
          if ((strlen(w) == pos_end - pos_begin) &&                            \
              (strncmp(labelstring + pos_begin, w, pos_end - pos_begin) ==     \
               0)) {                                                           \
            v = (value)i;                                                      \
            return;                                                            \
          } else {                                                             \
            i++;                                                               \
            pos_begin = pos_end + 1;                                           \
          }                                                                    \
        }                                                                      \
        if (labelstring[pos_end] == '\0')                                      \
          break;                                                               \
      }                                                                        \
      DAI_THROWE(UNKNOWN_ENUM_VALUE, "'" + std::string(w) + "' is not in [" +  \
                                         std::string(labelstring) + "]");      \
    }                                                                          \
                                                                               \
    operator value() const { return v; }                                       \
                                                                               \
    operator size_t() const { return (size_t)v; }                              \
                                                                               \
    operator char const *() const {                                            \
      static char labelstring[] = #val0 "," #__VA_ARGS__;                      \
      size_t pos_begin          = 0;                                           \
      size_t i                  = 0;                                           \
      for (size_t pos_end = 0;; pos_end++)                                     \
        if ((labelstring[pos_end] == ',') || (labelstring[pos_end] == '\0')) { \
          if ((size_t)v == i) {                                                \
            labelstring[pos_end] = '\0';                                       \
            return labelstring + pos_begin;                                    \
          } else {                                                             \
            i++;                                                               \
            pos_begin = pos_end + 1;                                           \
          }                                                                    \
        }                                                                      \
    }                                                                          \
                                                                               \
    friend std::istream& operator>>(std::istream& is, x& y) {                  \
      std::string s;                                                           \
      is >> s;                                                                 \
      y = x(s.c_str());                                                        \
      return is;                                                               \
    }                                                                          \
                                                                               \
    friend std::ostream& operator<<(std::ostream& os, const x& y) {            \
      os << (const char*)y;                                                    \
      return os;                                                               \
    }                                                                          \
                                                                               \
  protected:                                                                   \
    value v;                                                                   \
  }

#endif
