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

#ifndef _TRON_H
#define _TRON_H

class function {
public:
  virtual double fun(double* w)           = 0;
  virtual void grad(double* w, double* g) = 0;
  virtual void Hv(double* s, double* Hs)  = 0;

  virtual int get_nr_variable(void) = 0;
  virtual ~function(void) {}
};

class TRON {
public:
  TRON(const function* fun_obj, double eps = 0.1, int max_iter = 100,
       int max_cg_iter = 20);
  ~TRON();

  void tron(double* w, bool set_w_to_zero = true);
  void set_print_string(void (*i_print)(const char* buf));

private:
  int trcg(double delta, double* g, double* s, double* r);
  double norm_inf(int n, double* x);

  double eps;
  int max_iter;
  int max_cg_iter;
  function* fun_obj;
  void info(const char* fmt, ...);
  void (*tron_print_string)(const char* buf);
};
#endif
