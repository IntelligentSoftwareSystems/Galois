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
