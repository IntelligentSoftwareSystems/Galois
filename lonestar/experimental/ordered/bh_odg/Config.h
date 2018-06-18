#ifndef BH_CONFIG_H
#define BH_CONFIG_H

namespace bh {

struct Config {
  const double dtime; // length of one time step
  const double eps;   // potential softening parameter
  const double tol;   // tolerance for stopping recursion, <0.57 to bound error
  const double dthf, epssq, itolsq;
  Config()
      : dtime(0.5), eps(0.05), tol(0.025), dthf(dtime * 0.5), epssq(eps * eps),
        itolsq(1.0 / (tol * tol)) {}
};

} // end namespace bh

#endif //  BH_CONFIG_H
