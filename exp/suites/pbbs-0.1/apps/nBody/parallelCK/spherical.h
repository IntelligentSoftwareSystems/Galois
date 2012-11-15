// The following code was addapted from the PetFMM code
// to be used as part of the Problem Based Benchmark Suite (PBBS)
// Both codes live under he Gnu general public license
//
// Copyright (c) 2010 PetFMM developer team.
// Copyright (c) 2011 Guy Blelloch and the PBBS team
// 
// Permission to copy and modify this software and its documentation is hereby
// granted, provided that this notice is retained thereon and on all copies or
// modifications. Permission is hereby granted to use, reproduce, prepare
// derivative works, and to redistribute to others, so long as this original
// copyright notice is retained.
// 
// DISCLAIMER
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef included_Transform_SphericalHarmonic_hh
#define included_Transform_SphericalHarmonic_hh

#include "geometry.h"
#include <complex>

  template<int terms, int coefficients = terms*terms>
  struct Transform {
  public:
    typedef point3d point_type;
    typedef vect3d vect_type;
    typedef std::complex<double>  coeff_type;
    typedef double     real_type;
    const static int numTerms        = terms;
    const static int numCoefficients = coefficients;
    const static int precompSize     = 2*numTerms*2*numTerms;
    const real_type            eps;
    const std::complex<double> I;
  protected:
    real_type   prefactor[precompSize];  // \sqrt{\frac{(n - |m|)!}{(n + |m|)!}}
    real_type   Anm[precompSize];        // A^n_m = \frac{-1^n}{(n - m)! (n + m)!}
    real_type   AnmI[precompSize];       // 1/A^n_m 
  public:

  Transform() : eps(1e-20), I(0.0, 1.0) {};

  ~Transform() {};

  protected:
    real_type factorial(real_type n) const {
      //assert(n >= 0);
      if (n <= 1) {
        return(1);
      } else {
        n *= factorial(n-1);
        return(n);
      }
    };

  public:
    void precompute() {
      //   Calculate fac^n_m = \sqrt{\frac{(n - |m|)!}{(n + |m|)!}}
      //   Calculate A^n_m = \frac{-1^n}{(n - m)! (n + m)!}
      for(int n = 0, nm = 0; n < 2*numTerms; ++n) {
        for(int m = -n; m <= n; ++m, ++nm){
          //nm = (n*n) + (n+m);
          prefactor[nm] = sqrt(factorial(n - abs(m)) / factorial(n + abs(m)));
          Anm[nm]       = pow(-1.0, n)/sqrt(factorial(n - m) * factorial(n + m));
          AnmI[nm]      = 1/Anm[nm];
        }
      }
    };

    void evaluateMultipole(coeff_type array[], real_type r,real_type cosTheta,coeff_type eiphi) {
      const real_type s2 = sqrt((1-cosTheta)*(1+cosTheta));
      real_type       pn = 1.0;
      real_type powers[numTerms];
      powers[0] = 1.0;
      for(int i=1; i<numTerms; i++) powers[i] = powers[i-1]*r;
      coeff_type eim = coeff_type(1.0,0.0);

      for(int m = 0, fact = 1; m < numTerms; ++m, fact += 2) {
        real_type  p   = pn;
        const int  npn = m*m + 2*m;
        const int  nmn = m*m;
        coeff_type Ynm = prefactor[npn]*p*eim;
        real_type  p1  = p;

        array[npn] = powers[m]*Ynm;
        array[nmn] = powers[m]*std::conj(Ynm);
        p = cosTheta*(2*m+1)*p;
        for(int n = m+1; n < numTerms; ++n) {
          const int       npm = n*n+n+m;
          const int       nmm = n*n+n-m;
          const real_type p2  = p1;

          Ynm = prefactor[npm]*p*eim;
          array[npm] = powers[n]*Ynm;
          array[nmm] = powers[n]*std::conj(Ynm);
          p1 = p;
          p  = (cosTheta*(2*n+1)*p1 - (n+m)*p2)/(n-m+1);
        }
        pn = -pn*fact*s2;
        eim = eim * eiphi;
      }
    };

    void evaluateLocal(coeff_type array[], real_type r, real_type cosTheta, coeff_type eiphi) {
      const real_type s2 = sqrt((1-cosTheta)*(1+cosTheta));
      real_type       pn = 1.0;
      real_type powers[2*numTerms+1];
      powers[0] = 1.0;
      real_type ri = 1.0/r;
      for(int i=1; i<2*numTerms+1; i++) powers[i] = powers[i-1]*ri;
      coeff_type eim = coeff_type(1.0,0.0);

      for(int m = 0, fact = 1; m < 2*numTerms; ++m, fact += 2) {
        real_type  p   = pn;
        const int  npn = m*m + 2*m;
        const int  nmn = m*m;
        coeff_type Ynm = prefactor[npn]*p*eim;
        real_type  p1  = p;

	array[npn] = Ynm*powers[m+1];
        array[nmn] = std::conj(array[npn]);
        p = cosTheta*(2*m+1)*p;
        for(int n = m+1; n < 2*numTerms; ++n) {
          const int       npm = n*n+n+m;
          const int       nmm = n*n+n-m;
          const real_type p2  = p1;

          Ynm = prefactor[npm]*p*eim;
          array[npm] = Ynm*powers[n+1];
          array[nmm] = std::conj(array[npm]); 
          p1 = p;
          p  = (cosTheta*(2*n+1)*p1 - (n+m)*p2)/(n-m+1);
        }
        pn = -pn*fact*s2;
        eim = eim * eiphi;
      }
    };

    void evaluateMultipoleTheta(coeff_type multipole[], coeff_type multipoleTheta[], 
				real_type r, real_type cosTheta, real_type sinTheta, 
				coeff_type eiPhi) {
      const real_type s2 = sqrt((1-cosTheta)*(1+cosTheta));
      real_type       pn = 1.0;
      coeff_type dummy;
      coeff_type eim = coeff_type(1.0,0.0);

      for(int m = 0, fact = 1; m < numTerms; ++m, fact += 2) {
        real_type  p   = pn;
        const int  npn = m*m + 2*m;
        const int  nmn = m*m;
        coeff_type Ynm = prefactor[npn]*p*eim;
        real_type  p1  = p;

        multipole[npn] = Ynm;
        multipole[nmn] = std::conj(Ynm);
        p = cosTheta*(2*m+1)*p;

        coeff_type Yth = prefactor[npn]*(p-(m+1)*cosTheta*p1)/sinTheta*eim;

        multipoleTheta[npn] = Yth;
        multipoleTheta[nmn] = std::conj(Yth);
        for(int n = m+1; n < numTerms; ++n) {
          const int       npm = n*n+n+m;
          const int       nmm = n*n+n-m;
          const real_type p2  = p1;

          Ynm = prefactor[npm]*p*eim;
          multipole[npm] = Ynm;
          multipole[nmm] = std::conj(Ynm);
          p1 = p;
          p  = (cosTheta*(2*n+1)*p1 - (n+m)*p2)/(n-m+1);
          Yth = prefactor[npm]*((n-m+1)*p - (n+1)*cosTheta*p1)/sinTheta*eim;
          multipoleTheta[npm] = Yth;
          multipoleTheta[nmm] = std::conj(Yth);
        }
        pn = -pn*fact*s2;
        eim = eim * eiPhi;
      }
    };
  public:

  inline int powNeg1(int i) {
    return (i & 1) ? -1 : 1;
  }

    void M2Madd(coeff_type array[],  point_type newCenter,coeff_type coeff[],point_type center) {
      vect_type diff = newCenter - center;
      real_type r = diff.Length();
      real_type cosTheta = diff.z/r;
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      coeff_type multipole[numCoefficients];
      evaluateMultipole(multipole, r, cosTheta, std::conj(eiphi));

      for(int j = 0; j < numTerms; ++j) {
        for(int k = 0; k <= j; ++k) {
          const int  jk  = j*j + j+k;
          const int  jks = j*(j+1)/2+k;
          coeff_type bx  = 0.0;

          for(int n = 0; n <= j; ++n) {
            for(int m = -n; m <= fmin(k-1,n); ++m) {
              if (j-n >= k-m) {
                const int        jnkm  = (j-n)*(j-n) + j-n+k-m;
                const int        jnkms = (j-n)*(j-n+1)/2 + k-m;
                const int        nm    = n*n + n+m;
                const coeff_type cnm = (powNeg1((m-abs(m))/2)*powNeg1(n)*
					Anm[nm]*Anm[jnkm]*AnmI[jk]*multipole[nm]);
                bx += coeff[jnkms]*cnm;
              }
            }
            for(int m = k; m <= n; ++m) {
              if (j-n >= m-k) {
                const int        jnkm  = (j-n)*(j-n) + j-n+k-m;
                const int        jnkms = (j-n)*(j-n+1)/2 - k+m;
                const int        nm    = n*n + n+m;
                const coeff_type cnm   = powNeg1(k+n+m)*Anm[nm]*Anm[jnkm]*AnmI[jk]*multipole[nm];
                bx += std::conj(coeff[jnkms])*cnm;
              }
            }
          }
          array[jks] += bx;
        }
      }
    };

    void M2Ladd(coeff_type array[], point_type newCenter, coeff_type coeff[],point_type center) {
      vect_type diff = newCenter - center;
      real_type r = diff.Length();
      real_type cosTheta = diff.z/r;
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      // to save a little time with little loss
      int numSourceTerms = numTerms -1 ; 
      coeff_type co[coefficients];
      coeff_type local[precompSize];

      // copy into full form [-m,m]
      for (int n = 0; n < numSourceTerms; ++n) {
	int nns = n*(n+1);
	for(int m = -n; m < 0; ++m) 
	  co[nns + m] = std::conj(coeff[nns/2 - m]);
	for(int m = 0; m <= n; ++m) 
	  co[nns + m] = coeff[nns/2 + m];
      }
      
      evaluateLocal(local, r, cosTheta, eiphi);
      for(int j = 0; j < numTerms; ++j) {
        for(int k = 0; k <= j; k++) {
          const int  jk  = j*j + j+k;
          const int  jks = j*(j+1)/2 + k;
          coeff_type ax  = 0.0;
	  real_type zz[2*numTerms-1];
	  for (int m = -(numSourceTerms+1); m < numSourceTerms; m++) {
	    int ip = (1-2*((-abs(k-m) + abs(k) + abs(m))/2 & 1));  // i^{|k-m|-|k|-|m|}
	    // i^{|k-m|-|k|-|m|} * A^k_j * -1^{j}
	    // should it be? : i^{|k-m|-|k|-|m|} * A^k_j * -1^{j+k}
	    zz[m+numSourceTerms-1] = ip * pow(-1.0, j) * Anm[jk];
	  }

          for(int n = 0; n < numSourceTerms; ++n) {
	    int nns = n*(n+1);
	    int jn = j+n;
	    int jns = jn*(jn+1)-k;
            for(int m = -n; m <= n; ++m) {
              const int nm   = nns + m;
              const int jnkm = jns + m;
	      real_type srr = zz[m+numSourceTerms-1]*(Anm[nm]*AnmI[jnkm]);
	      ax += co[nm]*srr*local[jnkm];
            }
          }
          array[jks] += ax;
        }
      }
    };

    void L2Ladd(coeff_type array[],  point_type newCenter,  coeff_type coeff[],  point_type center) {
      vect_type diff = newCenter - center;
      real_type r = diff.Length();
      real_type cosTheta = diff.z/r;
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      coeff_type multipole[numCoefficients];
      evaluateMultipole(multipole, r, cosTheta, eiphi);
      for(int j = 0; j < numTerms; ++j) {
        for(int k = 0; k <= j; ++k) {
          const int  jk  = j*j + j+k;
          const int  jks = j*(j+1)/2 + k;
          coeff_type ax  = 0.0;

          for(int n = j; n < numTerms; ++n) {
            for(int m = j+k-n; m < 0; ++m) {
              const int        jnkm = (n-j)*(n-j) + n-j+m-k;
              const int        nm   = n*n + n-m;
              const int        nms  = n*(n+1)/2 - m;
              const coeff_type cnm  = powNeg1(k)*Anm[jnkm]*Anm[jk]*AnmI[nm]*multipole[jnkm];
              ax += std::conj(coeff[nms])*cnm;
            }
            for(int m = 0; m <= n; ++m) {
              if (n-j >= abs(m-k)) {
                const int        jnkm = (n-j)*(n-j) + n-j+m-k;
                const int        nm   = n*n + n+m;
                const int        nms  = n*(n+1)/2 + m;
                const coeff_type cnm  = powNeg1((m-k - abs(m-k))/2)*Anm[jnkm]*Anm[jk]*AnmI[nm]*multipole[jnkm];
                ax += coeff[nms]*cnm;
              }
            }
          }
          array[jks] += ax;
        }
      }
    };

    void clearM(coeff_type array[]) {
      for (int i=0; i < numTerms*(numTerms+1)/2; i++) array[i] = coeff_type(0.0,0.0);
    }

    void P2Madd(coeff_type array[], real_type gamma, point_type center, point_type x) {
      vect_type diff = x - center;
      real_type r = diff.Length();
      real_type cosTheta = (r==0) ? 1.0 : diff.z/r;
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      coeff_type multipole[numCoefficients];
      evaluateMultipole(multipole, r, cosTheta, std::conj(eiphi));

      for(int n = 0, nms = 0; n < numTerms; ++n) {
        for(int m = 0; m <= n; ++m, ++nms) {
          int nm = n*n + n+m;
          array[nms] += gamma*multipole[nm];
        }
      }
    };

    void M2P(real_type& potential, vect_type& field, point_type x, coeff_type coeff[], 
	     point_type center) {
      vect_type diff  = x - center;
      real_type r = diff.Length();
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      real_type cosTheta = (r==0) ? 1.0 : diff.z/r;
      real_type sinTheta = (r==0) ? 0.0 : rxy/r;
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      real_type        gx = 0.0, gxr = 0.0, gxth = 0.0, gxph = 0.0;
      real_type powers[numTerms+1];
      powers[0] = 1.0;
      real_type ri = 1.0/r;
      for(int i=1; i<numTerms+1; i++) powers[i] = powers[i-1]*ri;
      coeff_type multipole[numCoefficients];
      coeff_type multipoleTheta[numCoefficients];

      evaluateMultipoleTheta(multipole, multipoleTheta, r, cosTheta, sinTheta, eiphi);
      for(int n = 0; n < numTerms; ++n) {
        const int nm  = n*n+n;
        const int nms = n*(n+1)/2;

	real_type xx = .5*powers[n+1]*(multipole[nm]*coeff[nms]).real();
	gx  += xx;
        gxr  += -(n+1)*ri*xx;
        gxth += .5*(powers[n+1]*multipoleTheta[nm]*coeff[nms]).real();
        for(int m = 1; m <= n; ++m) {
          const int nm  = n*n + n+m;
          const int nms = n*(n+1)/2 + m;
	  coeff_type xx = (powers[n+1]*multipole[nm]*coeff[nms]);
          gx  += xx.real();
          gxr  += -(n+1)*ri*xx.real();
          gxth += powers[n+1]*(multipoleTheta[nm]*coeff[nms]).real();
          gxph += -m*xx.imag();
        }
      }
      real_type cosPhi = eiphi.real();
      real_type sinPhi = eiphi.imag();
      gx *= 2; gxr *= 2; gxth *= 2; gxph *= 2;
      const real_type gxx = sinTheta*cosPhi*gxr +cosTheta*cosPhi/r*gxth - sinPhi/r/sinTheta*gxph;
      const real_type gxy = sinTheta*sinPhi*gxr +cosTheta*sinPhi/r*gxth + cosPhi/r/sinTheta*gxph;
      const real_type gxz = cosTheta*gxr - sinTheta/r*gxth;
      potential = gx; //1.0/(4*M_PI) * gx;
      field.x  = gxx; //1.0/(4*M_PI) * gxx;
      field.y  = gxy; //1.0/(4*M_PI) * gxy;
      field.z  = gxz; //1.0/(4*M_PI) * gxz;
    };

    // gamma currently not used
    void L2P(real_type& potential, vect3d& field, point_type x, coeff_type coeff[], 
	     point_type center) {
      vect_type diff  = x - center;
      real_type r = diff.Length();
      real_type rxy = sqrt(diff.x*diff.x + diff.y*diff.y);
      real_type cosTheta = (r==0) ? 1.0 : diff.z/r;
      real_type sinTheta = (r==0) ? 0.0 : rxy/r;
      coeff_type eiphi = (rxy==0) ? coeff_type(1,0) : coeff_type(diff.x/rxy, diff.y/rxy);

      real_type gx = 0.0, gxr = 0.0, gxth = 0.0, gxph = 0.0;
      coeff_type multipole[numCoefficients];
      coeff_type multipoleTheta[numCoefficients];

      evaluateMultipoleTheta(multipole, multipoleTheta, r, cosTheta, sinTheta, eiphi);
      for(int n = 0; n < numTerms; ++n) {
        const int nm  = n*n+n;
        const int nms = n*(n+1)/2;

        gx  += (pow(r,n)*multipole[nm]*coeff[nms]).real();
        gxr  += (n*pow(r,n-1)*multipole[nm]*coeff[nms]).real();
        gxth += (pow(r,n)*multipoleTheta[nm]*coeff[nms]).real();
        for(int m = 1; m <= n; ++m) {
          const int nm  = n*n + n+m;
          const int nms = n*(n+1)/2 + m;

          gx  += 2*(pow(r,n)*multipole[nm]*coeff[nms]).real();
          gxr  += 2*(n*pow(r,n-1)*multipole[nm]*coeff[nms]).real();
          gxth += 2*(pow(r,n)*multipoleTheta[nm]*coeff[nms]).real();
          gxph += 2*(I*(m*pow(r,n)*multipole[nm]*coeff[nms])).real();
        }
      }
      real_type cosPhi = eiphi.real();
      real_type sinPhi = eiphi.imag();
      const real_type gxx = sinTheta*cosPhi*gxr+ cosTheta*cosPhi/r*gxth - sinPhi/r/sinTheta*gxph;
      const real_type gxy = sinTheta*sinPhi*gxr+ cosTheta*sinPhi/r*gxth + cosPhi/r/sinTheta*gxph;
      const real_type gxz = cosTheta*gxr - sinTheta/r*gxth;
      potential += gx; //1.0/(4*M_PI) * gx;
      field.x  += gxx; //1.0/(4*M_PI) * gxx;
      field.y  += gxy; //1.0/(4*M_PI) * gxy;
      field.z  += gxz; //1.0/(4*M_PI) * gxz;
    };

  };
#endif // included_Transform_SphericalHarmonic_hh
