/**
 * Material.h
 * DG++
 *
 * Created by Adrian Lew on 10/24/06.
 *  
 * Copyright (c) 2006 Adrian Lew
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including 
 * without limitation the rights to use, copy, modify, merge, publish, 
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included 
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include "galois/substrate/PerThreadStorage.h"

#include "AuxDefs.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <cmath>
#include <cassert>

/**
 \brief Base class for all materials.

 \warning This is a quick fix. Will be revised later.
 */

class Material {
};

/**
 \brief Material whose thermodynamic state depends only on the local strain.

 Simple materials are those for which the stress depends only on the strain.
 Assumes a homogeneous density in the reference configuration.

 Convention:\n
 Any 3x3 second-order tensor \f${\bf A}\f$ is represented by a 2x2 matrix with components  ordered
 in an array \f$A\f$ as \f$A_{iJ}\f$=A[i*3+J].\n
 Any 3x3x3x3 fourth-order tensor \f$\mathbb{C}\f$ is represented by a 3x3x3x3 matrix with components
 ordered in an array \f$C\f$ as \f$\mathbb{C}_{iJkL}\f$=C[i*27+J*9+k*3+l].\n
 */

class SimpleMaterial: public Material {
public:

  // (amber) some constants collected here
  //! Some constants collected together here
  static const double EPS;
  static const double PERT;
  static const double DET_MIN;

  static const size_t MAT_SIZE = 9;
  static const size_t NDF;
  static const size_t NDM;

  static const double I_MAT[MAT_SIZE];

  //! for use with getConstitutiveResponse
  //! optional to compute the tangents or skip
  //! the computation
  enum ConstRespMode {
    COMPUTE_TANGENTS,
    SKIP_TANGENTS,
  };

  //! \param rhoInput Density in reference configuration. If not provided, assumed to be zero.
  inline SimpleMaterial(double rhoInput = 0) :
    RefRho(rhoInput) {
  }

  virtual ~SimpleMaterial() {
  }

  //! Copy constructor
  //! \param SM SimpleMaterial object to be copied.
  inline SimpleMaterial(const SimpleMaterial &SM) :
    RefRho(SM.RefRho) {
  }

  virtual SimpleMaterial * clone() const = 0;

  /**
   \brief
   Returns the constitutive response of the material

   Given the local strain, it  returns the local stress, and if requested,
   the constitutive tangents.\n\n
   More precisely:\n
   The strain is assumed to be a 3x3 second-order tensor \f$F_{iJ}\f$. \n
   The stress is assumed to be a 3x3 second-order tensor \f$P_{iJ}(\bf{F})\f$.  \n
   The constitutive tangents are a 3x3x3x3 fourth-order tensor
   \f[
   A_{iJkL} = \frac{\partial P_{iJ}}{\partial F_{kL}}
   \f]

   @param strain strain tensor, input
   @param stress array where the stress tensor is returned
   @param tangents array where the constitutive tangents are returned. If not provided, not computed.
   @param mode tells whether to compute tangents vector or skip it @see ConstRespMode

   If cannot compute the constitutive relation for some reason, for example a
   negative determinant in the strain, it returns false. If successful, returns true.
   */

  virtual bool getConstitutiveResponse(const VecDouble& strain, VecDouble& stress, VecDouble& tangents
      , const ConstRespMode& mode) const = 0;

  //! Returns the (uniform) density of the reference configuration.
  double getDensityInReference() const {
    return RefRho;
  }

  /** Returns the local density that is a function of only the strain.
   The density is computed as \f[ \rho = \frac{\rho_0}{\text{det}(\nabla F)}, \f]
   where \f$F\f$ is the deformation gradient (as described above). <br>
   Returns true if the computation went well and false otherwise.

   \param strain Deformation gradient.
   \param LocDensity Computed local density.
   */
  bool getLocalMaterialDensity(const VecDouble* strain, double &LocDensity) const {
    assert((*strain).size () == MAT_SIZE);

    // Compute determinant of strain.
    double J = (*strain)[0] * ((*strain)[4] * (*strain)[8] - (*strain)[5] * (*strain)[7]) - (*strain)[1] * ((*strain)[3] * (*strain)[8]
        - (*strain)[5] * (*strain)[6]) + (*strain)[2] * ((*strain)[3] * (*strain)[7] - (*strain)[4] * (*strain)[6]);

    if (fabs(J) < 1.e-10)
      return false;
    else {
      LocDensity = RefRho / J;
      return true;
    }
  }

  //! returns a string with the name of the material
  virtual const std::string getMaterialName() const = 0;

  //! Consistency test\n
  //! Checks that the tangets are in fact the derivatives of the stress with respect to the
  //! strain.
  static bool consistencyTest(const SimpleMaterial &Smat);

  //! @return speed of sound 
  virtual double getSoundSpeed(void) const = 0;
private:
  double RefRho;
};



/**
 \brief NeoHookean constitutive behavior

 */

class NeoHookean: public SimpleMaterial {

  /**
   * Holds temporary vectors used by getConstitutiveResponse
   * Instead of allocating new arrays on the stack, we reuse
   * the same memory in the hope of better cache efficiency
   * There is on instance of this struct per thread
   */
  struct NeoHookenTmpVec {
    static const size_t MAT_SIZE = SimpleMaterial::MAT_SIZE;

    double F[MAT_SIZE];
    double Finv[MAT_SIZE]; 
    double C[MAT_SIZE]; 
    double Cinv[MAT_SIZE]; 
    double S[MAT_SIZE]; 
    double M[MAT_SIZE * MAT_SIZE];

  };

  /**
   * Per thread storage for NeoHookenTmpVec 
   */
  static galois::substrate::PerThreadStorage<NeoHookenTmpVec> perCPUtmpVec;

public:
  NeoHookean(double LambdaInput, double MuInput, double rhoInput = 0) :
    SimpleMaterial(rhoInput), Lambda(LambdaInput), Mu(MuInput) {
  }
  virtual ~NeoHookean() {
  }
  NeoHookean(const NeoHookean &NewMat) :
    SimpleMaterial(NewMat), Lambda(NewMat.Lambda), Mu(NewMat.Mu) {
  }
  virtual NeoHookean * clone() const {
    return new NeoHookean(*this);
  }

  bool getConstitutiveResponse(const VecDouble& strain, VecDouble& stress, VecDouble& tangents
      , const ConstRespMode& mode) const;

  const std::string getMaterialName() const {
    return "NeoHookean";
  }

  double getSoundSpeed(void) const {
    return sqrt((Lambda + 2.0 * Mu) / getDensityInReference());
  }

private:
  // Lame constants
  double Lambda;
  double Mu;
};

/** 
 \brief Linear Elastic constitutive behavior

 This is an abstract class that provides the constitutive response of a linear elastic material.
 However, it does not store the moduli, leaving that task for derived classes. In this way, we have the
 flexibility to handle cases in which the material has different types of anisotropy or under stress.

 */

class LinearElasticBase: public SimpleMaterial {
public:
  LinearElasticBase(double irho = 0) :
    SimpleMaterial(irho) {
  }
  virtual ~LinearElasticBase() {
  }
  LinearElasticBase(const LinearElasticBase &NewMat) :
    SimpleMaterial(NewMat) {
  }
  virtual LinearElasticBase * clone() const = 0;

  bool getConstitutiveResponse(const VecDouble& strain, VecDouble& stress, VecDouble& tangents
      , const ConstRespMode& mode) const;

  const std::string getMaterialName() const {
    return "LinearElasticBase";
  }

protected:
  virtual double getModuli(int i1, int i2, int i3, int i4) const = 0;

};

/**
 \brief Isotropic, unstressed Linear Elastic constitutive behavior

 */

class IsotropicLinearElastic: public LinearElasticBase {
public:
  static const int DELTA_MAT[][3];

  IsotropicLinearElastic(double iLambda, double imu, double irho = 0) :
    LinearElasticBase(irho), lambda(iLambda), mu(imu) {
  }
  virtual ~IsotropicLinearElastic() {
  }
  IsotropicLinearElastic(const IsotropicLinearElastic &NewMat) :
    LinearElasticBase(NewMat), lambda(NewMat.lambda), mu(NewMat.mu) {
  }
  virtual IsotropicLinearElastic * clone() const {
    return new IsotropicLinearElastic(*this);
  }

  const std::string getMaterialName() const {
    return "IsotropicLinearElastic";
  }

  double getSoundSpeed(void) const {
    return sqrt((lambda + 2.0 * mu) / getDensityInReference());
  }

protected:
  double getModuli(int i1, int i2, int i3, int i4) const {
    return lambda * DELTA_MAT[i1][i2] * DELTA_MAT[i3][i4] + mu * (DELTA_MAT[i1][i3] * DELTA_MAT[i2][i4] + DELTA_MAT[i1][i4] * DELTA_MAT[i2][i3]);
  }

private:
  double lambda;
  double mu;
};


#endif
