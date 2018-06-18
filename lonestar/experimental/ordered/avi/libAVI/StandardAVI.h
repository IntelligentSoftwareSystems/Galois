/**
 * StandardAVI.h
 * DG++
 *
 * Created by Adrian Lew on 9/23/08.
 *
 * Copyright (c) 2008 Adrian Lew
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

#ifndef _STANDARD_AVI_H_
#define _STANDARD_AVI_H_

#include "AVI.h"

/** \brief StandardAVI class implementation of AVI base class

 */

class StandardAVI : public AVI {
public:
  //! function used for obtaining value of boundary conditions
  typedef double (*BCFunc)(int, int, double);

  //! type of boundary conditions imposed
  //! ZERO means no boundary conditions
  enum BCImposedType { ZERO, ONE, TWO };

  //! StandardAVI constructor designed to handle integration of an individual
  //! element
  //!
  //! @param L2G local to global map for access to Petsc vectors
  //! @param MyRes  ref to a Stresswork for this element
  //! @param MassVec ref to the assembled mass vector for the mesh
  //! @param globalIdx is the Element's unique global index
  //! @param IFlag is a ref to the Boundary values indicator
  //! @param IVal is a ref to the actual values of boundary conditions, if IFlag
  //! is not false
  //! @param delta is a double that is used as a safety factor in computing the
  //! times step for an element
  //! @param time allows the AVI object's time to be set to some value
  //!

  StandardAVI(const LocalToGlobalMap& L2G, const DResidue& MyRes,
              const VecDouble& MassVec, size_t globalIdx, const MatBool& IFlag,
              const MatDouble& IVal, const double delta, const double time)

      : AVI(time), operation(MyRes), globalIdx(globalIdx), imposedFlags(IFlag),
        imposedValues(IVal), delta(delta) {

    init(L2G, MassVec);

    imposedTypes = std::vector<std::vector<BCImposedType>>(
        IFlag.size(), std::vector<BCImposedType>());

    for (size_t f = 0; f < IFlag.size(); ++f) {

      for (size_t a = 0; a < IFlag[f].size(); ++f) {
        if (!IFlag[f][a]) {
          imposedTypes[f].push_back(StandardAVI::ONE);

        } else {
          imposedTypes[f].push_back(StandardAVI::ZERO);
        }
      }
    }
  }

  //! StandardAVI constructor designed to handle integration of an individual
  //! element
  //!
  //! @param MyRes  ref to a Stresswork for this element
  //! @param MassVec ref to the assembled mass vector for the mesh
  //! @param L2G local to global map for access to Petsc vectors
  //! @param globalIdx is the Element's unique global index
  //! @param IType is a vector containing types of imposed boundary conditons
  //! @see StandardAVI::BCImposedType
  //! @param bcfunc_vec is a vector of function pointers used to obtain the
  //! value of boundary conditions
  //! @param delta is a double that is used as a safety factor in computing the
  //! times step for an element
  //! @param time allows the AVI object's time to be set to some value
  //!
  StandardAVI(const LocalToGlobalMap& L2G, const DResidue& MyRes,
              const VecDouble& MassVec, size_t globalIdx,
              const std::vector<std::vector<BCImposedType>>& IType,
              const std::vector<BCFunc>& bcfunc_vec, const double delta,
              const double time)

      : AVI(time), operation(MyRes), globalIdx(globalIdx), imposedTypes(IType),
        delta(delta) {

    init(L2G, MassVec);

    if (imposedFlags.size() != IType.size()) {
      imposedFlags.resize(IType.size());
    }

    if (imposedValues.size() != IType.size()) {
      imposedValues.resize(IType.size());
    }

    if (imposedTypes.size() != IType.size()) {
      imposedTypes.resize(IType.size());
    }

    for (size_t f = 0; f < IType.size(); f++) {
      if (imposedFlags[f].size() != IType[f].size()) {
        imposedFlags[f].resize(IType[f].size());
      }

      if (imposedValues[f].size() != IType[f].size()) {
        imposedValues[f].resize(IType[f].size());
      }

      if (imposedTypes[f].size() != IType[f].size()) {
        imposedTypes[f].resize(IType[f].size());
      }

      for (size_t a = 0; a < IType[f].size(); a++) {
        if (IType[f][a] != StandardAVI::ZERO) {
          imposedFlags[f][a] = false;
        } else {
          imposedFlags[f][a] = true;
        }
        imposedValues[f][a] = 0.0;
        imposedTypes[f][a]  = IType[f][a];
      }
    }

    for (size_t a = 0; a < bcfunc_vec.size(); a++) {
      avi_bc_func.push_back(bcfunc_vec[a]);
    }
  }

  //! Copy constructor
  StandardAVI(const StandardAVI& that)
      : AVI(that), operation(that.operation), MMdiag(that.MMdiag),
        globalIdx(that.globalIdx), avi_bc_func(that.avi_bc_func),
        imposedTypes(that.imposedTypes), imposedFlags(that.imposedFlags),
        imposedValues(that.imposedValues), nfields(that.nfields),
        delta(that.delta) {

    setTimeStep();
  }

  virtual StandardAVI* clone() const { return new StandardAVI(*this); }

  virtual const DResidue& getOperation() const { return operation; }

  size_t getFieldDof(size_t fieldnumber) const {
    return operation.getFieldDof(fieldnumber);
  }

  const VecSize_t& getFields() const { return operation.getFields(); }

  //! returns the element geometry
  const ElementGeometry& getGeometry() const {
    return operation.getElement().getGeometry();
  }

  //! returns the element
  const Element& getElement() const { return operation.getElement(); }

  //! Updates the force field through the operation Stresswork class
  bool getForceField(const MatDouble& argval, MatDouble& forcefield) const {
    operation.getVal(argval, forcefield);
    return (true);
  }

  size_t getGlobalIndex(void) const { return (globalIdx); }

  //! write the updated time vector into the argument provided
  //! value filled in is the one obtained from getNextTimeStamp ()
  virtual void computeLocalTvec(MatDouble& tnew) const;

  virtual bool vbInit(const MatDouble& q, const MatDouble& v,
                      const MatDouble& vb, const MatDouble& ti,
                      const MatDouble& tnew, MatDouble& qnew, MatDouble& vbinit,
                      MatDouble& forcefield, MatDouble& funcval,
                      MatDouble& deltaV) const;

  virtual bool update(const MatDouble& q, const MatDouble& v,
                      const MatDouble& vb, const MatDouble& ti,
                      const MatDouble& tnew, MatDouble& qnew, MatDouble& vnew,
                      MatDouble& vbnew, MatDouble& forcefield,
                      MatDouble& funcval, MatDouble& deltaV) const;

  bool gather(const LocalToGlobalMap& L2G, const VecDouble& Qval,
              const VecDouble& Vval, const VecDouble& Vbval,
              const VecDouble& Tval, MatDouble& q, MatDouble& v, MatDouble& vb,
              MatDouble& ti) const;

  bool assemble(const LocalToGlobalMap& L2G, const MatDouble& qnew,
                const MatDouble& vnew, const MatDouble& vbnew,
                const MatDouble& tnew, VecDouble& Qval, VecDouble& Vval,
                VecDouble& Vbval, VecDouble& Tval, VecDouble& LUpdate) const;

protected:
  //! Sets the time step for the element based upon the element geometry and
  //! sound speed.  The safety factor, delta, is set during the construction of
  //! the AVI object.  The optional parameter, epsilon, allows the time step to
  //! be adjusted further.
  //! @param epsilon:  optional parameter which allows the time step to be set
  //! to a fraction of its normal value.
  virtual void setTimeStep(double epsilon = 1.0) {
    timeStep = epsilon * delta *
               (operation.getElement().getGeometry().getInRadius()) /
               operation.getMaterial().getSoundSpeed();
  }

  virtual void computeDeltaV(const MatDouble& funcval, MatDouble& DeltaV) const;

  virtual bool getImposedValues(const GlobalElementIndex& ElementIndex,
                                const LocalToGlobalMap& L2G, size_t field,
                                size_t dof, double& qvalue,
                                double& vvalue) const;

  //! Option to round the time step in order to address round-off error.  Not
  //! currently used
  //! @param min_ts -- the smallest time step in mesh.  Every time step will be
  //! rounded to a value 3 significant digits
  //!     smaller than the min_ts
  void roundTimeStep(double min_ts) {
    char val[80], val2[80];
    sprintf(val, "%0.3e", min_ts / 1000.);
    double cut_off = strtod(val, NULL);
    strncpy(val2, val, 5);
    cut_off  = cut_off / strtod(val2, NULL);
    timeStep = floor(timeStep / cut_off) * cut_off;
  }

private:
  // disabling assignment
  StandardAVI& operator=(const StandardAVI& that) { return *this; }

  void init(const LocalToGlobalMap& L2G, const VecDouble& MassVec) {
    nfields = operation.getFields().size();
    setTimeStep();
    setDiagVals(MassVec, L2G,
                globalIdx); // set the diagonal values of the Mass Matrix
  }

  //! This function sets the Mass Matrix diagonal values for the element after
  //! they have been computed using DiagonalMassForSW and place into the petsc
  //! vector MassVec
  //! @param MassVec -- input in the form of petsc vector computed using
  //! DiagonalMassForSW
  //! @param L2G -- localtoglobal map for petsc vectors
  //! @param elem_index -- needed for proper indexing into the L2G map contains
  //! the element index locally
  void setDiagVals(const VecDouble& MassVec, const LocalToGlobalMap& L2G,
                   const GlobalElementIndex& elem_index);

  //! set the boundary conditions on the local element
  //! @param IFlag is set to true for each bc that is imposed
  //! \todo --Modify this to reflect changes to constructor
  //! @param IVal is set to value for each bc that is imposed
  void setBCs(const MatBool& IFlag, const MatDouble& IVal);

  const DResidue& operation;
  MatDouble MMdiag;

  size_t globalIdx;
  // MatDouble forcefield;

  std::vector<BCFunc> avi_bc_func;
  std::vector<std::vector<BCImposedType>> imposedTypes;
  MatBool imposedFlags;
  MatDouble imposedValues;

  size_t nfields;
  double delta; // safety factor in time step computation
};

#endif // _STANDARD_AVI_H_
