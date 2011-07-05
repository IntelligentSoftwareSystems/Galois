#ifndef _STANDARD_AVI_H_
#define _STANDARD_AVI_H_

#include "AVI.h"

/** \brief StandardAVI class implementation of AVI base class

 */

class StandardAVI: public AVI {
public:
  typedef double (*BCFunc) (int, int, double);

  enum BCImposedType {
    ZERO, ONE, TWO
  };

    //! StandardAVI constructor designed to handle integration of an individual element
    //! 
    //! @param MyRes  pointer to a Stresswork for this element
    //! @param MassVec pointer to the assembled mass vector for this element
    //! @param L2G local to global map for access to Petsc vectors
    //! @param local_index is the Element's index on the local processor
    //! @param globalIdx is the Element's unique global index
    //! @param IFlag is a pointer to the Boundary values indicator
    //! @param IVAl is a pointer to the actual values of boundary conditions, if IFlag is not false
    //! @param delta is a double that is used as a safety factor in computing the times step for an element
    //! @param time allows the AVI object's time to be set to some value
    //! 

    StandardAVI(
        const LocalToGlobalMap& L2G, 
        const DResidue& MyRes, 
        const VecDouble& MassVec, 
        size_t globalIdx,
        const MatBool& IFlag, 
        const MatDouble& IVal,
        const double delta, 
        const double time) 

      : AVI (),
        operation (MyRes),
        globalIdx (globalIdx),
        imposedFlags (IFlag),
        imposedValues (IVal), 
        delta (delta),
        timeStamp (time) {

        init(L2G, MassVec);


        imposedTypes = std::vector< std::vector<BCImposedType> > (IFlag.size (), std::vector<BCImposedType> ());

        for (size_t f = 0; f < IFlag.size (); ++f) {

          for (size_t a = 0; a < IFlag[f].size (); ++f) {
            if (!IFlag[f][a]) {
              imposedTypes[f].push_back (StandardAVI::ONE);

            } else {
              imposedTypes[f].push_back (StandardAVI::ZERO);
            }
          }
        }
    }

    StandardAVI(
        const LocalToGlobalMap& L2G, 
        const DResidue& MyRes, 
        const VecDouble& MassVec, 
        size_t globalIdx,
        const std::vector<std::vector<BCImposedType> >& IType, 
        const std::vector<BCFunc>& bcfunc_vec,
        const double delta, 
        const double time)

    :   AVI (),
        operation (MyRes),
        globalIdx (globalIdx),
        imposedTypes (IType),
        delta (delta),
        timeStamp (time) {

        init (L2G, MassVec);


        if(imposedFlags.size() != IType.size()) {
            imposedFlags.resize(IType.size());
        }

        if(imposedValues.size() != IType.size()) {
            imposedValues.resize(IType.size());
        }

        if(imposedTypes.size() != IType.size()) {
            imposedTypes.resize(IType.size());
        }

        for(size_t f = 0;f < IType.size();f++){
            if(imposedFlags[f].size() != IType[f].size()) {
                imposedFlags[f].resize(IType[f].size());
            }

            if(imposedValues[f].size() != IType[f].size()) {
                imposedValues[f].resize(IType[f].size());
            }

            if(imposedTypes[f].size() != IType[f].size()) {
                imposedTypes[f].resize(IType[f].size());
            }

            for(size_t a = 0;a < IType[f].size();a++){
                if(IType[f][a] !=  StandardAVI::ZERO){
                    imposedFlags[f][a] = false;
                }else{
                    imposedFlags[f][a] = true;
                }
                imposedValues[f][a] = 0.0;
                imposedTypes[f][a] = IType[f][a];
            }

        }

        for(size_t a = 0;a < bcfunc_vec.size();a++){
            avi_bc_func.push_back(bcfunc_vec[a]);
        }
    }

    
    //! Copy constructor
  StandardAVI (const StandardAVI& that) :
    AVI (that),
    operation (that.operation),
    MMdiag (that.MMdiag),
    globalIdx (that.globalIdx),
    avi_bc_func (that.avi_bc_func),
    imposedTypes (that.imposedTypes),
    imposedFlags (that.imposedFlags),
    imposedValues (that.imposedValues),
    nfields (that.nfields),
    delta (that.delta),
    timeStamp (that.timeStamp) {

      setTimeStep ();
    }

  virtual StandardAVI* clone () const {
    return new StandardAVI (*this);
  }

  //! returns time step for the Element
  double getTimeStep () const {
    return timeStep;
  }

  //! returns the last update time for this element
  double getTimeStamp () const {
    return timeStamp;
  }

  bool setTimeStamp (double timeval) {
    assert (timeval >= 0.0);
    timeStamp = timeval;
    return true;
  }
  ;


  //! Returns the next time at which the force field will be updated
  virtual double getNextTimeStamp () const { return getTimeStamp() + getTimeStep();}
  //! increment the time stamp
  virtual void incTimeStamp () { setTimeStamp(getNextTimeStamp()); }

  virtual const DResidue& getOperation () const { return operation; }

  size_t getFieldDof (size_t fieldnumber) const {
    return operation.getFieldDof (fieldnumber);
  }
  ;

  const std::vector<size_t>& getFields () const {
    return operation.getFields ();
  }
  ;

  //! returns the element geometry
  const ElementGeometry& getGeometry () const {
    return operation.getElement ().getGeometry ();
  }
  ;

  //! returns the element
  const Element& getElement () const {
    return operation.getElement ();
  }
  ;

  //! Updates the force field through the operation Stresswork class
  bool getForceField (const MatDouble& argval, MatDouble& forcefield) const {
    // TODO: change getVal so that a reference can be passed instead of pointer
    operation.getVal (argval, &forcefield);
    return (true);
  }
  ;


  size_t getGlobalIndex (void) const {
    return (globalIdx);
  }
  ;

  //! write the updated time vector into the argument provided
  //! value filled in is the one obtained from getNextTimeStamp ()
  virtual void computeLocalTvec (MatDouble& tnew) const;

  virtual bool vbInit (
    const MatDouble& q,
    const MatDouble& v,
    const MatDouble& vb,
    const MatDouble& ti,
    const MatDouble& tnew,
    MatDouble& qnew,
    MatDouble& vbinit,
    MatDouble& forcefield,
    MatDouble& funcval,
    MatDouble& deltaV
    ) const;


  virtual bool update (const MatDouble& q,
    const MatDouble& v,
    const MatDouble& vb,
    const MatDouble& ti,
    const MatDouble& tnew,
    MatDouble& qnew,
    MatDouble& vnew,
    MatDouble& vbnew,
    MatDouble& forcefield,
    MatDouble& funcval,
    MatDouble& deltaV
    ) const ;

  bool gather (
      const LocalToGlobalMap& L2G,
      const VecDouble& Qval,
      const VecDouble& Vval,
      const VecDouble& Vbval,
      const VecDouble& Tval,
      MatDouble& q,
      MatDouble& v,
      MatDouble& vb,
      MatDouble& ti
  ) const;

  bool assemble (const LocalToGlobalMap& L2G,
      const MatDouble& qnew,
      const MatDouble& vnew,
      const MatDouble& vbnew,
      const MatDouble& tnew,
      VecDouble& Qval,
      VecDouble& Vval,
      VecDouble& Vbval,
      VecDouble& Tval,
      VecDouble& LUpdate) const;

protected:
  //! Sets the time step for the element based upon the element geometry and sound speed.  The safety factor, delta, is set
  //! during the construction of the AVI object.  The optional parameter, epsilon, allows the time step to be adjusted further.
  //! @param epsilon--optional parameter which allows the time step to be set to a fraction of its normal value.
  virtual void setTimeStep (double epsilon = 1.0) {
     timeStep = epsilon * delta * (operation.getElement ().getGeometry ().getInRadius ()) / operation.getMaterial ().getSoundSpeed ();

   };

  virtual void computeDeltaV (const MatDouble& funcval, MatDouble& DeltaV) const;

  virtual bool getImposedValues (const GlobalElementIndex& ElementIndex,
      const LocalToGlobalMap& L2G, size_t field, size_t dof,
      double& qvalue, double& vvalue) const;

  //! Option to round the time step in order to address round-off error.  Not currently used
  //! @param min_ts -- the smallest time step in mesh.  Every time step will be rounded to a value 3 significant digits
  //!     smaller than the min_ts
  void roundTimeStep (double min_ts) {
    char val[80], val2[80];
    sprintf (val, "%0.3e", min_ts / 1000.);
    double cut_off = strtod (val, NULL);
    strncpy (val2, val, 5);
    cut_off = cut_off / strtod (val2, NULL);
    timeStep = floor (timeStep / cut_off) * cut_off;
  }
  ;

  
private:
  // disabling assignment
  StandardAVI& operator = (const StandardAVI& that) {
    return *this;
  }

  void init(const LocalToGlobalMap& L2G, const VecDouble& MassVec) {
         nfields = operation.getFields().size();
         setTimeStep();
         setDiagVals(MassVec, L2G, globalIdx); //set the diagonal values of the Mass Matrix
     }


  //! This function sets the Mass Matrix diagonal values for the element after they have
  //! been computed using DiagonalMassForSW and place into the petsc vector MassVec
  //! @param MassVec -- input in the form of petsc vector computed using DiagonalMassForSW
  //! @param L2G -- localtoglobal map for petsc vectors
  //! @param elem_index -- needed for proper indexing into the L2G map contains the element index locally
  void setDiagVals (const VecDouble& MassVec, const LocalToGlobalMap& L2G,
      const GlobalElementIndex& elem_index);

  //! set the boundary conditions on the local element
  //! @param IFlag is set to true for each bc that is imposed
  //! \todo --Modify this to reflect changes to constructor
  //! @param IVal is set to value for each bc that is imposed
  void setBCs (const MatBool&IFlag,
      const MatDouble& IVal);





  const DResidue&  operation;
  MatDouble MMdiag;

  size_t globalIdx;
  // MatDouble forcefield;

  std::vector<BCFunc> avi_bc_func;
  std::vector<std::vector<BCImposedType> > imposedTypes;
  MatBool imposedFlags;
  MatDouble imposedValues;

  size_t nfields;
  double delta; // safety factor in time step computation
  double timeStamp;
  double timeStep;



};

#endif // _STANDARD_AVI_H_
