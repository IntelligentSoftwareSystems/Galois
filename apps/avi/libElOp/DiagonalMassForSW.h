// Sriramajayam

#ifndef DIAGONALMASSMATRIXFORSW
#define DIAGONALMASSMATRIXFORSW

#include "StressWork.h"
#include "AuxDefs.h"

//! \brief Class to compute a diagonal mass matrix for StressWork.
/** This class computes a diagonalized form of the (exact) mass matrix 
 * \f$ M[f][a][b] = \int_E \rho_0 N_a^f N_b^f. \f$
 *
 * Since a diagonal mass matrix is often desired, the entries in each
 * row of the exact mass-matrix are lumped together on the diagonal
 * entry \f$ M[f][a][a] \f$.
 *
 * A mass-vector is assembled (instead of a matrix) with each entry 
 * compued as \f$ M[f][a] = \int_E \rho_0 N_a^f \f$ 
 * where \f$a\f$ runs over all degrees of freedom for
 * field \f$ f \f$.
 *
 * Keeping this in mind, this class inherits the class Residue to
 * assemble the mass vector.  It has pointers to the element for which
 * the mass matrix is to be computed and the material provided over
 * the element. It assumes that there are a minimum of two fields over
 * the element with an optional third.  It implements the member
 * function getVal of Residue to compute the elemental contribution to
 * the mass-vector.
 *
 */

class DiagonalMassForSW: public BaseResidue {
public:

  //! Constructor 
  //! \param IElm Pointer to element over which mass is to be compued.
  //! \param SM SimpleMaterial over the element.
  //! \param field1 Field number of first field.
  //! \param field2 Field number of second field.
  //! \param field3 Field number of third field, assumed to be non-existant by default.
  inline DiagonalMassForSW (const Element& IElm, const SimpleMaterial &SM, const std::vector<size_t>& fieldsUsed)
  : BaseResidue (IElm, SM, fieldsUsed) {
  assert (fieldsUsed.size() > 0 && fieldsUsed.size () <= 3);
}

  //! Destructor
  virtual ~DiagonalMassForSW() {
  }

  //! Copy constructor
  //! \param DMM DiagonalMassForSW object to be copied.
  inline DiagonalMassForSW(const DiagonalMassForSW & DMM) : BaseResidue (DMM) {

  }

  //! Cloning mechanism
  virtual DiagonalMassForSW * clone() const {
    return new DiagonalMassForSW(*this);
  }


  //! Computes the elemental contribution to the mass-vector.
  //! \param argval See Residue. It is a dummy argument since integrations are done over the reference.
  //! \param funcval See Residue.
  bool getVal(const MatDouble &argval, MatDouble* funcval) const;

};

#endif
