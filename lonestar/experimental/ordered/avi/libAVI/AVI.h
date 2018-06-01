/**
 * AVI.h
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

#ifndef AVI_CLASS
#define AVI_CLASS

#include <vector>
#include <list>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>

#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>


#include "AuxDefs.h"
#include "ElementalOperation.h"
#include "DiagonalMassForSW.h"
#include "StressWork.h"

#include "util.h"

/**
 \brief AVI: abstract base class for AVI

 An AVI object consists of:

 1) A force field  \f$f_a(q_1,\ldots,q_N,t)\f$, where \f$q_a\f$ are degrees of freedom,
 and \f$f_a\f$ is the force acting on degree of freedom "a".\n
 2) A mass matrix for the degrees of freedom of the potential. \n
 3) A time step \n
 4) The last time at which the force field was updated\n
 5) The last time each degree of freedom was updated\n

 The AVI class assumes that there exists four provided global
 arrays, accessed through a LocalToGlobalMap:

 1) An array Qval with the value of each degree of freedom at its last update, \f$q_a^{i-1}\f$\n
 2) An array Vval with the latest value of the time derivative of the
 degrees of freedom at its last update, \f$v_a^{i-1}\f$.\n
 3) An array Vbval with the latest value of the time derivative of the
 degree of freedom at the middle of a time step, \f$v_a^{i-1/2}\f$.\n
 4) An array Tval with the last time at which a degree of freedom was updated, \f$t_a^{i-1}\f$.

 Given \f$q_a^{i-1}\f$, \f$v_a^{i-1/2}\f$ and \f$t_a^{i-1}\f$, the class
 computes \f$q_a^i\f$, \f$v_a^i\f$ and \f$v_a^{i+1/2}\f$ for each
 degree of freedom of the object's force field. It also updates the
 values of \f$t_a^i\f$ and the objects latest time.

 The update is as follows:

 If \f$q_a\f$ does not have imposed values \f$q_a(t)\f$ (e.g., Dirichlet bounday conditions), then

 1) Compute \f$q_a^i = q_a^{i-1} + (t-t_a^i) v_a^{i-1/2}\f$ for each \f$a\f$, where \f$t\f$ is the current time of the object.\n
 2) Solve \f$ M_{ab} \Delta v_b = - \Delta t f_a(q_a^i)\f$ for \f$\Delta v_b\f$, where \f$f_a(t)\f$ is the force on node \f$a\f$ by the force field in the object computed with \f$q_a^i\f$.\n
 3) Compute \f$ v_a^i = v_a^{i-1/2} + \Delta v_b/2\f$\n
 4) Compute \f$ v_a^{i+1/2} = v_a^{i-1/2} + \Delta v_b\f$

 Else, set \f$q_a^i=q_a(t_a^i)\f$, \f$v_a^i=\dot{q}_a(t_a^i)\f$ and
 \f$v_a^{i+1/2}\f$ is defined to any value, it is not guaranteed to
 have any meaning.

 The class also has an initialization procedure to compute the
 initial values of \f$v_a^{1/2}\f$.

 Following the convention elsewhere in the code, degrees of freedom
 are labeled with field number f and degree of freedom number for
 that field, b. Even thought it is not the most natural numbering in
 the context of ODEs, it simplifies the communication with the rest
 of the code. Field and degree of freedom numbers begin from zero.

 The class separates the update process into three different functions: gather, update, assemble.
 In this way more flexibility is provided to be able to work with force fields that depend, perhaps,
 on sets of internal variables that are not necessarily updated with AVI.

 Imposed values need to be provided separately. A virtual procedure is constructed to this end.

 \todo The update routine does not take further arguments that may not need to be updated but
 may be needed to compute the force field, such as when dealing with plasticity or a coupled heat
 conduction problem. We will need to deal with this eventually.

 */

class AVI {
protected:
  double timeStamp;
  double timeStep;
  
public:
  AVI (double timeStamp): timeStamp(timeStamp) { }
  virtual ~AVI () { }
  virtual AVI * clone () const = 0;
  //  virtual int getElemIndex(void) const = 0;

  //! Returns the current value of the time step for the potential
  double getTimeStep () const {
    return timeStep;
  }

  //! Returns the last time at which the force field was updated
  double getTimeStamp () const {
    return timeStamp;
  }

  //! set the last update time. Returns true upon success. \n	       
  //!
  //! @param timeval  value of the last update time 
  bool setTimeStamp (double timeval) {
    assert (timeval >= 0.0);
    timeStamp = timeval;
    return true;
  }

  //! Returns the next time at which the force field will be updated
  double getNextTimeStamp () const {
    return getTimeStamp() + getTimeStep();
  }

  //! increment the time stamp
  void incTimeStamp () {
    setTimeStamp(getNextTimeStamp()); 
  }

  //! Returns the field numbers used whose degrees of freedom participate in the force field computation 
  //!
  //! getFields()[i] returns the field number beginning from zero.\n
  //! If the degree of freedom \f$q_a\f$ is indexed with [f][b] locally, then it corresponds to field 
  //! getFields()[f] in the Global arrays.
  virtual const VecSize_t& getFields () const = 0;

  //! Returns the number of degrees of freedom per field used
  //! for the computation \n
  //!
  //! getFieldDof(fieldnum) returns the number of deegrees of freedom
  //! in the participating field number "fieldnumber". The argument 
  //! fieldnumber begins from  zero.\n
  //
  virtual size_t getFieldDof (size_t fieldnumber) const = 0;

  //! Returns the global element index for the AVI object
  virtual size_t getGlobalIndex (void) const = 0;

  virtual const DResidue& getOperation () const = 0;

  virtual const Element& getElement () const = 0;

  virtual const ElementGeometry& getGeometry () const = 0;

  //! write the updated time vector into the argument provided
  //! value filled in is the one obtained from getNextTimeStamp ()
  //! @param tnew
  virtual void computeLocalTvec (MatDouble& tnew) const = 0;

  //! Initialization routine to set values at the half time step
  //! This function is called only once per element and is called after gather
  //! and before update and assemble.
  //! @param q used to calculate the velocity at the half step.
  //! local vector with the last known values of the degrees of freedom, \f$q_a^{i-1}\f$ in
  //! the notation above.  \n
  //! @param v previous values of the time derivatives of velocity
  //! @param vb previous  values of the time derivatives at the
  //! middle of a time step
  //! of the degrees of freedom,  \f$v_a^{i-1/2}\f$ in the notation above.\n
  //! @param ti local vector with the last update time of a the degrees of freedom,
  //! \f$t_a^{i-1}\f$ in the notation above.\n
  //!
  //! @param tnew updated local time vector @see AVI::computeLocalTvec ()
  //! @param qnew updated value of q vector
  //! @param vbinit initialized value of vb vector
  //! @param forcefield temporary intermediate vector 
  //! @param funcval temporary intermediate vector 
  //! @param deltaV temporary intermediate vector 
  //! For example, q[f][b] = Q[L2G.map(ElementIndex,getFields()[f],b)]
  //!
  //! This function uses getImposedValues to fill the imposed values arrays.

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
    ) const = 0;

  //! Computes the value of the force field at a given configuration. Returns true upon success.
  //!
  //! @param argval values of the degrees of freedom for the force field of the object.
  //! argval[f][a] contains the value of the degree of freedom indexed with field "f" 
  //! and degree of freedom number in that field "a".
  //!
  //! @param forcefield values of \f$f_b\f$ upon exit. forcefield[f][a] contains the value
  //! of \f$f_b\f$ for the degree of freedom indexed with field "f" and degree of
  //! freedom number in that field "a". 
  virtual bool
  getForceField (const MatDouble& argval, MatDouble& forcefield) const = 0;

  //! update of the given forcefield. It returns the new values for the degrees of freedom
  //! and its time derivatives. These need yet to be assembled. Returns true upon success.
  //! 
  //! The forcefield time is updated. Of course, this does not happen with the last update time of each
  //! degree of freedom, which are updated in the assemble part.
  //!
  //! In the following vectors, index [f][b] indicates the value for the "b"-th degree of freedom 
  //! of field "f".
  //!
  //! @param q vector with the last known values of the degrees of freedom, \f$q_a^{i-1}\f$ in 
  //! the notation above.  \n
  //! @param v vector with the last known values of the time derivatives of the degrees of freedom,  
  //! \f$v_a^{i-1}\f$ in the notation above.\n
  //! @param vb vector with the last known values of the time derivatives at the middle of a time step
  //! of the degrees of freedom,  \f$v_a^{i-1/2}\f$ in the notation above.\n
  //! @param ti vector with the last update time of a the degrees of freedom, \f$t_a^{i-1}\f$ in 
  //! the notation above.\n
  //! @param tnew vector with the updated time of a the degrees of freedom, \f$t_a^{i}\f$ in 
  //! @param qnew Upon exit, new values of the degrees of freedom, \f$q_a^{i}\f$ in 
  //! the notation above.\n
  //! @param vnew Upon exit, new values of the time derivatives of the degrees of freedom, 
  //! \f$v_a^{i}\f$ in  the notation above.\n
  //! @param vbnew Upon exit, new values of the time derivatives at the middle of a time step
  //!  of the degrees of freedom, \f$v_a^{i+1/2}\f$ in  the notation above.\n
  //!
  //! @param forcefield temporary intermediate vector 
  //! @param funcval temporary intermediate vector 
  //! @param deltaV temporary intermediate vector 
  //!


  virtual bool update (
    const MatDouble& q,
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
    ) const = 0;

  //! Gathers the values needed from the global arrays to perform the force field computation.
  //! 
  //! 
  //! identify its global degrees of freedom. This information is not embedded in the object. \n
  //! @param L2G Local to Global map used to find the values in the Global arrays\n
  //! @param Qval Global array with the last updated values of the degrees of freedom.\n
  //! @param Vval Global array with the last updated values of the time derivatives of the
  //! degrees of freedom.\n
  //! @param Vbval Global array with the last updated values of the time derivatives  of the
  //! degrees of freedom at the middle of the time step.\n
  //! @param Tval Global array with the times at which the degrees of freedom were last updated.\n
  //! @param q Upon exit, local vector with the last known values of the degrees of freedom, \f$q_a^{i-1}\f$ in 
  //! the notation above.  \n
  //! @param v Upon exit, local vector with the last known values of the time derivatives of the 
  //! degrees of freedom,  
  //! \f$v_a^{i-1}\f$ in the notation above.\n
  //! @param vb Upon exit, local vector with the last known values of the time derivatives at the 
  //! middle of a time step
  //! of the degrees of freedom,  \f$v_a^{i-1/2}\f$ in the notation above.\n
  //! @param ti Upon exit, local vector with the last update time of a the degrees of freedom, 
  //! \f$t_a^{i-1}\f$ in the notation above.\n
  //!
  //! For example, q[f][b] = Q[L2G.map(ElementIndex,getFields()[f],b)]
  //!
  //! This function uses getImposedValues to fill the imposed values arrays.
  virtual bool
      gather ( const LocalToGlobalMap& L2G,
          const VecDouble& Qval,
          const VecDouble& Vval,
          const VecDouble& Vbval,
          const VecDouble& Tval, 
          MatDouble& q,
          MatDouble& v,
          MatDouble& vb,
          MatDouble& ti) const = 0;

  //! Assembles the updated values in the global array, including the latest time of update of the 
  //! degrees of freedom in the object.
  //! 
  //! 
  //! identify its global degrees of freedom. This information is not embedded in the object. \n
  //! @param L2G Local to Global map used to find the values in the Global arrays\n
  //! @param qnew  local vector with the updated values of the degrees of freedom, \f$q_a^{i}\f$ in 
  //! the notation above.  \n
  //! @param vnew Upon exit, local vector with the updated values of the time derivatives of the 
  //! degrees of freedom,  
  //! \f$v_a^{i}\f$ in the notation above.\n
  //! @param vbnew Upon exit, local vector with the updated values of the time derivatives at the 
  //! middle of a time step
  //! of the degrees of freedom,  \f$v_a^{i+1/2}\f$ in the notation above.\n
  //! @param tnew updated values of time vector
  //! @param Qval Global array where to assemble the updated values of the degrees of freedom.\n
  //! @param Vval Global array where to assemble the  time derivatives of the
  //! degrees of freedom.\n
  //! @param Vbval Global array where to assemble the time derivatives  of the
  //! degrees of freedom at the middle of the time step.\n
  //! @param Tval Global array where to assemble the times at which the degrees of freedom were last 
  //! updated.\n
  //! @param LUpdate Global array to keep track of which element updated the Dof last.  This is used
  //!    to keep Dofs from being updated out of order due to messaging delays.  contains the global
  //!    elem index of the last element to update each Dof.
  //!
  //! For example, Q[L2G.map(ElementIndex,getFields()[f],b)]=q[f][b]
  //!
  virtual bool assemble (const LocalToGlobalMap& L2G,
      const MatDouble& qnew,
      const MatDouble& vnew,
      const MatDouble& vbnew,
      const MatDouble& tnew,
      VecDouble& Qval,
      VecDouble& Vval,
      VecDouble& Vbval,
      VecDouble& Tval,
      VecDouble& LUpdate) const = 0;

protected:
  virtual void setTimeStep (double epsilon = 1.0) = 0;

  //! Solves the system of equations with the mass matrix for \f$\Delta v_b\f$. \n
  //! This is made a virtual function so that, if the mass matrix happens to be
  //! diagonal, it is done efficiently.
  //!
  //! @param funcval  values of \f$\Delta t f_a(q_a^i)\f$. 
  //! Notice that the multiplication by \f$\Delta t\f$ is already included. 
  //! funcval[f][b] contains the value of \f$\Delta t f_a(q_a^i)\f$ for the degree of
  //! freedom indexed with field "f" and degree of freedom number in that field "b".
  //!
  //! @param DeltaV values of \f$\Delta v_a\f$ upon exit. DeltaV[f][b] contains the value
  //! of \f$\Delta v_a\f$ for the degree of freedom indexed with field "f" and degree of
  //! freedom number in that field "b". 

  virtual void
  computeDeltaV (const MatDouble& funcval, MatDouble& DeltaV) const = 0;

  //! Imposed values of the degrees of freedom and its time derivative
  //!
  //! @param ElementIndex GlobalElementIndex or index of the force field in the AVI object, used to
  //! identify its global degrees of freedom. This information is not embedded in the object. \n
  //! @param L2G Local to Global map used to find the values in the Global arrays\n
  //! @param field field number for which the imposed values are sought, starting from zero.\n
  //! @param dof degree of freedom number within field "field" for which the imposed values are sought,
  //! starting from zero.\n
  //! @param qvalue upon exit, imposed value of the degree of freedom\n
  //! @param vvalue upon exit, imposed value of the time derivative of the degree of freedom\n
  //! 
  //! The values of ElementIndex and L2G are not always needed, but they are included here to provide
  //! a general interface. 
  //!
  //! \todo There has to be a cleaner way to deal with the value of the boundary condition other
  //! than providing a general interface to code them in the derived classes. 
  virtual bool getImposedValues (const GlobalElementIndex& ElementIndex,
      const LocalToGlobalMap& L2G, size_t field, size_t dof,
      double& qvalue, double& vvalue) const = 0;

public:
  //! @return string representation for printing debugging etc
  virtual const std::string toString () const {
    std::ostringstream ss;
    ss << "AVI(id: " << getGlobalIndex() << ", " << getNextTimeStamp() << " )";
    return ss.str ();
  }

  //! @return for use with std::ostream
  friend std::ostream& operator << (std::ostream& out, const AVI& avi) {
    out << avi.toString ();
    return out;
  }
};


/**
 * A comparator class for comparing two AVI objects
 * according to their time stamps
 */
struct AVIComparator {
//  static const double EPS = 1e-12;

  //! tie break comparison
  //! @param left pointer to first AVI object
  //! @param right pointer to second AVI object
  static inline int compare (const AVI* const left, const AVI* const right) {
    int result = DoubleComparator::compare (left->getNextTimeStamp (), right->getNextTimeStamp ());

    if (result == 0) {
      result = left->getGlobalIndex() - right->getGlobalIndex();
    }

    return result;
  }

  //! return true if left < right
  //! @param left pointer to first AVI object
  //! @param right pointer to second AVI object
  bool operator () (const AVI* const left, const AVI* const right) const {
    return compare (left, right) < 0;
  }
};

//! since C++ priority_queue is a max heap, this 
//! comparator allows using C++ priority_queue as a 
//! min heap by inverting the comparison
struct AVIReverseComparator: public AVIComparator {
  //! @returns true if left > right
  //! @param left pointer to first AVI object
  //! @param right pointer to second AVI object
  bool operator () (const AVI* const left, const AVI* const right) const {
    return compare (left, right) > 0;
  }
};


#endif
