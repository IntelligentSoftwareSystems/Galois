/**
 * ElementalOperation.h
 * DG++
 *
 * Created by Adrian Lew on 10/25/06.
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

#ifndef ELEMENTALOPERATION
#define ELEMENTALOPERATION

#include <vector>
#include <iostream>

#include <cstdlib>

#include "AuxDefs.h"
#include "Element.h"
#include "Material.h"


/**
   \brief Computes a residue  on an element

   A Residue object computes the values of a vector function of some or all the 
   fields in the element, with the distinguishing feature that there is one
   component of the function per degree of freedom of the participating fields. 
   
   A Residue object contains two concepts:\n
   1) A procedure to compute the value of the function\n
   2) A pointer to the element over which to compute it\n 

   Each residue acts then as a container of a pointer to an element object, 
   and an operation to perform on that object. In this way the same element can be
   used for several operation in a single computation (The alternative of adding 
   more layers of inheritance to the element classes makes the last possibility very 
   clumsy).
   
   Additionally, operation that need higher-levels of specialization, such as special classes
   of elements, can perform type-checking in their own implementation. 

   The class residue is in fact very similar to a general container,
   in the sense that the object it points to does not need to be an
   element but can be any object that permits the computation of the
   function and can use the (field, dof) notation to label the inputs and 
   outputs.
   
   The precise fields to be utilized in the computation of the
   operation may vary from element type to element type, hence these
   will generally be specified.

   More precisely, a residual is a function 
   \f[
   F^f_a(u^0_0,u^0_1, \ldots, u^0_{n_0-1}, u^0_1, \ldots, u^{K}_{n_{K-1}-1}),
   \f]
   where \f$u^f_a\f$  is the a-th degree of freedom of the f-th participating field in 
   the function. A total of K of the element fields participate as arguments in the 
   function. The f-th participating field has a total of \f$n_f\f$ degrees of freedom. 
   The coefficient of the force "f" runs from 0 to K-1, and "a" ia the degree of freedom 
   index that ranges from 0 to \f$n_f\f$. 

   We need to specify  specify which field in the element will represent the f-th 
   participating field in the function F. For instance, the field number 2 in the element
   can be used as the first argument of F, i.e., as participating field number 0.

   \todo This class does not accept the input of additional parameters that may
   be needed for the evaluation of T that may not be solved for.
   
   \todo The assembly  procedure should probably be changed
*/
class Residue
{
 public:
  Residue() {}
  virtual ~Residue() {}
  Residue(const Residue &NewEl) {}
  virtual Residue * clone() const = 0;

  //! Returns the fields used for the computation of the residue\n
  //!
  //! getFields()[i] returns the field number beginning from zero.\n
  //! The variable \f$u^f_a\f$ is then computed with field getFields()[f]
  //! in the element.
  virtual const VecSize_t& getFields() const = 0;


  //! Returns the number of degrees of freedom per field used
  //! for the computation of the residue\n
  //!
  //! getFieldDof(fieldnum) returns the number of deegrees of freedom
  //! in the participating fieldo number "fieldnumber". The argument 
  //! fieldnumber begins from  zero.\n
  //! The number of different values of "a" in  \f$u^{f}_a\f$ is
  //! then computed with field getFieldDof(f)
  virtual size_t  getFieldDof(size_t fieldnumber) const = 0;


  //! Returns the value of the residue given the values of the fields.
  //!
  //! @param argval vector of vector<double> containing the values of the degrees of 
  //! freedom for each field.\n
  //! argval[f][a] contains the value of degree of freedom "a" for participating
  //! field "f".
  //!
  //! @param funcval It  returns a vector< vector<double> > with the values of each
  //! component of the residual function. We have that \f$F^f_a\f$=funcval[f][a].
  //! The vector funcval is resized and zeroed in getVal.
  //! 
  //!
  //! The function returns true if successful, false otherwise.
  virtual bool getVal(const MatDouble &argval, MatDouble& funcval ) const = 0;

  virtual const Element& getElement () const = 0;

  virtual const SimpleMaterial& getMaterial () const = 0;


  //! \brief assemble Residual Vector
  //! 
  //! Assembles the contributions from an Array of residual objects ResArray 
  //! into ResVec. The contribution from each ResArray, ResArray[e], is mapped
  //! into ResVec  with a LocalToGlobalMap. 
  //!
  //!
  //! @param ResArray Array of residue  objects
  //! @param L2G LocalToGlobalMap
  //! @param Dofs PetscVector with values of degrees of freedom
  //! @param ResVec Pointer to a PetscVector where to assemble the residues.
  //! ResVec is zeroed in assemble
  //!
  //! This is precisely what's done:\n
  //! 1) assemble input gathered as \n
  //!        argval[f][a] = Dofs[L2G(f,a)]\n
  //! 2) Computation of the local residue funcval as
  //!        ResArray[i]->getVal(argval, funcval)\n
  //! 3) assemble output gathered as\n
  //!        ResVec[L2G(f,a)] += funcval[f][a]\n
  //!
  //! Behavior:\n
  //! Successful assembly returns true, unsuccessful false
  //!
  //! \warning: The residue object that computes the contributions of element 
  //! "e" is required to have position "e" in ResArray as well, so that 
  //! the  LocalToGlobalMap object is used consistently
  //!
  //! \todo A defect of this implementation is that all fields in the element enter as 
  //! arguments for Residue and its derivative. It would be good to have the flexibility to extract
  //! a subset of the degrees of freedom of the element as the argument to Residue and its 
  //! derivative.  In this way it is possible to act with different elemental operation on 
  //! different degrees of freedom naturally, i.e., without having to artificially return a Residue
  //! that has presumed contributions from all degrees of freedom.
  

  static bool assemble(std::vector<Residue *> &ResArray, 
		       const LocalToGlobalMap & L2G,
		       const VecDouble  & Dofs,
		       VecDouble&  ResVec);
};


/**
 * Base class for common functionality
 */

class BaseResidue: public Residue {

protected:
  const Element& element;
  const SimpleMaterial& material;
  const VecSize_t& fieldsUsed;


  BaseResidue (const Element& element, const SimpleMaterial& material, const VecSize_t& fieldsUsed)
  : element (element), material (material), fieldsUsed (fieldsUsed) {
  }

  BaseResidue (const BaseResidue& that) : element (that.element), material (that.material)
  , fieldsUsed (that.fieldsUsed) {

  }

public:
  virtual const Element& getElement () const {
    return element;
  }

  virtual const VecSize_t& getFields () const {
    return fieldsUsed;
  }

  virtual const SimpleMaterial& getMaterial () const {
    return material;
  }

  virtual size_t getFieldDof (size_t fieldNum) const {
    return element.getDof (fieldsUsed[fieldNum]);
  }

};




/**
   \brief Computes a residue and its derivative  on an element

   See Residue class for an explanation.

   This class just adds a function getDVal that contains a vector 
   to return the derivative

*/

class DResidue: public BaseResidue {
 public: 
  DResidue (const Element& element, const SimpleMaterial& material, const VecSize_t& fieldsUsed)
  : BaseResidue(element, material, fieldsUsed) {}

  DResidue(const DResidue &NewEl): BaseResidue(NewEl) {}

  virtual DResidue * clone() const = 0;
 

  //! Returns the value of the residue and its derivative given the values of the fields.
  //!
  //! @param argval vector of vector<double> containing the values of the degrees of 
  //! freedom for each field.\n
  //! argval[f][a] contains the value of degree of freedom "a" for participating
  //! field "f".
  //!
  //! @param funcval It  returns a vector< vector<double> > with the values of each
  //! component of the residual function. We have that \f$F^f_a\f$=funcval[f][a].
  //! The vector funcval is resized and zeroed in getVal.
  //! 
  //! @param dfuncval It  returns a vector< vector< vector< vector<double> > > >
  //! with the values of each
  //! component of the derivative of the residual function. 
  //! We have that \f$\frac{\partial F^f_a}{\partial u^g_b}\f$=dfuncval[f][a][g][b].
  //! The vector dfuncval is resized and zeroed in getVal.
  //!
  //! The function returns true if successful, false otherwise.
  virtual bool getDVal(const MatDouble& argval, MatDouble& funcval,
		       FourDVecDouble& dfuncval) const = 0;

  
  //! Consistency test for DResidues. 
  static bool consistencyTest(const DResidue & DRes,  
			      const VecSize_t& DofPerField,
			      const MatDouble &argval);

  //! \brief assemble Residual Vector and it Derivative
  //! 
  //! Assembles the contributions from an Array of dresidual objects DResArray 
  //! into DResVec. The contribution from each DResArray, DResArray[e], is 
  //! mapped into DResVec  with a LocalToGlobalMap. 
  //!
  //! @param DResArray Array of dresidue  objects
  //! @param L2G LocalToGlobalMap
  //! @param Dofs PetscVector with values of degrees of freedom
  //! @param ResVec Pointer to a PetscVector where to assemble the dresidues. 
  //! ResVec is zeroed in assemble
  //! @param DResMat Pointer to a PetscVector where to assemble the dresidues. 
  //! DResMat is zeroed in assemble
  //!
  //! This is precisely what's done:\n
  //! 1) assemble input gathered as \n
  //!        argval[f][a] = Dofs[L2G(f,a)]\n
  //! 2) Computation of the local residue  funcval and its derivative dfucnval as
  //!        DResArray[i]->getVal(argval, funcval, dfuncval)\n
  //! 3) assemble output gathered as\n
  //!        ResVec[L2G(f,a)] += funcval[f][a]\n
  //!        DResMat[L2G(f,a)][L2G(g,b)] += dfuncval[f][a][g][b]
  //!
  //! Behavior:\n
  //! Successful assembly returns true, unsuccessful false
  //!
  //! \warning: The residue object that computes the contributions of element 
  //! "e" is required to have position "e" in DResArray as well, so that 
  //! the  LocalToGlobalMap object is used consistently
  //!
  //! \todo This  structure has to be revised. In the implementation of both 
  //! assemble functions I had to use a dynamic_cast to prevent writing two
  //! versions of essentially the same code, one for residue and another for 
  //  DResidue. This should have been possible through polymorphism.
  //! There must be a flaw in the abstraction.
  

  static bool assemble(std::vector<DResidue *> &DResArray, 
		       const LocalToGlobalMap & L2G,
		       const VecDouble  & Dofs,
		       VecDouble&  ResVec,
		       MatDouble&  DResMat);
};



#endif

