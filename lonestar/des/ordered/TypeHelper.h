/**
 * This file belongs to the Galois project, a C++ library for exploiting parallelism.
 * The code is being released under the terms of XYZ License (a copy is located in
 * LICENSE.txt at the top-level directory).
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

#ifndef DES_ORD_TYPE_HELPER_H
#define DES_ORD_TYPE_HELPER_H

#include "abstractMain.h"
#include "SimInit.h"


namespace des_ord {

template<des::NullEventOpt NULL_EVENT_OPT=des::NEEDS_NULL_EVENTS>
struct TypeHelper {
  typedef des::Event<des::LogicUpdate> Event_ty;
  typedef Event_ty::BaseSimObj_ty BaseSimObj_ty;
  typedef des_ord::SimObject<Event_ty> SimObj_ty;

  typedef des::SimGate<SimObj_ty> SimGate_ty;
  typedef des::Input<SimObj_ty> Input_ty;
  typedef des::Output<SimObj_ty> Output_ty;

  typedef des::SimInit<NULL_EVENT_OPT, SimGate_ty, Input_ty, Output_ty> SimInit_ty;
  
};

}
#endif // DES_ORD_TYPE_HELPER_H
