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
