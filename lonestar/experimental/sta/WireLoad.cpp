#include "WireLoad.h"

static IdealWireLoad idealWL;
WireLoad* idealWireLoad = &idealWL;

static SDFWireLoad sdfWL;
WireLoad* sdfWireLoad = &sdfWL;
