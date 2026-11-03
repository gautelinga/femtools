// Probes is ProbeCollection<Probe>, whose code is in its header; this unit only
// instantiates it once.
#include "Probes.h"

template class dolfin::ProbeCollection<dolfin::Probe>;
