#ifndef __PROBES_H
#define __PROBES_H

#include "Probe.h"
#include "ProbeCollection.h"

namespace dolfin
{

  // A Function's values at many points (`ProbeCollection`)
  class Probes : public ProbeCollection<Probe>
  {

  public:

    using ProbeCollection<Probe>::ProbeCollection;

  };

}

#endif
