"""femtools: tools for legacy-dolfin solvers.

    femtools.xdmf        XDMFFile2, a drop-in for dolfin.XDMFFile that keeps P2
    femtools.compress    every HDF5 dataset shuffled and gzipped
    femtools.hdf5_rows   rows of HDF5 datasets read, written and exchanged between ranks
    femtools.facets      a facet file cut to selected markers (femtools-cut-facets)
    femtools.probes      Probes and GradProbes, from fenicstools (compiled on import)
    femtools.gradient    the weighted gradient matrix, from fenicstools (compiled on import)
    femtools.periodic    the periodic box a constrained_domain is built with, and its checks
    femtools.dolfin_layout  a Mesh or Function written as dolfin's HDF5File writes it, compressed
    femtools.periodic_partition  a parallel mesh partition the periodic dofmap cannot break
    femtools.abort       a command's entry point: one rank's failure ends the MPI job
    femtools.memory      freed heap handed back to the operating system

`import femtools` imports none of them; xdmf, compress, hdf5_rows, facets and
dolfin_layout need numpy and h5py alone, abort and memory the standard library.
"""
__version__ = "0.1.0"
