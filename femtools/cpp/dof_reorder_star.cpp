// Copyright (C) 2008-2015 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Niclas Jansson 2009
// Modified by Garth N. Wells 2010-2012
// Modified by Mikael Mortensen, 2012.
// Modified by Joachim B Haga, 2012
// Modified by Martin Alnaes, 2013-2015
//
// Modified for stokes-beadpack, 2026-10-25, since moved to femtools: only
// DofMapBuilder::compute_node_reordering, copied from dolfin 2019.2.0.dev0, with the
// reordering graph a star a cell instead of a clique.

// The function dolfin orders a rank's owned dofs with, to be preloaded over libdolfin's own
// (femtools/dof_reorder.py builds it). dolfin joins every node of a cell to every other:
// Taylor-Hood has no common block size, so that is every dof, as many entries as the mixed
// matrix's pattern, and SCOTCH copies the graph. Here each owned node of a cell is joined to
// the cell's first, both ways, and SCOTCH orders that. Any permutation of a rank's owned nodes
// is a valid numbering, so the graph decides locality and nothing else.
// DOF_REORDER_GRAPH=clique restores dolfin's.
#include <algorithm>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
#include <boost/multi_array.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Set.h>
#include <dolfin/common/Timer.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/SCOTCH.h>
#include <dolfin/la/IndexMap.h>
#include <dolfin/log/log.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/fem/DofMapBuilder.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::compute_node_reordering(
  IndexMap& index_map,
  std::vector<int>& old_to_new_local,
  const std::unordered_map<int, std::vector<int>>& node_to_sharing_processes,
  const std::vector<std::size_t>& old_local_to_global,
  const std::vector<std::vector<la_index>>& node_dofmap,
  const std::vector<short int>& node_ownership,
  const std::set<std::size_t>& global_nodes,
  MPI_Comm mpi_comm)
{
  // Count number of locally owned nodes
  std::size_t owned_local_size = 0;
  std::size_t unowned_local_size = 0;
  for (auto node = node_ownership.begin(); node != node_ownership.end(); ++node)
  {
    if (*node >= 0)
      ++owned_local_size;
    else if (*node == -1)
      ++unowned_local_size;
    else
      dolfin_error("DofMap.cpp",
                   "compute node reordering",
                   "Invalid node ownership index.");
  }
  dolfin_assert((unowned_local_size+owned_local_size) == node_ownership.size());
  dolfin_assert((unowned_local_size+owned_local_size)
                == old_local_to_global.size());

  // Create global-to-local index map for local un-owned nodes
  std::vector<std::pair<std::size_t, int>> node_pairs;
  node_pairs.reserve(unowned_local_size);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] == -1)
      node_pairs.push_back(std::make_pair(old_local_to_global[i] , i));
  }
  std::map<std::size_t, int>
    global_to_local_nodes_unowned(node_pairs.begin(), node_pairs.end());
  std::vector<std::pair<std::size_t, int>>().swap(node_pairs);

  // Build graph for re-ordering. Below block is scoped to clear
  // working data structures once graph is constructed.
  Graph graph(owned_local_size);
  const char* shape = std::getenv("DOF_REORDER_GRAPH");
  const bool clique = shape && std::string(shape) == "clique";

  // Create contiguous local numbering for locally owned dofs
  std::size_t my_counter = 0;
  std::vector<int> old_to_contiguous_node_index(node_ownership.size(), -1);
  for (std::size_t i = 0; i < node_ownership.size(); ++i)
  {
    if (node_ownership[i] >= 0)
      old_to_contiguous_node_index[i] = my_counter++;
  }

  // Build local graph, based on old dof map, with contiguous
  // numbering
  for (std::size_t cell = 0; cell < node_dofmap.size(); ++cell)
  {
    // Cell dofmaps with old local indices
    const std::vector<la_index>& nodes = node_dofmap[cell];
    std::vector<int> local_old;

    // Loop over nodes collecting valid local nodes
    for (std::size_t i = 0; i < nodes.size(); ++i)
    {
      if (global_nodes.find(nodes[i]) != global_nodes.end())
        continue;

      // Old node index (0)
      const int n0_old = nodes[i];

      // New node index (0)
      dolfin_assert(n0_old < (int) old_to_contiguous_node_index.size());
      const int n0_local = old_to_contiguous_node_index[n0_old];

      // Add to graph if node n0_local is owned
      if (n0_local != -1)
      {
        dolfin_assert(n0_local < (int) graph.size());
        local_old.push_back(n0_local);
      }
    }

    if (clique)
    {
      for (std::size_t i = 0; i < local_old.size(); ++i)
        for (std::size_t j = 0; j < local_old.size(); ++j)
          if (i != j)
            graph[local_old[i]].insert(local_old[j]);
    }
    else
    {
      // a star: every owned node of the cell joined to the cell's first, both ways
      for (std::size_t i = 1; i < local_old.size(); ++i)
      {
        graph[local_old[0]].insert(local_old[i]);
        graph[local_old[i]].insert(local_old[0]);
      }
    }
  }

  // Reorder nodes
  const std::string ordering_library
    = dolfin::parameters["dof_ordering_library"];
  std::vector<int> node_remap;
  if (ordering_library == "Boost")
    node_remap = BoostGraphOrdering::compute_cuthill_mckee(graph, true);
  else if (ordering_library == "SCOTCH")
    node_remap = SCOTCH::compute_gps(graph);
  else if (ordering_library == "random")
  {
    // NOTE: Randomised dof ordering should only be used for
    // testing/benchmarking
    node_remap.resize(graph.size());
    for (std::size_t i = 0; i < node_remap.size(); ++i)
      node_remap[i] = i;
    std::random_shuffle(node_remap.begin(), node_remap.end());
  }
  else
  {
    dolfin_error("DofMapBuilder.cpp",
                 "reorder degrees of freedom",
                 "The requested ordering library '%s' is unknown",
                 ordering_library.c_str());
  }

  // Compute offset for owned nodes
  const std::size_t process_offset
    = MPI::global_offset(mpi_comm, owned_local_size, true);

  // Allocate space
  old_to_new_local.clear();
  old_to_new_local.resize(node_ownership.size(), -1);

  // Renumber owned nodes, and buffer nodes that are owned but shared
  // with another process
  const std::size_t mpi_size = MPI::size(mpi_comm);
  std::vector<std::vector<std::size_t>> send_buffer(mpi_size);
  std::vector<std::vector<std::size_t>> recv_buffer(mpi_size);
  std::size_t counter = 0;
  for (std::size_t old_node_index_local = 0;
       old_node_index_local < node_ownership.size();
       ++old_node_index_local)
  {
    // Skip nodes that are not owned (will receive global index later)
    if (node_ownership[old_node_index_local] < 0)
      continue;

    // Set new node number
    dolfin_assert(counter < node_remap.size());
    dolfin_assert(old_node_index_local < old_to_new_local.size());
    old_to_new_local[old_node_index_local] = node_remap[counter];

    // If this node is shared and owned, buffer old and new (global)
    // node index for sending
    if (node_ownership[old_node_index_local] == 0)
    {
      auto it = node_to_sharing_processes.find(old_node_index_local);
      if (it != node_to_sharing_processes.end())
      {
        for (auto p = it->second.begin(); p != it->second.end(); ++p)
        {
          // Buffer old and new global indices to send
          send_buffer[*p].push_back(old_local_to_global[old_node_index_local]);
          send_buffer[*p].push_back(process_offset + node_remap[counter]);
        }
      }

    }
    ++counter;
  }

  MPI::all_to_all(mpi_comm, send_buffer, recv_buffer);

  std::vector<std::size_t> local_to_global_unowned(unowned_local_size);
  //  off_process_owner.resize(unowned_local_size);
  std::size_t off_process_node_counter = 0;

  for (std::size_t src = 0; src != mpi_size; ++src)
    for (auto q = recv_buffer[src].begin();
         q != recv_buffer[src].end(); q += 2)
    {
      const std::size_t received_old_node_index_global = *q;
      const std::size_t received_new_node_index_global = *(q + 1);

      auto it
        = global_to_local_nodes_unowned.find(received_old_node_index_global);
      dolfin_assert(it != global_to_local_nodes_unowned.end());

      const int received_old_node_index_local = it->second;
      local_to_global_unowned[off_process_node_counter]
        = received_new_node_index_global;
      // off_process_owner[off_process_node_counter] = src;

      const int new_index_local = owned_local_size + off_process_node_counter;
      dolfin_assert(old_to_new_local[received_old_node_index_local] < 0);
      old_to_new_local[received_old_node_index_local] = new_index_local;
      off_process_node_counter++;
    }

  index_map.set_local_to_global(local_to_global_unowned);

  // Sanity check
  for (auto it : old_to_new_local)
  {
    dolfin_assert(it != -1);
  }
}
