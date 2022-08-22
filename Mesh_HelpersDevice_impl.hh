#pragma once

#include "Mesh_HelpersDevice_decl.hh"

namespace Amanzi {
namespace AmanziMesh {
namespace MeshAlgorithms {

template<>
KOKKOS_INLINE_FUNCTION
int getFaceDirectionInCell(const MeshCache<MemSpace_type::DEVICE>& mesh, const Entity_ID f, const Entity_ID c)
{
  int dir = 0;
  auto cf_cd = mesh.getCellFacesAndDirections(c);
  for (int j=0; j!=cf_cd.first.size(); ++j) {
    if (cf_cd.first[j] == f) {
      dir = cf_cd.second[j];
      break;
    }
  }
  return dir;
}



} // namespace MeshAlgorithms
} // namespace AmanziMesh
} // namspace Amanzi
