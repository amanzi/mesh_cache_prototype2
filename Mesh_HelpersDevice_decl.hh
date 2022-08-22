#pragma once

#include "Mesh_Helpers_decl.hh"
#include "MeshCache_decl.hh"

namespace Amanzi {
namespace AmanziMesh {
namespace MeshAlgorithms {

template<>
KOKKOS_INLINE_FUNCTION
int getFaceDirectionInCell(const MeshCache<MemSpace_type::DEVICE>& mesh, const Entity_ID f, const Entity_ID c);


} // namespace MeshAlgorithms
} // namespace AmanziMesh
} // namspace Amanzi
