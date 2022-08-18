/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Ethan Coon (coonet@ornl.gov)
           Julien Loiseau (jloiseau@lanl.gov)
           Rao Garimella (rao@lanl.gov)
*/
//! Caches mesh information for fast repeated access.

#pragma once

#include "MeshCache_decl.hh"
#include "KokkosUtils.hh"
#include "MeshFramework.hh"


namespace Amanzi {
namespace AmanziMesh {

template<typename T, typename Func>
RaggedArray_DualView<T>
asRaggedArray_DualView(Func mesh_func, Entity_ID count)
{
  RaggedArray_DualView<T> adj;
  adj.rows.resize(count+1);

  // do a count first, setting rows
  std::vector<T> ents;
  int total = 0;
  for (Entity_ID i=0; i!=count; ++i) {
    view<MemSpace_type::HOST>(adj.rows)[i] = total;

    mesh_func(i, ents);
    total += ents.size();
  }
  view<MemSpace_type::HOST>(adj.rows)[count] = total;
  adj.entries.resize(total);

  for (Entity_ID i=0; i!=count; ++i) {
    mesh_func(i, ents);
    Kokkos::View<T*, Kokkos::DefaultHostExecutionSpace> row_view = adj.template getRow<MemSpace_type::HOST>(i);
    my_deep_copy(row_view, ents);
  }

  adj.rows.template modify<typename RaggedArray_DualView<Entity_ID>::host_mirror_space>();
  adj.rows.template sync<typename RaggedArray_DualView<Entity_ID>::execution_space>();
  adj.entries.template modify<typename RaggedArray_DualView<Entity_ID>::host_mirror_space>();
  adj.entries.template sync<typename RaggedArray_DualView<Entity_ID>::execution_space>();
  return adj;
}



// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------
template<MemSpace_type MEM>
MeshCache<MEM>::MeshCache()
  : is_ordered_(false),
    is_logical_(false),
    has_edges_(false),
    has_nodes_(true),
    manifold_dim_(-1),
    space_dim_(-1)
{}


template<MemSpace_type MEM>
template<MemSpace_type MEM_OTHER>
MeshCache<MEM>::MeshCache(MeshCache<MEM_OTHER>& other)
  : data_(other.data_),
    is_ordered_(other.is_ordered_),
    is_logical_(other.is_logical_),
    has_edges_(other.has_edges_),
    has_nodes_(other.has_nodes_),
    manifold_dim_(other.manifold_dim_),
    space_dim_(other.space_dim_),
    framework_mesh_(other.framework_mesh_),
    algorithms_(other.algorithms_)
{}


// -----------------------------------------------------------------------------
// Accessors / Mutators
// -----------------------------------------------------------------------------
template<MemSpace_type MEM>
void
MeshCache<MEM>::setMeshFramework(const std::shared_ptr<MeshFramework>& framework_mesh)
{
  framework_mesh_ = framework_mesh;
  // always save the algorithms, so we can throw away the data
  algorithms_ = framework_mesh->getAlgorithms();
  has_edges_ = framework_mesh->hasEdges();
  has_nodes_ = framework_mesh->hasNodes();
  // comm_ = framework_mesh_->getComm();
  // gm_ = framework_mesh_->getGeometricModel();
  space_dim_ = framework_mesh_->getSpaceDimension();
  manifold_dim_ = framework_mesh_->getManifoldDimension();
  is_logical_ = framework_mesh_->isLogical();

  is_ordered_ = framework_mesh_->isOrdered();
  has_edges_ = framework_mesh_->hasEdges();

  // bool natural_ordered_maps = plist_->get<bool>("natural map ordering", false);
  // maps_.initialize(*framework_mesh_, natural_ordered_maps);

  ncells_owned = framework_mesh_->getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
  ncells_owned = framework_mesh_->getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
  nfaces_owned = framework_mesh_->getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);
  nfaces_owned = framework_mesh_->getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);
  nedges_owned = framework_mesh_->getNumEntities(Entity_kind::EDGE, Parallel_type::OWNED);
  nedges_owned = framework_mesh_->getNumEntities(Entity_kind::EDGE, Parallel_type::OWNED);
  nnodes_owned = framework_mesh_->getNumEntities(Entity_kind::NODE, Parallel_type::OWNED);
  nnodes_owned = framework_mesh_->getNumEntities(Entity_kind::NODE, Parallel_type::OWNED);
  nboundary_faces_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_FACE, Parallel_type::OWNED);
  nboundary_faces_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_FACE, Parallel_type::OWNED);
  nboundary_nodes_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_NODE, Parallel_type::OWNED);
  nboundary_nodes_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_NODE, Parallel_type::OWNED);
}


template<MemSpace_type MEM>
Entity_ID
MeshCache<MEM>::getNumEntities(const Entity_kind kind, const Parallel_type ptype) const
{
  Entity_ID nowned, nall;
  switch(kind) {
    case (Entity_kind::CELL) :
      nowned = ncells_owned; nall = ncells_all;
      break;
    case (Entity_kind::FACE) :
      nowned = nfaces_owned; nall = nfaces_all;
      break;
    case (Entity_kind::EDGE) :
      nowned = nedges_owned; nall = nedges_all;
      break;
    case (Entity_kind::NODE) :
      nowned = nnodes_owned; nall = nnodes_all;
      break;
    case (Entity_kind::BOUNDARY_FACE) :
      nowned = nboundary_faces_owned; nall = nboundary_faces_all;
      break;
    case (Entity_kind::BOUNDARY_NODE) :
      nowned = nboundary_nodes_owned; nall = nboundary_nodes_all;
      break;
    default :
      nowned = -1; nall = -1;
  }

  switch(ptype) {
    case (Parallel_type::OWNED) :
      return nowned;
      break;
    case (Parallel_type::ALL) :
      return nall;
      break;
    case Parallel_type::GHOST :
      return nall - nowned;
      break;
    default :
      return 0;
  }
}


// common error messaging
template<MemSpace_type MEM>
void MeshCache<MEM>::throwAccessError_(const std::string& func_name) const
{
  Errors::Message msg;
  msg << "MeshCache<MEM>" << func_name << " cannot compute this quantity -- not cached and framework does not exist.";
  Exceptions::amanzi_throw(msg);
}


// -----------------------------------------------------------------------------
// Topology
// -----------------------------------------------------------------------------

//---------------------
// Downward adjacencies
//---------------------
// Get faces of a cell
//
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
size_type
MeshCache<MEM>::getCellNumFaces(const Entity_ID c) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.cell_faces_cached);
    return data_.cell_faces.size<MEM>(c);
  } else {
    if (data_.cell_faces_cached) return getCellNumFaces<AccessPattern::CACHE>(c);
    return getCellFaces(c).size();
  }
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto)
MeshCache<MEM>::getCellFaces(const Entity_ID c) const
{
  if (data_.cell_faces_cached) return data_.cell_faces.getRow<MEM>(c);

  if constexpr(MEM == MemSpace_type::HOST) {
    if (framework_mesh_.get()) {
      Entity_ID_List cfaces;
      framework_mesh_->getCellFaces(c, cfaces);
      return cfaces;
    }
  }
  throwAccessError_("getCellNumFaces");
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
const Entity_ID& MeshCache<MEM>::getCellFace(const Entity_ID c, const size_type i) const
{
  assert(data_.cell_faces_cached);
  return data_.cell_faces.get<MEM>(c,i);
}


// cell-face adjacencies
template<MemSpace_type MEM>
void MeshCache<MEM>::cacheCellFaces()
{
  if (data_.cell_faces_cached) return;

  auto get_cell_faces = [=](Entity_ID c, Entity_ID_List& cfaces) { framework_mesh_->getCellFaces(c, cfaces); };
  data_.cell_faces = asRaggedArray_DualView<Entity_ID>(get_cell_faces, ncells_all);

  auto get_cell_face_dirs = [=](Entity_ID c, Entity_Direction_List& dirs) { framework_mesh_->getCellFaceDirs(c, dirs); };
  data_.cell_face_directions = asRaggedArray_DualView<int>(get_cell_face_dirs, ncells_all);
  data_.cell_faces_cached = true;
}


//-------------------
// Upward adjacencies
//-------------------
// The cells are returned in no particular order. Also, the order of cells
// is not guaranteed to be the same for corresponding faces on different
// processors
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
size_type
MeshCache<MEM>::getFaceNumCells(const Entity_ID f, const Parallel_type ptype) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.face_cells_cached);
    if (ptype == Parallel_type::ALL) {
      return data_.face_cells.size<MEM>(f);
    } else {
      int count = 0;
      int n_all = data_.face_cells.size<MEM>(f);
      for (int j=0; j!=n_all; ++j) {
        if (getFaceCell(f,j) < ncells_owned) ++count;
        else break;
      }
      return count;
    }

  } else {
    if (data_.face_cells_cached) return getFaceNumCells<AccessPattern::CACHE>(f, ptype);
    return getFaceCells(f, ptype).size();
  }
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // cEntity_ID_View
MeshCache<MEM>::getFaceCells(const Entity_ID f, const Parallel_type ptype) const
{
  if (data_.face_cells_cached) return data_.face_cells.getRow<MEM>(f);

  if constexpr(MEM == MemSpace_type::HOST) {
    if (framework_mesh_.get()) {
      Entity_ID_List fcells;
      framework_mesh_->getFaceCells(f, ptype, fcells);
      return fcells;
    }
  }
  throwAccessError_("getCellNumFaces");
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
const Entity_ID&
MeshCache<MEM>::getFaceCell(const Entity_ID f, const size_type i) const
{
  assert(data_.face_cells_cached);
  return data_.face_cells.get<MEM>(f,i);
}


// cache
template<MemSpace_type MEM>
void MeshCache<MEM>::cacheFaceCells()
{
  if (data_.face_cells_cached) return;

  auto get_face_cells = [=](Entity_ID f, Entity_ID_List& fcells) { framework_mesh_->getFaceCells(f, Parallel_type::OWNED, fcells); };
  data_.face_cells = asRaggedArray_DualView<Entity_ID>(get_face_cells, nfaces_all);
  data_.face_cells_cached = true;
}


} // namespace AmanziMesh
} // namespace Amanzi


