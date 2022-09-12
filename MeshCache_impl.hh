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
#include "Mesh_HelpersDevice_decl.hh"

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
MeshCacheBase::MeshCacheBase()
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
  : MeshCacheBase(other) {}


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
  ncells_all = framework_mesh_->getNumEntities(Entity_kind::CELL, Parallel_type::ALL);
  nfaces_owned = framework_mesh_->getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);
  nfaces_all = framework_mesh_->getNumEntities(Entity_kind::FACE, Parallel_type::ALL);
  nboundary_faces_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_FACE, Parallel_type::OWNED);
  nboundary_faces_all = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_FACE, Parallel_type::ALL);

  if (hasEdges()) {
    nedges_owned = framework_mesh_->getNumEntities(Entity_kind::EDGE, Parallel_type::OWNED);
    nedges_all = framework_mesh_->getNumEntities(Entity_kind::EDGE, Parallel_type::ALL);
  }
  if (hasNodes()) {
    nnodes_owned = framework_mesh_->getNumEntities(Entity_kind::NODE, Parallel_type::OWNED);
    nnodes_all = framework_mesh_->getNumEntities(Entity_kind::NODE, Parallel_type::ALL);
    nboundary_nodes_owned = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_NODE, Parallel_type::OWNED);
    nboundary_nodes_all = framework_mesh_->getNumEntities(Entity_kind::BOUNDARY_NODE, Parallel_type::ALL);
  }

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
KOKKOS_INLINE_FUNCTION void MeshCache<MEM>::throwAccessError_(const std::string& func_name) const
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
  static_assert(AP != AccessPattern::COMPUTE);
  static_assert(AP != AccessPattern::FRAMEWORK);
  // this is where a generic function would probably help?
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.cell_faces_cached);
    return data_.cell_faces.size<MEM>(c);
  } else {
    if (data_.cell_faces_cached) return getCellNumFaces<AccessPattern::CACHE>(c);
    return getCellFaces(c).size();
  }
}

template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
decltype(auto)
MeshCache<MEM>::getCellFaces(const Entity_ID c) const
{
  MeshCache<MEM>::data_type<Entity_ID> cfaces; 
  getCellFaces<AP>(c, cfaces);
  return cfaces; 
}


template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
void
MeshCache<MEM>::getCellFaces(const Entity_ID c,
  MeshCache<MEM>::data_type<Entity_ID>& cfaces) const
{
  cfaces = RaggedGetter<MEM,AP>::get(data_.cell_faces_cached,
    data_.cell_faces,
    [&](const int i) { assert(framework_mesh_.get()); 
      std::vector<Entity_ID> cf; 
      framework_mesh_->getCellFaces(i, cf);
      return cf; }, 
    nullptr, 
    c);
}



template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
const Entity_ID& MeshCache<MEM>::getCellFace(const Entity_ID c, const size_type i) const
{
  assert(data_.cell_faces_cached);
  return data_.cell_faces.get<MEM>(c,i);
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // Kokkos::pair<cEntity_ID_View, cEntity_Direction_View>
MeshCache<MEM>::getCellFacesAndDirections(const Entity_ID c) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    Entity_ID_List cfaces;
    Entity_Direction_List dirs;
    getCellFacesAndDirs(c, cfaces, &dirs);
    return Kokkos::pair(cfaces, dirs);
  } else {
    cEntity_ID_View cfaces;
    cEntity_Direction_View dirs;
    getCellFacesAndDirs(c, cfaces, &dirs);
    return Kokkos::pair(cfaces, dirs);
  }
}


template<MemSpace_type MEM>
template<typename cEntity_ID_View_type, typename cEntity_Direction_View_type>
KOKKOS_INLINE_FUNCTION
void
MeshCache<MEM>::getCellFacesAndDirs(const Entity_ID c,
                         cEntity_ID_View_type& faces,
                         cEntity_Direction_View_type * const dirs) const
{
  if constexpr(MEM == MemSpace_type::DEVICE) {
    static_assert(std::is_const_v<typename cEntity_ID_View_type::value_type>);
    static_assert(std::is_const_v<typename cEntity_Direction_View_type::value_type>);

    if (data_.cell_faces_cached) {
      faces = data_.cell_faces.getRow<MEM>(c);
      if (dirs) *dirs = data_.cell_face_directions.getRow<MEM>(c);
      return;
    }

  } else {
    if (data_.cell_faces_cached) {
      faces = asVector(data_.cell_faces.getRow<MEM>(c));
      if (dirs) *dirs = asVector(data_.cell_face_directions.getRow<MEM>(c));
      return;
    }

    if (framework_mesh_.get()) {
      framework_mesh_->getCellFacesAndDirs(c, faces, dirs);
      return;
    }
  }
  throwAccessError_("getCellFacesAndDirections");
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // Kokkos::pair<cEntity_ID_View, cPoint_View>
MeshCache<MEM>::getCellFacesAndBisectors(const Entity_ID c) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    Entity_ID_List cfaces;
    Point_List bisectors;
    getCellFacesAndBisectors(c, cfaces, &bisectors);
    return Kokkos::pair(cfaces, bisectors);
  } else {
    cEntity_ID_View cfaces;
    cPoint_View bisectors;
    getCellFacesAndBisectors(c, cfaces, &bisectors);
    return Kokkos::pair(cfaces, bisectors);
  }
}


template<MemSpace_type MEM>
template<typename cEntity_ID_View_type, typename cPoint_View_type>
KOKKOS_INLINE_FUNCTION
void
MeshCache<MEM>::getCellFacesAndBisectors(
  const Entity_ID c,
  cEntity_ID_View_type& faces,
  cPoint_View_type * const bisectors) const
{
  if constexpr(MEM == MemSpace_type::DEVICE) {
    static_assert(std::is_const_v<typename cEntity_ID_View_type::value_type>);
    static_assert(std::is_const_v<typename cPoint_View_type::value_type>);

    if (data_.cell_faces_cached) {
      faces = data_.cell_faces.getRow<MEM>(c);
      if (bisectors) *bisectors = data_.cell_face_bisectors.getRow<MEM>(c);
      return;
    }

  } else {
    if (data_.cell_faces_cached) {
      faces = asVector(data_.cell_faces.getRow<MEM>(c));
      if (bisectors) *bisectors = asVector(data_.cell_face_bisectors.getRow<MEM>(c));
      return;
    }

    if (framework_mesh_.get()) {
      framework_mesh_->getCellFacesAndBisectors(c, faces, bisectors);
      return;
    }
  }
  throwAccessError_("getCellFacesAndDirections");
}


// cell-face adjacencies
template<MemSpace_type MEM>
void MeshCache<MEM>::cacheCellFaces()
{
  if (data_.cell_faces_cached) return;

  auto lambda1 = [this](Entity_ID c, Entity_ID_List& cfaces) { this->framework_mesh_->getCellFaces(c, cfaces); };
  data_.cell_faces = asRaggedArray_DualView<Entity_ID>(lambda1, ncells_all);

  auto lambda2 = [this](Entity_ID c, Entity_Direction_List& dirs) { this->framework_mesh_->getCellFaceDirs(c, dirs); };
  data_.cell_face_directions = asRaggedArray_DualView<int>(lambda2, ncells_all);
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
  static_assert(AP != AccessPattern::COMPUTE);
  static_assert(AP != AccessPattern::FRAMEWORK);
  // this is where a generic function would probably help?
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
    if (data_.face_cells_cached) return getFaceNumCells<AccessPattern::CACHE>(f);
    return getFaceCells(f).size();
  }
}


// template<MemSpace_type MEM>
// template<class Entity_ID_View_type>
// KOKKOS_INLINE_FUNCTION
// void
// MeshCache<MEM>::getFaceCells(const Entity_ID f, const Parallel_type ptype, Entity_ID_View_type fcells) const
// {
//   if (data_.face_cells_cached) {
//     fcells = data_.face_cells.getRow<MEM>(f);

//   if constexpr(MEM == MemSpace_type::HOST) {
//     if (framework_mesh_.get()) {
//       Entity_ID_List fcells;
//       framework_mesh_->getFaceCells(f, ptype, fcells);
//       return fcells;
//     }
//   }
//   throwAccessError_("getFaceCells");
// }
// }


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // cEntity_ID_View
MeshCache<MEM>::getFaceCells(const Entity_ID f, const Parallel_type ptype) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    Entity_ID_List fcells;
    getFaceCells(f, ptype, fcells);
    return fcells;
  } else {
    cEntity_ID_View fcells;
    getFaceCells(f, ptype, fcells);
    return fcells;
  }
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
const Entity_ID&
MeshCache<MEM>::getFaceCell(const Entity_ID f, const size_type i) const
{
  assert(data_.face_cells_cached);
  return data_.face_cells.get<MEM>(f,i);
}


template<MemSpace_type MEM>
template<typename cEntity_ID_View_type>
KOKKOS_INLINE_FUNCTION
void
MeshCache<MEM>::getFaceCells(const Entity_ID f,
                             const Parallel_type ptype,
                             cEntity_ID_View_type& fcells) const
{
  if constexpr(MEM == MemSpace_type::DEVICE) {
    static_assert(std::is_const_v<typename cEntity_ID_View_type::value_type>);

    if (data_.face_cells_cached) {
      fcells = data_.face_cells.getRow<MEM>(f);
      return;
    }

  } else {
    if (data_.face_cells_cached) {
      fcells = asVector(data_.face_cells.getRow<MEM>(f));
      return;
    }
    if (framework_mesh_.get()) {
      framework_mesh_->getFaceCells(f, ptype, fcells);
      return;
    }
  }
  throwAccessError_("getFaceCells");
}


// cache
template<MemSpace_type MEM>
void MeshCache<MEM>::cacheFaceCells()
{
  if (data_.face_cells_cached) return;

  auto lambda = [this](Entity_ID f, Entity_ID_List& fcells) { this->framework_mesh_->getFaceCells(f, Parallel_type::OWNED, fcells); };
  data_.face_cells = asRaggedArray_DualView<Entity_ID>(lambda, nfaces_all);
  data_.face_cells_cached = true;
}


// -----------------------------------------------------------------------------
// Geometry
// -----------------------------------------------------------------------------
// Cell Geometry
// -----------------------------------------------------------------------------

// centroids
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
decltype(auto) MeshCache<MEM>::getCellCentroid(const Entity_ID c) const
{
  return Getter<MEM,AP>::get(data_.cell_geometry_cached,
    data_.cell_centroids,
    [&](const int i) { assert(framework_mesh_.get()); return framework_mesh_->getCellCentroid(i); }, 
    nullptr, 
    c);
}


// extent
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
decltype(auto) MeshCache<MEM>::getCellVolume(const Entity_ID c) const
{
  return Getter<MEM,AP>::get(data_.cell_geometry_cached,
    data_.cell_volumes,
    [&](const int i) { assert(framework_mesh_.get()); return framework_mesh_->getCellVolume(i); }, 
    nullptr, 
    c);
}

template<MemSpace_type MEM>
void MeshCache<MEM>::cacheCellGeometry()
{
  assert(framework_mesh_.get());
  if (data_.cell_geometry_cached) return;
  data_.cell_volumes.resize(ncells_all);
  data_.cell_centroids.resize(ncells_all);
  for (Entity_ID i=0; i!=ncells_all; ++i) {
    // note this must be on host
    std::tie(view<MemSpace_type::HOST>(data_.cell_volumes)[i],
             view<MemSpace_type::HOST>(data_.cell_centroids)[i]) =
      framework_mesh_->computeCellGeometry(i);
  }

  data_.cell_volumes.template modify<typename Point_DualView::host_mirror_space>();
  data_.cell_volumes.template sync<typename Point_DualView::execution_space>();
  data_.cell_centroids.template modify<typename Point_DualView::host_mirror_space>();
  data_.cell_centroids.template sync<typename Point_DualView::execution_space>();
  data_.cell_geometry_cached = true;
}


// -----------------------------------------------------------------------------
// Face Geometry
// -----------------------------------------------------------------------------
// face centroids
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point MeshCache<MEM>::getFaceCentroid(const Entity_ID f) const
{
  return Getter<MEM,AP>::get(data_.face_geometry_cached,
    data_.face_centroids,
    [&](const int i) { assert(framework_mesh_.get()); return framework_mesh_->getFaceCentroid(i); }, 
    nullptr, 
    f);
}

template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
double MeshCache<MEM>::getFaceArea(const Entity_ID f) const
{
  return Getter<MEM,AP>::get(data_.face_geometry_cached,
    data_.face_areas,
    [&](const int i) { assert(framework_mesh_.get()); return framework_mesh_->getFaceArea(i); }, 
    nullptr, 
    f);
}


// Normal vector of a face
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point MeshCache<MEM>::getFaceNormal(const Entity_ID f) const
{
  return getFaceNormal<AP>(f, -1, nullptr);
}

template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point MeshCache<MEM>::getFaceNormal(const Entity_ID f, const Entity_ID c,
        int* orientation) const
{
  AmanziGeometry::Point normal;
  if constexpr (MEM == MemSpace_type::DEVICE){
    assert(data_.face_geometry_cached); 
  }else {
    if(!data_.face_geometry_cached)
      if (framework_mesh_.get()) 
        return framework_mesh_->getFaceNormal(f, c, orientation);
  }

  auto fcells = getFaceCells(f, Parallel_type::ALL);
  if (orientation) *orientation = 0;

  Entity_ID cc;
  std::size_t i;
  if (c < 0) {
    cc = fcells[0];
    i = 0;
  } else {
    cc = c;
    auto ncells = fcells.size();
    for (i=0; i!=ncells; ++i)
      if (fcells[i] == cc) break;
  }
  normal = data_.face_normals.get<MEM>(f,i);

  if (getSpaceDimension() == getManifoldDimension()) {
    if (c < 0) {
      normal *= MeshAlgorithms::getFaceDirectionInCell(*this, f, cc);
    } else if (orientation) {
      *orientation = MeshAlgorithms::getFaceDirectionInCell(*this, f, cc);
    }
  } else if (c < 0) {
    Errors::Message msg("MeshFramework: asking for the natural normal of a submanifold mesh is not valid.");
    Exceptions::amanzi_throw(msg);
  }
  return normal;

}


template<MemSpace_type MEM>
void MeshCache<MEM>::cacheFaceGeometry()
{
  assert(framework_mesh_.get());
  if (data_.face_geometry_cached) return;
  data_.face_areas.resize(nfaces_all);
  data_.face_centroids.resize(nfaces_all);

  // slurp down the RaggedArray for normals using a lambda that, as a side
  // effect, captures area and centroid too.
  auto area_view = view<MemSpace_type::HOST>(data_.face_areas);
  auto centroid_view = view<MemSpace_type::HOST>(data_.face_centroids);
  auto lambda = [&,this](const Entity_ID& f, Point_List& normals) {
    auto area_cent_normal = this->framework_mesh_->computeFaceGeometry(f);
    area_view[f] = std::get<0>(area_cent_normal);
    centroid_view[f] = std::get<1>(area_cent_normal);
    normals = std::get<2>(area_cent_normal);
  };
  data_.face_normals = asRaggedArray_DualView<AmanziGeometry::Point>(lambda, nfaces_all);

  // still must sync areas/centroids
  data_.face_areas.template modify<typename Double_DualView::host_mirror_space>();
  data_.face_areas.template sync<typename Double_DualView::execution_space>();
  data_.face_centroids.template modify<typename Point_DualView::host_mirror_space>();
  data_.face_centroids.template sync<typename Point_DualView::execution_space>();
  data_.face_geometry_cached = true;

  // cache normal directions -- make this a separate call?  Think about
  // granularity here.
  auto lambda2 = [&,this](const Entity_ID& f, Entity_Direction_List& dirs) {
    // This NEEDS to call the framework or be passed an host mesh to call the function on the host. 
    Entity_ID_List fcells; 
    framework_mesh_->getFaceCells(f, Parallel_type::ALL, fcells);
    dirs.resize(fcells.size());
    for (int i=0; i!=fcells.size(); ++i) {
      this->framework_mesh_->getFaceNormal(f, fcells[i], &dirs[i]);
    }
  };
  data_.face_normal_directions = asRaggedArray_DualView<int>(lambda2, nfaces_all);

  // cache cell-face-bisectors -- make this a separate call?  Think about
  // granularity here.
  auto lambda3 = [&,this](const Entity_ID& c, Point_List& bisectors) {
    Entity_ID_List cfaces;
    this->framework_mesh_->getCellFacesAndBisectors(c, cfaces, &bisectors);
  };
  data_.cell_face_bisectors = asRaggedArray_DualView<AmanziGeometry::Point>(lambda3, ncells_all);

}



} // namespace AmanziMesh
} // namespace Amanzi


