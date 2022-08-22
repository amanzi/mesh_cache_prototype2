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

  adj.template update<MemSpace_type::DEVICE>(); 
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
MeshCache<MEM>::MeshCache()
  : MeshCacheBase() {}

template<MemSpace_type MEM>
template<MemSpace_type MEM_OTHER>
MeshCache<MEM>::MeshCache(MeshCache<MEM_OTHER>& other)
{ data_ = other.data_;
  is_ordered_ = other.is_ordered_;
  is_logical_ =other.is_logical_;
  has_edges_ = other.has_edges_;
  has_nodes_ = other.has_nodes_;
  manifold_dim_ = other.manifold_dim_;
  space_dim_ = other.space_dim_;
  framework_mesh_ = other.framework_mesh_;
  algorithms_ = other.algorithms_;
  ncells_owned = other.ncells_owned; 
  ncells_all = other.ncells_all;
  nfaces_owned = other.nfaces_owned; 
  nfaces_all = other.nfaces_all;
  nedges_owned = other.nedges_owned;
  nedges_all = other.nedges_all;
  nnodes_owned = other.nnodes_owned; 
  nnodes_all = other.nnodes_all;
  nboundary_faces_owned = other.nboundary_faces_owned; 
  nboundary_faces_all = other.nboundary_faces_all;
  nboundary_nodes_owned = other.nboundary_nodes_owned;
  nboundary_nodes_all = other.nboundary_nodes_all;
}


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
KOKKOS_INLINE_FUNCTION
decltype(auto)
MeshCache<MEM>::getCellFaces(const Entity_ID c) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    if (data_.cell_faces_cached){ 
      return asVector(data_.cell_faces.getRow<MEM>(c));
    }
    if (framework_mesh_.get()) {
      Entity_ID_List cfaces;
      framework_mesh_->getCellFaces(c, cfaces);
      return cfaces;
    }
  }else{
    return data_.cell_faces.getRow<MEM>(c);
  }
  throwAccessError_("getCellFaces");
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
decltype(auto) // cEntity_Direction_View
MeshCache<MEM>::getCellFaceDirections(const Entity_ID c) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    if (data_.cell_faces_cached) 
      return asVector(data_.cell_face_directions.getRow<MEM>(c));
    if (framework_mesh_.get()) {
      Entity_Direction_List cfdirs;
      framework_mesh_->getCellFaceDirs(c, cfdirs);
      return cfdirs;
    }
  }else{ 
    return data_.cell_face_directions.getRow<MEM>(c);
  }
  throwAccessError_("getCellFaceDirections");
}


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // Kokkos::pair<cEntity_ID_View, cEntity_Direction_View>
MeshCache<MEM>::getCellFacesAndDirections(const Entity_ID c) const
{

  if constexpr(MEM == MemSpace_type::HOST) {
    if (data_.cell_faces_cached) {
      return Kokkos::pair(asVector(data_.cell_faces.getRow<MEM>(c)),
                       asVector(data_.cell_face_directions.getRow<MEM>(c)));
    }
    if (framework_mesh_.get()) {
      Entity_Direction_List cfdirs;
      Entity_ID_List cfaces;
      framework_mesh_->getCellFacesAndDirs(c, cfaces, &cfdirs);
      return Kokkos::pair(cfaces, cfdirs);
    }
  }else{ 
    return Kokkos::pair(data_.cell_faces.getRow<MEM>(c),
                        data_.cell_face_directions.getRow<MEM>(c));
  }
  throwAccessError_("getCellFacesAndDirections");
}

template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // Kokkos::pair<cEntity_ID_View, cPoint_View>
MeshCache<MEM>::getCellFacesAndBisectors(const Entity_ID c) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    if (data_.cell_faces_cached) {
      return Kokkos::pair(asVector(data_.cell_faces.getRow<MEM>(c)),
                        asVector(data_.cell_face_bisectors.getRow<MEM>(c)));
    }
    if (framework_mesh_.get()) {
      Point_List bisectors;
      Entity_ID_List cfaces;
      framework_mesh_->getCellFacesAndBisectors(c, cfaces, &bisectors);
      return Kokkos::pair(cfaces, bisectors);
    }
  }else{
    return Kokkos::pair(data_.cell_faces.getRow<MEM>(c),
                        data_.cell_face_bisectors.getRow<MEM>(c));
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


template<MemSpace_type MEM>
KOKKOS_INLINE_FUNCTION
decltype(auto) // cEntity_ID_View
MeshCache<MEM>::getFaceCells(const Entity_ID f, const Parallel_type ptype) const
{
  if constexpr(MEM == MemSpace_type::HOST) {
    if (data_.face_cells_cached) return asVector(data_.face_cells.getRow<MEM>(f));
    if (framework_mesh_.get()) {
      Entity_ID_List fcells;
      framework_mesh_->getFaceCells(f, ptype, fcells);
      return fcells;
    }
  }else{
    return data_.face_cells.getRow<MEM>(f);
  }
  throwAccessError_("getCellFaces");
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
AmanziGeometry::Point MeshCache<MEM>::getCellCentroid(const Entity_ID c) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    // Need to check why this trips on GPU 
    //assert(data_.cell_geometry_cached);
    return view<MEM>(data_.cell_centroids)[c];
  } else if constexpr(AP == AccessPattern::FRAMEWORK) {
    static_assert(MEM == MemSpace_type::HOST);
    assert(framework_mesh_.get());
    return framework_mesh_->getCellCentroid(c);
  } else if constexpr(AP == AccessPattern::COMPUTE) {
    // here is where we would normally put something like
    // return MeshAlgorithms::computeCellCentroid(*this, c);
    // and implement the algorithm on device
  } else {
    if (data_.cell_geometry_cached) return getCellCentroid<AccessPattern::CACHE>(c);
    // return getCellCentroid<AccessPattern::COMPUTE>(c);
    return getCellCentroid<AccessPattern::FRAMEWORK>(c);
  }
}


// extent
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
double MeshCache<MEM>::getCellVolume(const Entity_ID c) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.cell_geometry_cached);
    return view<MEM>(data_.cell_volumes)[c];
  } else if constexpr(AP == AccessPattern::FRAMEWORK) {
    static_assert(MEM == MemSpace_type::HOST);
    assert(framework_mesh_.get());
    return framework_mesh_->getCellVolume(c);
  } else if constexpr(AP == AccessPattern::COMPUTE) {
    // here is where we would normally put something like
    // return MeshAlgorithms::computeCellVolume(*this, c);
    // and implement the algorithm on device
  } else {
    if (data_.cell_geometry_cached) return getCellVolume<AccessPattern::CACHE>(c);
    // return getCellVolume<AccessPattern::COMPUTE>(c);
    return getCellVolume<AccessPattern::FRAMEWORK>(c);
  }
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

  Kokkos::deep_copy(data_.cell_volumes.view_device(),data_.cell_volumes.view_host()); 
  Kokkos::deep_copy(data_.cell_centroids.view_device(),data_.cell_centroids.view_host()); 
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
  if constexpr(AP == AccessPattern::CACHE) {
    // Need to check why this trips on GPU
    //assert(data_.face_geometry_cached);
    return view<MEM>(data_.face_centroids)[f];
  } else if constexpr(AP == AccessPattern::FRAMEWORK) {
    static_assert(MEM == MemSpace_type::HOST);
    assert(framework_mesh_.get());
    return framework_mesh_->getFaceCentroid(f);
  } else if constexpr(AP == AccessPattern::COMPUTE) {
    // here is where we would normally put something like
    // return MeshAlgorithms::computeFaceCentroid(*this, f);
    // and implement the algorithm on device
  } else {
    if (data_.face_geometry_cached) return getFaceCentroid<AccessPattern::CACHE>(f);
    // return getFaceCentroid<AccessPattern::COMPUTE>(f);
    return getFaceCentroid<AccessPattern::FRAMEWORK>(f);
  }
}

template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
double MeshCache<MEM>::getFaceArea(const Entity_ID f) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.face_geometry_cached);
    return view<MEM>(data_.face_areas)[f];
  } else if constexpr(AP == AccessPattern::FRAMEWORK) {
    static_assert(MEM == MemSpace_type::HOST);
    assert(framework_mesh_.get());
    return framework_mesh_->getFaceArea(f);
  } else if constexpr(AP == AccessPattern::COMPUTE) {
    // here is where we would normally put something like
    // return MeshAlgorithms::computeFaceArea(*this, f);
    // and implement the algorithm on device
  } else {
    if (data_.face_geometry_cached) return getFaceArea<AccessPattern::CACHE>(f);
    // return getFaceArea<AccessPattern::COMPUTE>(f);
    return getFaceArea<AccessPattern::FRAMEWORK>(f);
  }
}


// Normal vector of a face
template<MemSpace_type MEM>
template<AccessPattern AP>
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point MeshCache<MEM>::getFaceNormal(const Entity_ID f) const
{
  if constexpr(AP == AccessPattern::CACHE) {
    assert(data_.face_geometry_cached);
    return data_.face_normals.get<MEM>(f,0);
  } else if constexpr(AP == AccessPattern::FRAMEWORK) {
    static_assert(MEM == MemSpace_type::HOST);
    assert(framework_mesh_.get());
    return framework_mesh_->getFaceNormal(f);
  } else if constexpr(AP == AccessPattern::COMPUTE) {
    // here is where we would normally put something like
    // return MeshAlgorithms::computeFaceNormal(*this, f);
    // and implement the algorithm on device
  } else {
    if (data_.face_geometry_cached) return getFaceNormal<AccessPattern::CACHE>(f);
    // return getFaceNormal<AccessPattern::COMPUTE>(f);
    return getFaceNormal<AccessPattern::FRAMEWORK>(f);
  }
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

  Kokkos::deep_copy(data_.face_areas.view_device(),data_.face_areas.view_host()); 
  Kokkos::deep_copy(data_.face_centroids.view_device(),data_.face_centroids.view_host()); 
  data_.face_geometry_cached = true;

  // cache normal directions -- make this a separate call?  Think about
  // granularity here.
  auto lambda2 = [&,this](const Entity_ID& f, Entity_Direction_List& dirs) {
    Entity_ID_List fcells; 
    this->framework_mesh_->getFaceCells(f, Parallel_type::ALL, fcells);
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


