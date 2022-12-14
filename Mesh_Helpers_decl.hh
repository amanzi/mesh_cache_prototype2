/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella
      Ethan Coon (coonet@ornl.gov)
*/
// A collection of helper functions used to compute basic geometric operations on Mesh objects.
/*
  Note, these are templated on Mesh_type because they may be used by either
  MeshFramework objects or MeshCache objects.

  These provide default calculations of fundamental topological and geometric
  concepts.  While they can be called by users of the Mesh library, typically
  they are used _inside_ of the library, and should not be called by users.

  For the most part, these are used by member functions of the MeshFramework
  and MeshCache classes to provide default implementations.  They are
  implemented as external, nonmember functions because, 1, they use the common
  API of multiple classes, and so are useful for multiple classes
  (MeshFramework and MeshCache), and 2, they use the API, and need no "private"
  data, so they improve encapsulation of the class.

  Note that, if these become commonly used by users of the mesh library, they
  should get moved into Mesh_Algorithms, which provides functions that are
  commonly used by clients _outside_ of this library.
*/


#pragma once

#include "Point.hh"
#include "MeshDefs.hh"

namespace Amanzi {
namespace AmanziMesh {
namespace MeshAlgorithms {

//
// topology algorithms
//
template<class Mesh_type>
Cell_type getCellType(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
int getFaceDirectionInCell(const Mesh_type& mesh, const Entity_ID f, const Entity_ID c);

template<class Mesh_type>
Entity_ID_List
computeCellEdges(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
Entity_ID_List
computeCellNodes(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
Entity_ID_List
computeNodeCells(const Mesh_type& mesh, const Entity_ID n);


//
// Geometry algorithms
//
template<class Mesh_type>
std::pair<double,AmanziGeometry::Point>
computeCellGeometry(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
std::tuple<double,AmanziGeometry::Point,Point_List>
computeFaceGeometry(const Mesh_type& mesh, const Entity_ID f);

template<class Mesh_type>
std::pair<AmanziGeometry::Point,AmanziGeometry::Point>
computeEdgeGeometry(const Mesh_type& mesh, const Entity_ID e);

template<class Mesh_type>
Point_List
computeBisectors(const Mesh_type& mesh, const Entity_ID c,
                 const Entity_ID_List& faces);

template<class Mesh_type>
void debugCell(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
Point_List getEdgeCoordinates(const Mesh_type& mesh, const Entity_ID e);

template<class Mesh_type>
Point_List getFaceCoordinates(const Mesh_type& mesh, const Entity_ID f);

template<class Mesh_type>
Point_List getCellCoordinates(const Mesh_type& mesh, const Entity_ID c);

template<class Mesh_type>
std::size_t getMaxCellNumNodes(const Mesh_type& mesh);

template<class Mesh_type>
std::size_t getMaxCellNumFaces(const Mesh_type& mesh);

template<class Mesh_type>
std::size_t getMaxCellNumEdges(const Mesh_type& mesh);

template<class Mesh_type>
Entity_ID_List getCellFaceAdjacentCells(const Mesh_type& mesh,
        Entity_ID c, Parallel_type ptype);

template<class Mesh_type> 
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point getFaceCentroid(const Mesh_type& mesh, const Entity_ID f);

template<class Mesh_type> 
AmanziGeometry::Point getCellCentroid(const Mesh_type& mesh, const Entity_ID c);

} // namespace MeshAlgorithms
} // namespace AmanziMesh
} // namspace Amanzi
