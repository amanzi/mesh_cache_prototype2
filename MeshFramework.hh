/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella, others
*/

//! The interface for meshes provided by external frameworks.
/*!

Developer note:

Most of this interface is here for testing.  Very little is likely to be used
in the final code, because largely the interface will be used to generate a
MeshCache object.  The MeshCache will then provide the full interface using
fast access, non-virtual methods.

A new Framework really must supply only a handful of methods, but may choose to
provide more, as long as they are consistent.

Note that the framework is split into two classes, MeshFramework and
MeshFrameworkAlgorithms, both of which must exist.  For many, the algorithms
will be the default class.  But some "special" frameworks may implement special
algorithms.  If they do so, the MeshCache object will need the algorithms even
if the Framework itself is deleted (hence the split).

*/

#pragma once

#include <string>

#include "MeshDefs.hh"
#include "Mesh_Helpers_decl.hh"

namespace Amanzi {

class VerboseObject;
namespace AmanziGeometry {
class Point;
}

namespace AmanziMesh {

class MeshFramework;

//
// This class provides default, virtual algorithms for computing geometric
// quantities given nodal coordinates and topological information.
//
// Split into two classes to aid in deletion of the MeshFramework class, while
// keeping the MeshFrameworkAlgorithms class around for use by the Cache.
//
struct MeshFrameworkAlgorithms {
  // lumped things for more efficient calculation
  virtual std::pair<double, AmanziGeometry::Point>
  computeCellGeometry(const MeshFramework& mesh, const Entity_ID c) const;

  virtual std::tuple<double, AmanziGeometry::Point, Point_List>
  computeFaceGeometry(const MeshFramework& mesh, const Entity_ID f) const;

};


//
// The framework class itself provides setters/getters/attributes, all
// topology, and coordinates.
//
class MeshFramework  {
 protected:
  MeshFramework();

 public:
  virtual ~MeshFramework() = default;


  // space dimension describes the dimension of coordinates in space
  std::size_t getSpaceDimension() const { return space_dim_; }
  void setSpaceDimension(unsigned int dim) { space_dim_ = dim; }

  // manifold dimension describes the dimensionality of the corresponding R^n
  // manifold onto which this mesh can be projected.
  std::size_t getManifoldDimension() const { return manifold_dim_; }
  void setManifoldDimension(const unsigned int dim) { manifold_dim_ = dim; }

  // Helper class that provides geometric algorithms.  Sometimes it is useful
  // to keep the algorithms object even if this class is deleted.
  std::shared_ptr<const MeshFrameworkAlgorithms> getAlgorithms() const { return algorithms_; }
  void setAlgorithms(const std::shared_ptr<const MeshFrameworkAlgorithms>& algorithms ) {
    algorithms_ = algorithms; }

  // Some meshes have edges
  //
  // DEVELOPER NOTE: frameworks that do not implement edges need not provide
  // any edge method -- defaults here all throw errors.
  virtual bool hasEdges() const { return false; }

  // DEVELOPER NOTE: frameworks that do not implement nodes DO need to provide
  // ALL node methods to have them throw errors.  The default here assumes
  // nodes exist.
  virtual bool hasNodes() const { return true; }

  // Some meshes may natively order in the ExodusII ordering
  virtual bool isOrdered() const { return false; }

  // Some meshes can be deformed.
  virtual bool isDeformable() const { return false; }

  // Some meshes are logical meshes and do not have coordinate info.
  virtual bool isLogical() const { return false; }

  // ----------------
  // Entity meta-data
  // ----------------
  virtual std::size_t getNumEntities(const Entity_kind kind, const Parallel_type ptype) const = 0;

  // Parallel type of the entity.
  //
  // DEVELOPER NOTE: meshes which order entities by OWNED, GHOSTED need not
  // implement this method.
  virtual Parallel_type getEntityPtype(const Entity_kind kind, const Entity_ID entid) const;

  // Cell types: UNKNOWN, TRI, QUAD, etc. See MeshDefs.hh
  //
  // DEVELOPER NOTE: Default implementation guesses based on topology.
  virtual Cell_type getCellType(const Entity_ID cellid) const;


  //---------------------
  // Geometry
  //---------------------
  // locations
  virtual AmanziGeometry::Point getNodeCoordinate(const Entity_ID node) const = 0;
  virtual AmanziGeometry::Point getCellCentroid(const Entity_ID c) const;
  virtual AmanziGeometry::Point getFaceCentroid(const Entity_ID f) const;

  virtual Point_List getCellCoordinates(const Entity_ID c) const;
  virtual Point_List getFaceCoordinates(const Entity_ID f) const;

  // extent
  virtual double getCellVolume(const Entity_ID c) const;
  virtual double getFaceArea(const Entity_ID f) const;

  // lumped things for more efficient calculation
  std::pair<double, AmanziGeometry::Point>
  computeCellGeometry(const Entity_ID c) const {
    return algorithms_->computeCellGeometry(*this, c);
  }

  std::tuple<double, AmanziGeometry::Point, Point_List>
  computeFaceGeometry(const Entity_ID f) const {
    return algorithms_->computeFaceGeometry(*this, f);
  }

  // Normal vector of a face
  //
  // The vector is normalized and then weighted by the area of the face.
  //
  // Orientation is the natural orientation, e.g. that it points from cell 0 to
  // cell 1 with respect to face_cell adjacency information.
  inline
  AmanziGeometry::Point getFaceNormal(const Entity_ID f) const {
    return getFaceNormal(f, -1, nullptr);
  }

  // Normal vector of a face, outward with respect to a cell.
  //
  // The vector is normalized and then weighted by the area of the face.
  //
  // Orientation, if provided, returns the direction of
  // the natural normal (1 if outward, -1 if inward).
  virtual AmanziGeometry::Point getFaceNormal(const Entity_ID f,
          const Entity_ID c, int * const orientation=nullptr) const;


  //---------------------
  // Downward adjacencies
  //---------------------
  // Get faces of a cell
  //
  // On a distributed mesh, this will return all the faces of the
  // cell, OWNED or GHOST. If the framework supports it, the faces will be
  // returned in a standard order according to Exodus II convention
  // for standard cells; in all other situations (not supported or
  // non-standard cells), the list of faces will be in arbitrary order
  //
  // EXTENSIONS: MSTK FRAMEWORK: by the way the parallel partitioning,
  // send-receive protocols and mesh query operators are designed, a side 
  // effect of this is that master and ghost entities will have the same
  // hierarchical topology.
  void getCellFaces(const Entity_ID c,
                    Entity_ID_List& faces) const {
    getCellFacesAndDirs(c, faces, nullptr);
  }

  void getCellFaceDirs(const Entity_ID c,
                       Entity_Direction_List& dirs) const {
    Entity_ID_List faces;
    getCellFacesAndDirs(c, faces, &dirs);
  }

  // Get faces of a cell and directions in which the cell uses the face
  //
  // On a distributed mesh, this will return all the faces of the
  // cell, OWNED or GHOST. If the framework supports it, the faces will be
  // returned in a standard order according to Exodus II convention
  // for standard cells.
  //
  // In 3D, direction is 1 if face normal points out of cell
  // and -1 if face normal points into cell
  // In 2D, direction is 1 if face/edge is defined in the same
  // direction as the cell polygon, and -1 otherwise
  virtual void getCellFacesAndDirs(
    const Entity_ID c,
    Entity_ID_List& faces,
    Entity_Direction_List * const dirs) const = 0;

  // Get the bisectors, i.e. vectors from cell centroid to face centroids.
  virtual void getCellFacesAndBisectors(
          const Entity_ID cellid,
          Entity_ID_List& faceids,
          Point_List * const bisectors) const;

  virtual void getCellNodes(const Entity_ID c, Entity_ID_List& nodes) const;


  // Get nodes of face
  //
  // In 3D, the nodes of the face are returned in ccw order consistent
  // with the face normal.
  virtual void getFaceNodes(const Entity_ID f, Entity_ID_List& nodes) const = 0;


  //-------------------
  // Upward adjacencies
  //-------------------
  // The cells are returned in no particular order. Also, the order of cells
  // is not guaranteed to be the same for corresponding faces on different
  // processors
  virtual void getFaceCells(const Entity_ID f,
                            const Parallel_type ptype,
                            Entity_ID_List& cells) const = 0;

  // Cells of type 'ptype' connected to a node
  // NOTE: The order of cells is not guaranteed to be the same for
  // corresponding nodes on different processors
  virtual void getNodeCells(const Entity_ID nodeid,
                            const Parallel_type ptype,
                            Entity_ID_List& cellids) const;

  // Faces of type parallel 'ptype' connected to a node
  // NOTE: The order of faces is not guarnateed to be the same for
  // corresponding nodes on different processors
  virtual void getNodeFaces(const Entity_ID nodeid,
                            const Parallel_type ptype,
                            Entity_ID_List& faceids) const = 0;


protected:
  void throwNotImplemented_(const std::string& fname) const;
  Cell_type getCellType_(const Entity_ID c, const Entity_ID_List& faces) const;

 protected:
  std::shared_ptr<const MeshFrameworkAlgorithms> algorithms_;
  std::size_t space_dim_;
  std::size_t manifold_dim_;
};




}  // namespace AmanziMesh
}  // namespace Amanzi

