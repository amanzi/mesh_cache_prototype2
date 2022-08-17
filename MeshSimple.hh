/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

//! A very simple structured 3D mesh, mostly for testing and serial applications.
/*!

Note this implements the bare minimum functionality, and is pretty
inefficiently implemented.

*/

#ifndef AMANZI_MESH_SIMPLE_HH_
#define AMANZI_MESH_SIMPLE_HH_

#include <memory>
#include <vector>

#include "MeshFramework.hh"

namespace Amanzi {
namespace AmanziMesh {

class MeshSimple : public MeshFramework {
 public:

  MeshSimple(double x0, double y0, double z0,
              double x1, double y1, double z1,
              int nx, int ny, int nz);
  virtual ~MeshSimple() = default;

  bool hasEdges() const override { return edges_requested_; }

  // Number of entities of any kind (cell, face, node) and in a
  // particular category (OWNED, GHOST, ALL)
  virtual std::size_t getNumEntities(const Entity_kind kind,
                            const Parallel_type ptype) const override;


  // Node coordinates - 3 in 3D and 2 in 2D
  virtual AmanziGeometry::Point getNodeCoordinate(const Entity_ID nodeid) const override;

  virtual void getCellFacesAndDirs(
    const Entity_ID c,
    Entity_ID_List& faces,
    Entity_Direction_List * const dirs) const override;

  virtual void getFaceEdgesAndDirs(const Entity_ID f,
          Entity_ID_List& edges,
          Entity_Direction_List * const dirs=nullptr) const;

  virtual void getFaceNodes(const Entity_ID f, Entity_ID_List& nodes) const override;


  virtual void getFaceCells(const Entity_ID f,
                          const Parallel_type ptype,
                          Entity_ID_List& cells) const override;

  virtual void getNodeFaces(const Entity_ID nodeid,
                            const Parallel_type ptype,
                            Entity_ID_List& faceids) const override;

 private:
  void CreateCache_();

private:
  bool edges_requested_;
  std::vector<double> coordinates_;

  unsigned int node_index_(int i, int j, int k) const;
  unsigned int cell_index_(int i, int j, int k) const;

  unsigned int xyface_index_(int i, int j, int k) const;
  unsigned int yzface_index_(int i, int j, int k) const;
  unsigned int xzface_index_(int i, int j, int k) const;

  unsigned int xedge_index_(int i, int j, int k) const;
  unsigned int yedge_index_(int i, int j, int k) const;
  unsigned int zedge_index_(int i, int j, int k) const;

  int nx_, ny_, nz_;  // number of cells in the three coordinate directions
  double x0_, x1_, y0_, y1_, z0_, z1_;  // coordinates of lower left front and upper right back of brick

  int num_cells_, num_faces_, num_edges_, num_nodes_;

  // mesh connectivity arrays
  std::vector<Entity_ID> cell_to_face_;
  std::vector<Entity_ID> face_to_edge_;
  std::vector<Entity_ID> face_to_node_;
  std::vector<Entity_ID> edge_to_node_;

  std::vector<Entity_ID> node_to_face_;
  std::vector<Entity_ID> node_to_edge_;
  std::vector<Entity_ID> edge_to_face_;
  std::vector<Entity_ID> face_to_cell_;

  // orientation arrays
  std::vector<int> cell_to_face_dirs_;
  std::vector<int> face_to_edge_dirs_;
};


// -------------------------
// Template & inline members
// ------------------------
inline
unsigned int MeshSimple::node_index_(int i, int j, int k) const {
  return i + j * (nx_ + 1) + k * (nx_ + 1) * (ny_ + 1);
}

inline
unsigned int MeshSimple::cell_index_(int i, int j, int k) const {
  return i + j * nx_ + k * nx_ * ny_;
}

inline
unsigned int MeshSimple::xyface_index_(int i, int j, int k) const {
  return i + j * nx_ + k * nx_ * ny_;
}

inline
unsigned int MeshSimple::xzface_index_(int i, int j, int k) const {
  return i + j * nx_ + k * nx_ * (ny_ + 1) + xyface_index_(0, 0, nz_ + 1);
}

inline
unsigned int MeshSimple::yzface_index_(int i, int j, int k) const {
  return i + j * (nx_ + 1) + k * (nx_ + 1) * ny_ + xzface_index_(0, 0, nz_);
}

inline
unsigned int MeshSimple::xedge_index_(int i, int j, int k) const {
  return i + j * nx_ + k * nx_ * (ny_ + 1);
}

inline
unsigned int MeshSimple::yedge_index_(int i, int j, int k) const {
  return i + j * (nx_ + 1) + k * (nx_ + 1) * ny_ + xedge_index_(0, 0, nz_ + 1);
}

inline
unsigned int MeshSimple::zedge_index_(int i, int j, int k) const {
  return i + j * (nx_ + 1) + k * (nx_ + 1) * (ny_ + 1) + yedge_index_(0, 0, nz_ + 1);
}

}  // namespace AmanziMesh
}  // namespace Amanzi

#endif

