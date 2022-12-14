/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors:
*/

#include <algorithm>

#include "errors.hh"
#include "MeshSimple.hh"

namespace Amanzi {
namespace AmanziMesh {

//---------------------------------------------------------
// Constructor
//---------------------------------------------------------
MeshSimple::MeshSimple(double x0, double y0, double z0,
                       double x1, double y1, double z1,
                       int nx, int ny, int nz)
  : MeshFramework(),
    nx_(nx), ny_(ny), nz_(nz),
    x0_(x0), x1_(x1),
    y0_(y0), y1_(y1),
    z0_(z0), z1_(z1)
{
  setSpaceDimension(3);
  setManifoldDimension(3);
  edges_requested_ = false;
  CreateCache_();
}


//---------------------------------------------------------
// Update
//---------------------------------------------------------
void MeshSimple::CreateCache_()
{
  // clear old cache
  coordinates_.clear();

  cell_to_face_.clear();
  face_to_node_.clear();
  edge_to_node_.clear();

  // build new cache
  num_cells_ = nx_ * ny_ * nz_;
  num_nodes_ = (nx_ + 1) * (ny_ + 1) * (nz_ + 1);
  num_faces_ = (nx_ + 1) * ny_ * nz_ + nx_ * (ny_ + 1) * nz_ + nx_ * ny_ * (nz_ + 1);
  num_edges_ = nx_ * (ny_ + 1) * (nz_ + 1) + (nx_ + 1) * ny_ * (nz_ + 1) + (nx_ + 1) * (ny_ + 1) * nz_;

  // -- node coordinates
  coordinates_.resize(3 * num_nodes_);

  double hx = (x1_ - x0_) / nx_;
  double hy = (y1_ - y0_) / ny_;
  double hz = (z1_ - z0_) / nz_;

  for (int iz = 0; iz <= nz_; iz++) {
    for (int iy = 0; iy <= ny_; iy++) {
      for (int ix = 0; ix <= nx_; ix++) {
        int istart = 3 * node_index_(ix, iy, iz);
        coordinates_[istart]     = x0_ + ix * hx;
        coordinates_[istart + 1] = y0_ + iy * hy;
        coordinates_[istart + 2] = z0_ + iz * hz;
      }
    }
  }

  // -- connectivity arrays
  cell_to_face_.resize(6 * num_cells_);
  cell_to_face_dirs_.resize(6 * num_cells_);
  face_to_cell_.assign(2 * num_faces_, -1); 

  face_to_node_.resize(4 * num_faces_);
  node_to_face_.resize(13 * num_nodes_);  // 1 extra for num faces

  if (edges_requested_) {
    face_to_edge_.resize(4 * num_faces_);
    face_to_edge_dirs_.resize(4 * num_faces_);
    edge_to_node_.resize(2 * num_edges_);
  }

  // loop over cells and initialize cell <-> face
  for (int iz = 0; iz < nz_; iz++) {
    for (int iy = 0; iy < ny_; iy++) {
      for (int ix = 0; ix < nx_; ix++) {
        int istart = 6 * cell_index_(ix,iy,iz);
        int jstart = 0;

        cell_to_face_[istart]     = xzface_index_(ix,  iy,  iz);
        cell_to_face_[istart + 1] = yzface_index_(ix+1,iy,  iz);
        cell_to_face_[istart + 2] = xzface_index_(ix,  iy+1,iz);
        cell_to_face_[istart + 3] = yzface_index_(ix,  iy,  iz);
        cell_to_face_[istart + 4] = xyface_index_(ix,  iy,  iz);
        cell_to_face_[istart + 5] = xyface_index_(ix,  iy,  iz+1);

        cell_to_face_dirs_[istart]     = 1;
        cell_to_face_dirs_[istart + 1] = 1;
        cell_to_face_dirs_[istart + 2] = -1;
        cell_to_face_dirs_[istart + 3] = -1;
        cell_to_face_dirs_[istart + 4] = -1;
        cell_to_face_dirs_[istart + 5] = 1;

        jstart = 2 * xzface_index_(ix, iy, iz);
        face_to_cell_[jstart + 1] = cell_index_(ix, iy, iz);

        jstart = 2 * yzface_index_(ix+1, iy, iz);
        face_to_cell_[jstart + 1] = cell_index_(ix, iy, iz);

        jstart = 2 * xzface_index_(ix, iy+1, iz);
        face_to_cell_[jstart] = cell_index_(ix, iy, iz);

        jstart = 2 * yzface_index_(ix, iy, iz);
        face_to_cell_[jstart] = cell_index_(ix, iy, iz);

        jstart = 2 * xyface_index_(ix, iy, iz);
        face_to_cell_[jstart] = cell_index_(ix, iy, iz);

        jstart = 2 * xyface_index_(ix, iy, iz+1);
        face_to_cell_[jstart + 1] = cell_index_(ix, iy, iz);
      }
    }
  }

  // loop over faces and initialize face <-> node
  // -- xy faces
  for (int iz = 0; iz <= nz_; iz++) {
    for (int iy = 0; iy < ny_; iy++) {
      for (int ix = 0; ix < nx_; ix++) {
        int istart = 4 * xyface_index_(ix, iy, iz);
        int jstart = 0;
        int nfaces = 0;

        face_to_node_[istart]     = node_index_(ix,  iy,  iz);
        face_to_node_[istart + 1] = node_index_(ix+1,iy,  iz);
        face_to_node_[istart + 2] = node_index_(ix+1,iy+1,iz);
        face_to_node_[istart + 3] = node_index_(ix,  iy+1,iz);

        jstart = 13 * node_index_(ix,iy,iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix+1, iy, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix+1, iy+1, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix, iy+1, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;
      }
    }
  }

  // -- xz faces
  for (int iz = 0; iz < nz_; iz++) {
    for (int iy = 0; iy <= ny_; iy++) {
      for (int ix=0; ix < nx_; ix++) {
        int istart = 4 * xzface_index_(ix, iy, iz);
        int jstart = 0;
        int nfaces = 0;

        face_to_node_[istart]     = node_index_(ix,  iy, iz);
        face_to_node_[istart + 1] = node_index_(ix+1,iy, iz);
        face_to_node_[istart + 2] = node_index_(ix+1,iy, iz+1);
        face_to_node_[istart + 3] = node_index_(ix,  iy, iz+1);

        jstart = 13 * node_index_(ix, iy, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix+1, iy, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix+1, iy, iz+1);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix, iy, iz+1);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;
      }
    }
  }

  // -- yz faces
  for (int iz = 0; iz < nz_; iz++) {
    for (int iy = 0; iy < ny_; iy++) {
      for (int ix = 0; ix <= nx_; ix++) {
        int istart = 4 * yzface_index_(ix, iy, iz);
        int jstart = 0;
        int nfaces = 0;

        face_to_node_[istart]     = node_index_(ix, iy,  iz);
        face_to_node_[istart + 1] = node_index_(ix, iy+1,iz);
        face_to_node_[istart + 2] = node_index_(ix, iy+1,iz+1);
        face_to_node_[istart + 3] = node_index_(ix, iy,  iz+1);

        jstart = 13 * node_index_(ix, iy, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix, iy+1, iz);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix, iy+1, iz+1);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;

        jstart = 13 * node_index_(ix, iy, iz+1);
        nfaces = node_to_face_[jstart];
        node_to_face_[jstart + 1 + nfaces] = xyface_index_(ix, iy, iz);
        (node_to_face_[jstart])++;
      }
    }
  }

  if (edges_requested_) { 
    // loop over faces and initialize face -> edge
    // -- xy faces
    for (int iz = 0; iz <= nz_; iz++) {
      for (int iy = 0; iy < ny_; iy++) {
        for (int ix = 0; ix < nx_; ix++) {
          int istart = 4 * xyface_index_(ix, iy, iz);

          face_to_edge_[istart]     = xedge_index_(ix,  iy,  iz);
          face_to_edge_[istart + 1] = yedge_index_(ix+1,iy,  iz);
          face_to_edge_[istart + 2] = xedge_index_(ix,  iy+1,iz);
          face_to_edge_[istart + 3] = yedge_index_(ix,  iy,  iz);

          face_to_edge_dirs_[istart]     = 1;
          face_to_edge_dirs_[istart + 1] = 1;
          face_to_edge_dirs_[istart + 2] = -1;
          face_to_edge_dirs_[istart + 3] = -1;
        }
      }
    }

    // -- xz faces
    for (int iz = 0; iz < nz_; iz++) {
      for (int iy = 0; iy <= ny_; iy++) {
        for (int ix=0; ix < nx_; ix++) {
          int istart = 4 * xzface_index_(ix, iy, iz);

          face_to_edge_[istart]     = xedge_index_(ix,  iy, iz);
          face_to_edge_[istart + 1] = zedge_index_(ix+1,iy, iz);
          face_to_edge_[istart + 2] = xedge_index_(ix,  iy, iz+1);
          face_to_edge_[istart + 3] = zedge_index_(ix,  iy, iz);

          face_to_edge_dirs_[istart]     = 1;
          face_to_edge_dirs_[istart + 1] = 1;
          face_to_edge_dirs_[istart + 2] = -1;
          face_to_edge_dirs_[istart + 3] = -1;
        }
      }
    }

    // -- yz faces
    for (int iz = 0; iz < nz_; iz++) {
      for (int iy = 0; iy < ny_; iy++) {
        for (int ix = 0; ix <= nx_; ix++) {
          int istart = 4 * yzface_index_(ix, iy, iz);

          face_to_edge_[istart]     = yedge_index_(ix, iy,  iz);
          face_to_edge_[istart + 1] = zedge_index_(ix, iy+1,iz);
          face_to_edge_[istart + 2] = yedge_index_(ix, iy,  iz+1);
          face_to_edge_[istart + 3] = zedge_index_(ix, iy,  iz);

          face_to_edge_dirs_[istart]     = 1;
          face_to_edge_dirs_[istart + 1] = 1;
          face_to_edge_dirs_[istart + 2] = -1;
          face_to_edge_dirs_[istart + 3] = -1;
        }
      }
    }

    // loop over edges and initialize edge -> nodes
    // -- x edges
    for (int iz = 0; iz <= nz_; iz++) {
      for (int iy = 0; iy <= ny_; iy++) {
        for (int ix = 0; ix < nx_; ix++) {
          int istart = 2 * xedge_index_(ix, iy, iz);

          edge_to_node_[istart]     = node_index_(ix,  iy,  iz);
          edge_to_node_[istart + 1] = node_index_(ix+1,iy,  iz);
        }
      }
    }

    // -- y edges
    for (int iz = 0; iz <= nz_; iz++) {
      for (int iy = 0; iy < ny_; iy++) {
        for (int ix = 0; ix <= nx_; ix++) {
          int istart = 2 * yedge_index_(ix, iy, iz);

          edge_to_node_[istart]     = node_index_(ix, iy,   iz);
          edge_to_node_[istart + 1] = node_index_(ix, iy+1, iz);
        }
      }
    }

    // -- z edges
    for (int iz = 0; iz < nz_; iz++) {
      for (int iy = 0; iy <= ny_; iy++) {
        for (int ix = 0; ix <= nx_; ix++) {
          int istart = 2 * zedge_index_(ix, iy, iz);

          edge_to_node_[istart]     = node_index_(ix, iy, iz);
          edge_to_node_[istart + 1] = node_index_(ix, iy, iz+1);
        }
      }
    }
  }
}


//---------------------------------------------------------
// TBW
//---------------------------------------------------------
std::size_t MeshSimple::getNumEntities(AmanziMesh::Entity_kind kind,
                                       AmanziMesh::Parallel_type ptype) const
{
  switch (kind) {
    case Entity_kind::FACE:
      return (ptype != AmanziMesh::Parallel_type::GHOST) ? num_faces_ : 0;
      break;
    case Entity_kind::BOUNDARY_FACE:
    //  Same problem as the constructor, cannot be -1 for std::size_t
      return 0;
      break;
    case Entity_kind::NODE:
      return (ptype != AmanziMesh::Parallel_type::GHOST) ? num_nodes_ : 0;
      break;
    case Entity_kind::BOUNDARY_NODE:
      return 0;
      break;
    case Entity_kind::CELL:
      return (ptype != AmanziMesh::Parallel_type::GHOST) ? num_cells_ : 0;
      break;
    default:
      throw std::exception();
      break;
  }
}


//---------------------------------------------------------
// Connectivity: cell -> faces
//---------------------------------------------------------
void MeshSimple::getCellFacesAndDirs(const AmanziMesh::Entity_ID cellid,
        Entity_ID_List& faceids,
        Entity_Direction_List *cfacedirs) const
{
  unsigned int offset = (unsigned int) 6*cellid;

  faceids.clear();
  auto it = cell_to_face_.begin() + offset;
  faceids.insert(faceids.end(), it, it + 6);

  if (cfacedirs) {
    cfacedirs->clear();
    auto jt = cell_to_face_dirs_.begin() + offset;
    cfacedirs->insert(cfacedirs->begin(), jt, jt + 6);
  }
}


//---------------------------------------------------------
// Connectivity: face -> nodes
//---------------------------------------------------------
void MeshSimple::getFaceNodes(AmanziMesh::Entity_ID face,
        AmanziMesh::Entity_ID_List& nodeids) const
{
  unsigned int offset = (unsigned int) 4*face;
  nodeids.clear();
  for (int i = 0; i < 4; i++) {
    nodeids.push_back(*(face_to_node_.begin()+offset));
    offset++;
  }
}


//---------------------------------------------------------
// Connectivity: face -> edges
//---------------------------------------------------------
void MeshSimple::getFaceEdgesAndDirs(const Entity_ID faceid,
        Entity_ID_List& edgeids,
        Entity_Direction_List *fedgedirs) const
{
  unsigned int offset = (unsigned int) 4*faceid;

  edgeids.clear();
  auto it = face_to_edge_.begin() + offset;
  edgeids.insert(edgeids.begin(), it, it + 4);

  if (fedgedirs) {
    fedgedirs->clear();
    auto jt = face_to_edge_dirs_.begin();
    fedgedirs->insert(fedgedirs->begin(), jt, jt + 4);
  }
}


//---------------------------------------------------------
// Cooordinates of a node
//---------------------------------------------------------
AmanziGeometry::Point
MeshSimple::getNodeCoordinate(const AmanziMesh::Entity_ID local_node_id) const
{
  unsigned int offset = (unsigned int) 3*local_node_id;
  AmanziGeometry::Point ncoord;
  ncoord.set(3, &(coordinates_[offset]));
  return ncoord;
}



//---------------------------------------------------------
// Faces of type 'ptype' connected to a node
//---------------------------------------------------------
void MeshSimple::getNodeFaces(const AmanziMesh::Entity_ID nodeid,
        const AmanziMesh::Parallel_type ptype,
        AmanziMesh::Entity_ID_List& faceids) const
{
  unsigned int offset = (unsigned int) 13*nodeid;
  unsigned int nfaces = node_to_face_[offset];

  faceids.clear();

  for (int i = 0; i < nfaces; i++) 
    faceids.push_back(node_to_face_[offset+i+1]);
}


//---------------------------------------------------------
// Cells connected to a face
//---------------------------------------------------------
void MeshSimple::getFaceCells(const AmanziMesh::Entity_ID faceid,
        const AmanziMesh::Parallel_type ptype,
        AmanziMesh::Entity_ID_List& cellids) const
{
  unsigned int offset = (unsigned int) 2*faceid;

  cellids.clear();

  if (face_to_cell_[offset] != -1)
    cellids.push_back(face_to_cell_[offset]);
  if (face_to_cell_[offset+1] != -1)
    cellids.push_back(face_to_cell_[offset+1]);
}


}  // namespace AmanziMesh
}  // namespace Amanzi

