/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella, others
*/

// Default imlementations of MeshFramework.

#include "errors.hh"
#include "exceptions.hh"
#include "MeshFramework.hh"
#include "Mesh_Helpers_impl.hh"

namespace Amanzi {
namespace AmanziMesh {

MeshFramework::MeshFramework()
// These cannot be -1 or cannot be std::size_t
  : space_dim_(0),
    manifold_dim_(0)
{
  algorithms_ = std::make_shared<MeshFrameworkAlgorithms>();
}


Parallel_type
MeshFramework::getEntityPtype(const Entity_kind kind, const Entity_ID entid) const
{
  return entid >= getNumEntities(kind, Parallel_type::OWNED) ? Parallel_type::OWNED : Parallel_type::GHOST;
}


Cell_type
MeshFramework::getCellType(const Entity_ID c) const
{
  Entity_ID_List faces;
  getCellFaces(c, faces);
  return getCellType_(c, faces);
}

Cell_type
MeshFramework::getCellType_(const Entity_ID c,
                           const Entity_ID_List& faces) const
{
  if (getManifoldDimension() == 2) {
    switch (faces.size()) {
    case 3:
      return Cell_type::TRI;
      break;
    case 4:
      return Cell_type::QUAD;
      break;
    default:
      return Cell_type::POLYGON;
    }
  } else if (getManifoldDimension() == 3) {
    int nquads = 0;
    for (const auto& f : faces) {
      Entity_ID_List fnodes;
      getFaceNodes(f, fnodes);
      if (fnodes.size() == 4) nquads++;
    }

    switch (faces.size()) {
      case 4:
        if (nquads == 0) return Cell_type::TET;
        else return Cell_type::POLYHED;
        break;
      case 5:
        if (nquads == 1) return Cell_type::PYRAMID;
        else if (nquads == 3) return Cell_type::PRISM;
        else return Cell_type::POLYHED;
        break;
      case 6:
        if (nquads == 6) return Cell_type::HEX;
        else return Cell_type::POLYHED;
        break;
      default:
        return Cell_type::POLYHED;
    }
  } else {
    Errors::Message msg;
    msg << "Mesh of manifold_dimension = " << getManifoldDimension() << " not supported";
    Exceptions::amanzi_throw(msg);
  }
  return Cell_type::UNKNOWN;
}



Point_List
MeshFramework::getCellCoordinates(const Entity_ID c) const
{
  return MeshAlgorithms::getCellCoordinates(*this, c);
}

Point_List
MeshFramework::getFaceCoordinates(const Entity_ID f) const
{
  return MeshAlgorithms::getFaceCoordinates(*this, f);
}


//
// Cell geometry
//
double
MeshFramework::getCellVolume(const Entity_ID c) const
{
  return computeCellGeometry(c).first;
}

AmanziGeometry::Point
MeshFramework::getCellCentroid(const Entity_ID c) const
{
  return computeCellGeometry(c).second;
}


//
// face geometry
//
double
MeshFramework::getFaceArea(const Entity_ID f) const
{
  return std::get<0>(computeFaceGeometry(f));
}

AmanziGeometry::Point
MeshFramework::getFaceCentroid(const Entity_ID f) const
{
  return std::get<1>(computeFaceGeometry(f));
}

AmanziGeometry::Point
MeshFramework::getFaceNormal(const Entity_ID f, const Entity_ID c, int * const orientation) const
{
  auto geom = computeFaceGeometry(f);

  Entity_ID_List fcells;
  getFaceCells(f, Parallel_type::ALL, fcells);
  if (orientation) *orientation = 0;
  Entity_ID cc = (c < 0) ? fcells[0] : c;

  int i = std::find(fcells.begin(), fcells.end(), cc) - fcells.begin();
  AmanziGeometry::Point normal = std::get<2>(geom)[i];

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



//
// bisectors
//
void
MeshFramework::getCellFacesAndBisectors(const Entity_ID cellid,
        Entity_ID_List& faceids, Point_List * const bisectors) const
{
  getCellFaces(cellid, faceids);
  if (bisectors)
    *bisectors = MeshAlgorithms::computeBisectors(*this, cellid, faceids);
}


void
MeshFramework::getCellNodes(const Entity_ID c, Entity_ID_List& nodes) const
{
  nodes = MeshAlgorithms::computeCellNodes(*this, c);
}

void
MeshFramework::getNodeCells(const Entity_ID n, const Parallel_type ptype, Entity_ID_List& cells) const
{
  cells.resize(0);
  Entity_ID_List faces, fcells;
  getNodeFaces(n, Parallel_type::ALL, faces);
  for (const auto& f : faces) {
    getFaceCells(f, ptype, fcells);
    for (const auto& c : fcells) {
      if (std::find(cells.begin(), cells.end(), c) == cells.end()) {
        cells.emplace_back(c);
      }
    }
  }
}


void
MeshFramework::throwNotImplemented_(const std::string& methodname) const
{
  Errors::Message msg;
  msg << "MeshFramework does not yet implement " << methodname;
  Exceptions::amanzi_throw(msg);
}


std::pair<double, AmanziGeometry::Point>
MeshFrameworkAlgorithms::computeCellGeometry(const MeshFramework& mesh, const Entity_ID c) const
{
  return MeshAlgorithms::computeCellGeometry(mesh, c);
}
// std::pair<double, AmanziGeometry::Point>
// MeshFrameworkAlgorithms::computeCellGeometry(const Mesh& mesh, const Entity_ID c) const
// {
//   return MeshAlgorithms::computeCellGeometry(mesh, c);
// }

std::tuple<double, AmanziGeometry::Point, Point_List>
MeshFrameworkAlgorithms::computeFaceGeometry(const MeshFramework& mesh, const Entity_ID f) const
{
  return MeshAlgorithms::computeFaceGeometry(mesh, f);
}
// std::tuple<double, AmanziGeometry::Point, Point_List>
// MeshFrameworkAlgorithms::computeFaceGeometry(const Mesh& mesh, const Entity_ID f) const
// {
//   return MeshAlgorithms::computeFaceGeometry(mesh, f);
// }

// Point_List
// MeshFrameworkAlgorithms::computeBisectors(const Mesh& mesh, const Entity_ID c, const Entity_ID_List& faces) const
// {
//   return MeshAlgorithms::computeBisectors(mesh, c, faces);
// }




} // namespace AmanziMesh
} // namespace Amanzi
