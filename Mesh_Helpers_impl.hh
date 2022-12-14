/*
  Copyright 2010-201x held jointly by LANL, ORNL, LBNL, and PNNL.
  Amanzi is released under the three-clause BSD License.
  The terms of use and "as is" disclaimer for this license are
  provided in the top-level COPYRIGHT file.

  Authors: Rao Garimella
      Ethan Coon (coonet@ornl.gov)
*/

#pragma once

#include "Point.hh"
#include "Geometry.hh"
#include "MeshDefs.hh"

namespace Amanzi {
namespace AmanziMesh {
namespace MeshAlgorithms {


template<class Mesh_type>
Cell_type getCellType(const Mesh_type& mesh, const Entity_ID c)
{
  auto faces = mesh.getCellFaces(c);
  if (mesh.getManifoldDimension() == 2) {
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
  } else if (mesh.getManifoldDimension() == 3) {
    int nquads = 0;
    for (const auto& f : faces) {
      Entity_ID_List fnodes;
      mesh.getFaceNodes(f, fnodes);
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
    msg << "Mesh of manifold_dimension = " << mesh.getManifoldDimension() << " not supported";
    Exceptions::amanzi_throw(msg);
  }
  return Cell_type::UNKNOWN;
}



template<class Mesh_type>
int getFaceDirectionInCell(const Mesh_type& mesh, const Entity_ID f, const Entity_ID c)
{
  int dir = 0;
  Entity_ID_List cfaces;
  Entity_Direction_List dirs;
  mesh.getCellFacesAndDirs(c, cfaces, &dirs);
  for (int j=0; j!=cfaces.size(); ++j) {
    if (cfaces[j] == f) {
      dir = dirs[j];
      break;
    }
  }
  return dir;
}


template<class Mesh_type>
std::vector<int>
mapFaceToCellEdges(const Mesh_type& mesh, const Entity_ID f, const Entity_ID c)
{
  auto fedges = mesh.getFaceEdges(f);
  auto cedges = mesh.getCellEdges(c);
  std::vector<int> map(fedges.size());
  for (int i=0; i!=fedges.size(); ++i)
    map[i] = std::find(cedges.begin(), cedges.end(), fedges[i]) - cedges.begin();
  return map;
}


template<class Mesh_type>
Entity_ID_List
computeCellEdges(const Mesh_type& mesh, const Entity_ID c)
{
  Entity_ID_List edges;
  Entity_ID_List faces, fedges;
  mesh.getCellFaces(c, faces);
  for (const auto& f : faces) {
    mesh.getFaceEdges(f, fedges);
    for (const auto& e : fedges) {
      if (std::find(edges.begin(), edges.end(), e) == edges.end()) {
        edges.emplace_back(e);
      }
    }
  }
  return edges;
}

template<class Mesh_type>
Entity_ID_List
computeCellNodes(const Mesh_type& mesh, const Entity_ID c)
{
  Entity_ID_List nodes;
  Entity_ID_List faces, fnodes;
  mesh.getCellFaces(c, faces);
  for (const auto& f : faces) {
    mesh.getFaceNodes(f, fnodes);
    for (const auto& n : fnodes) {
      if (std::find(nodes.begin(), nodes.end(), n) == nodes.end()) {
        nodes.emplace_back(n);
      }
    }
  }
  return nodes;
}

template<class Mesh_type>
std::pair<double, AmanziGeometry::Point>
computeCellGeometry(const Mesh_type& mesh, const Entity_ID c)
{
  if (mesh.getManifoldDimension() == 2) {
    auto ccoords = mesh.getCellCoordinates(c);
    auto vol_cent = std::make_pair((double)0,
            AmanziGeometry::Point(mesh.getSpaceDimension()));
    AmanziGeometry::Point normal(mesh.getSpaceDimension());
    AmanziGeometry::polygon_get_area_centroid_normal(ccoords,
            &vol_cent.first, &vol_cent.second, &normal);
    return vol_cent;
  } else {

    Entity_ID_List faces;
    std::vector<std::size_t> nfnodes;
    Entity_Direction_List fdirs;
    Point_List cfcoords;

    mesh.getCellFacesAndDirs(c, faces, &fdirs);
    nfnodes.resize(faces.size());

    for (int j = 0; j != faces.size(); ++j) {
      auto fcoords = mesh.getFaceCoordinates(faces[j]);
      nfnodes[j] = fcoords.size();

      if (fdirs[j] == 1) {
        for (int k = 0; k != nfnodes[j]; ++k)
          cfcoords.emplace_back(fcoords[k]);
      } else {
        for (int k = nfnodes[j]-1; k >= 0; --k)
          cfcoords.emplace_back(fcoords[k]);
      }
    }

    auto ccoords = mesh.getCellCoordinates(c);
    auto vol_cent = std::make_pair((double)0,
            AmanziGeometry::Point(mesh.getSpaceDimension()));
    AmanziGeometry::polyhed_get_vol_centroid(ccoords, faces.size(), nfnodes,
            cfcoords, &vol_cent.first, &vol_cent.second);
    return vol_cent;
  }
}


template<class Mesh_type>
std::tuple<double,AmanziGeometry::Point,Point_List>
computeFaceGeometry(const Mesh_type& mesh, const Entity_ID f)
{
  if (mesh.getManifoldDimension() == 3) {
    // 3D Elements with possibly curved faces
    Point_List fcoords = mesh.getFaceCoordinates(f);

    double area;
    AmanziGeometry::Point centroid(3);
    AmanziGeometry::Point normal(3);
    AmanziGeometry::polygon_get_area_centroid_normal(fcoords, &area, &centroid, &normal);

    Entity_ID_List fcells;
    mesh.getFaceCells(f, Parallel_type::ALL, fcells);
    Point_List normals(fcells.size(), normal);

    for (int i=0; i!=fcells.size(); ++i) {
      int dir = MeshAlgorithms::getFaceDirectionInCell(mesh, f, fcells[i]);
      normals[i] = dir * normals[i];
    }
    return std::make_tuple(area, centroid, normals);

  } else if (mesh.getManifoldDimension() == 2) {

    if (mesh.getSpaceDimension() == 2) {
      // 2D mesh
      auto fcoords = mesh.getFaceCoordinates(f);
      assert(fcoords.size() == 2);

      AmanziGeometry::Point evec = fcoords[1] - fcoords[0];
      double area = std::sqrt(evec*evec);
      AmanziGeometry::Point centroid = 0.5 * (fcoords[0] + fcoords[1]);
      AmanziGeometry::Point normal(evec[1], -evec[0]);

      // only one normal needed
      Entity_ID_List fcells;
      mesh.getFaceCells(f, Parallel_type::ALL, fcells);
      Point_List normals(fcells.size(), normal);
      for (int i=0; i!=fcells.size(); ++i) {
        int dir = MeshAlgorithms::getFaceDirectionInCell(mesh, f, fcells[i]);
        normals[i] = dir * normals[i];
      }
      return std::make_tuple(area, centroid, normals);

    } else {
      // Surface mesh - cells are 2D, coordinates are 3D
      // Since the face likely forms a discontinuity in the surface
      // (or may even be the intersection of several surfaces), we
      // have to compute an outward normal to the face with respect to
      // each cell.
      auto fcoords = mesh.getFaceCoordinates(f);
      assert(fcoords.size() == 2);

      AmanziGeometry::Point evec = fcoords[1] - fcoords[0];
      double area = AmanziGeometry::norm(evec);
      AmanziGeometry::Point centroid = 0.5 * (fcoords[0] + fcoords[1]);

      Entity_ID_List cellids;
      mesh.getFaceCells(f, Parallel_type::ALL, cellids);
      Point_List normals(cellids.size());
      for (int i = 0; i < cellids.size(); i++) {
        AmanziGeometry::Point cvec = fcoords[0] - mesh.getCellCentroid(cellids[i]);
        AmanziGeometry::Point trinormal = cvec^evec;
        normals[i] = evec^trinormal;

        // renomralize to area
        double len = norm(normals[i]);
        normals[i] *= (area/len);
      }
      return std::make_tuple(area, centroid, normals);
    }
  }

  Errors::Message msg;
  msg << "Invalid mesh argument to MeshAlgorithm: manifold_dim = " << mesh.getManifoldDimension()
      << ", space_dim = " << mesh.getSpaceDimension();
  Exceptions::amanzi_throw(msg);
  return std::make_tuple(0, AmanziGeometry::Point(), std::vector<AmanziGeometry::Point>());
}


template<class Mesh_type>
Point_List
computeBisectors(const Mesh_type& mesh, const Entity_ID c,
        const Entity_ID_List& faces)
{
  Point_List bisectors(faces.size());
  for (int i = 0; i != faces.size(); ++i)
    bisectors[i] = mesh.getFaceCentroid(faces[i]) - mesh.getCellCentroid(c);
  return bisectors;
}


template<class Mesh_type>
void debugCell(const Mesh_type& mesh, const Entity_ID c)
{
  auto cc = mesh.getCellCentroid(c);
  std::stringstream stream;
  stream << "Debug cell: LID " << c << " Rank " << mesh.get_comm()->MyPID()
         << " GID " << mesh.getEntityGID(Entity_kind::CELL, c)
         << " of type " << to_string(mesh.getCellType(c)) << std::endl
          << "  centroid = " << mesh.getCellCentroid(c) << std::endl;
  Entity_ID_List faces;
  Entity_Direction_List f_dirs;
  mesh.getCellFacesAndDirs(c, faces, &f_dirs);
  for (int fi=0; fi!=faces.size(); ++fi) {
    auto fc = mesh.getFaceCentroid(faces[fi]);
    stream << "  face " << faces[fi]
           << " GID " << mesh.getEntityGID(Entity_kind::FACE, faces[fi])
           << " dir " << f_dirs[fi] << " centroid " << fc << " normal " << mesh.getFaceNormal(faces[fi]) << " bisector " <<  fc - cc << std::endl;
  }
  std::cout << stream.str();
}


template<class Mesh_type>
Point_List getFaceCoordinates(const Mesh_type& mesh, const Entity_ID f)
{
  Entity_ID_List nodes;
  mesh.getFaceNodes(f, nodes);

  Point_List coords;
  coords.reserve(nodes.size());

  for (const auto& n : nodes) {
    coords.emplace_back(mesh.getNodeCoordinate(n));
  }
  return coords;
}

template<class Mesh_type>
Point_List getCellCoordinates(const Mesh_type& mesh, const Entity_ID c)
{
  Entity_ID_List nodes;
  mesh.getCellNodes(c, nodes);

  Point_List coords;
  coords.reserve(nodes.size());

  for (const auto& n : nodes) {
    coords.emplace_back(mesh.getNodeCoordinate(n));
  }
  return coords;
}

template<class Mesh_type>
std::size_t getMaxCellNumNodes(const Mesh_type& mesh)
{
  auto ncells = mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
  int max_nnodes(0);
  for (int c=0; c!=ncells; ++c) {
    max_nnodes = std::max(max_nnodes, (int) mesh.getCellNumNodes(c));
  }
  int max_nnodes_g(0);
  mesh.getComm()->MaxAll(&max_nnodes, &max_nnodes_g, 1);
  return max_nnodes_g;
}


template<class Mesh_type>
std::size_t getMaxCellNumFaces(const Mesh_type& mesh)
{
  auto ncells = mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
  int max_nfaces(0);
  for (int c=0; c!=ncells; ++c) {
    max_nfaces = std::max(max_nfaces, (int) mesh.getCellNumFaces(c));
  }
  int max_nfaces_g(0);
  mesh.getComm()->MaxAll(&max_nfaces, &max_nfaces_g, 1);
  return max_nfaces_g;
}

template<class Mesh_type>
std::size_t getMaxCellNumEdges(const Mesh_type& mesh)
{
  auto ncells = mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
  int max_nedges(0);
  for (int c=0; c!=ncells; ++c) {
    max_nedges = std::max(max_nedges, (int) mesh.getCellNumEdges(c));
  }
  int max_nedges_g(0);
  mesh.getComm()->MaxAll(&max_nedges, &max_nedges_g, 1);
  return max_nedges_g;
}


template<class Mesh_type>
Entity_ID_List getCellFaceAdjacentCells(const Mesh_type& mesh,
        Entity_ID c, Parallel_type ptype)
{
  auto cfaces = mesh.getCellFaces(c);
  Entity_ID_List adj_cells;
  for (const auto& f : cfaces) {
    auto fcells = mesh.getFaceCells(f, ptype);
    for (const auto& fc : fcells) {
      if (c != fc &&
          std::find(adj_cells.begin(), adj_cells.end(), fc) == adj_cells.end())
        adj_cells.push_back(fc);
    }
  }
  return adj_cells;
}

template<class Mesh_type> 
KOKKOS_INLINE_FUNCTION
AmanziGeometry::Point getFaceCentroid(const Mesh_type& mesh, const Entity_ID f)
{
  auto nodes = mesh.getFaceNodes(f); 
  AmanziGeometry::Point res; 
  for(int i = 0 ; i < nodes.size(); ++i){
    auto p = mesh.template getNodeCoordinate<AccessPattern::CACHE>(nodes[i]); 
    res = res + p;
  }
  return res/nodes.size(); 
}

template<class Mesh_type> 
AmanziGeometry::Point getCellCentroid(const Mesh_type& mesh, const Entity_ID c)
{
  double vol(0.0);
  AmanziGeometry::Point res; 

  // referencer point is the first vertex of cell
  Entity_ID_List faces;
  Entity_Direction_List dirs;
  mesh.getCellFacesAndDirs(c, faces, &dirs);
  int nfaces = faces.size();

  // special case of column mesh - only top and bottom faces are returned 
  if (nfaces == 2) {
    assert(false);
  }

  // general case
  auto nodes0 = mesh.getFaceNodes(faces[0]);
  auto p0 = mesh.template getNodeCoordinate<AccessPattern::CACHE>(nodes0[0]); 

  for (int n = 0 ; n < nfaces; ++n) {
    int f = faces[n];
    auto nodes = mesh.getFaceNodes(f);
    int nnodes = nodes.size();

    // for a 3D simplex
    if (mesh.getManifoldDimension() == 3) {
      auto pf = getFaceCentroid(mesh, f);

      for (int i1 = 0 ; i1 < nnodes; ++i1) {
        int i2 = (i1 + 1) % nnodes;
        auto p1 = mesh.template getNodeCoordinate<AccessPattern::CACHE>(nodes[i1]); 
        auto p2 = mesh.template getNodeCoordinate<AccessPattern::CACHE>(nodes[i2]); 

        // face nodes are ordered wrt the given normal
        // since algorithm uses exterior normal, we correct for normal orientation
        double voltet = (((p1 - p0)^(p2 - p0)) * (pf - p0)) * dirs[i1];
        vol += voltet;
        res += (p0 + p1 + p2 + pf) * voltet;
      }
    }
    else if (mesh.getManifoldDimension() == 2) {
      assert(false);
    }
  }

  return res / (24 * vol); 
}

} // namespace MeshAlgorithms
} // namespace AmanziMesh
} // namspace Amanzi
