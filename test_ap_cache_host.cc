#include "Point.hh"
#include "MeshDefs.hh"
#include "MeshSimple.hh"
#include "MeshCache.hh"

using namespace Amanzi;
using namespace Amanzi::AmanziMesh;

#include "test_helpers.hh"

int main(int argc, char** argv)
{
  Kokkos::initialize(argc, argv);
  {std::size_t nx=10, ny=10, nz=10;
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 1) nx = atoi(argv[1]);

    std::cout<<"nx: "<<nx<<" ny: "<<ny<<" nz: "<<nz<<std::endl;

    auto framework = std::make_shared<MeshSimple>(0,0,0,1,1,1,nx,ny,nz);
    MeshCache<MemSpace_type::DEVICE> mesh_on_device(framework);
    mesh_on_device.cacheFaceCells();
    mesh_on_device.cacheCellFaces();
    mesh_on_device.cacheFaceNodes(); 
    mesh_on_device.cacheCellGeometry();
    mesh_on_device.cacheFaceGeometry();
    mesh_on_device.cacheNodeCoordinates();

    MeshCache<MemSpace_type::HOST> mesh(mesh_on_device);
    mesh_on_device.destroyFramework();
    mesh.destroyFramework();

    assert(nx*ny*nz == mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED));
    //assert(close(0.3333333*0.3333333*0.3333333, mesh.getCellVolume(0), 1.e-5));

    // do some realish work
    Entity_ID ncells = mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
    Entity_ID nfaces = mesh.getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);

    // Check compute 
    for (Entity_ID c=0; c!=ncells; ++c) {
      Entity_ID_List cfaces;
      mesh.getCellFaces(c, cfaces);
      auto cc = mesh.getCellCentroid<AccessPattern::COMPUTE>(c);
      AmanziGeometry::Point sum(0., 0., 0.);
      for (auto f : cfaces) {
        assert(f < nfaces);
        auto fc = mesh.getFaceCentroid<AccessPattern::COMPUTE>(f);

        // compute sum of all bisectors
        sum += (fc - cc);
      }
      assert(close(0., AmanziGeometry::norm(sum), 0., 1.e-6));
    }

    for (Entity_ID c=0; c!=ncells; ++c) {
      Entity_ID_List cfaces;
      mesh.getCellFaces(c, cfaces);
      auto cc = mesh.getCellCentroid(c);
      AmanziGeometry::Point sum(0., 0., 0.);
      for (auto f : cfaces) {
        assert(f < nfaces);
        auto fc = mesh.getFaceCentroid(f);

        // compute sum of all bisectors
        sum += (fc - cc);
      }
      assert(close(0., AmanziGeometry::norm(sum), 0., 1.e-6));
    }

    // compute the gradient of a field?
    std::vector<double> p(ncells);
    for (Entity_ID c=0; c!=ncells; ++c) {
      auto cc = mesh.getCellCentroid(c);
      p[c] = cc[0] + cc[1];
    }

    std::vector<double> p_faces(nfaces);
    for (Entity_ID f=0; f!=nfaces; ++f) {
      auto fc = mesh.getFaceCentroid(f);
      p_faces[f] = fc[0] + fc[1];
    }

    // these mallocs are amortized across as big a range as possible
    AmanziMesh::Entity_ID_List fcells, cfaces;
    std::vector<AmanziGeometry::Point> bisectors;
    Entity_Direction_List dirs;

    // compute transmissibliilty
    std::vector<double> trans(nfaces, 0.);
    for (int c = 0; c < ncells; ++c) {
      mesh.getCellFacesAndBisectors(c, cfaces, &bisectors);
      int ncfaces = cfaces.size();

      for (int i = 0; i < ncfaces; i++) {
        int f = cfaces[i];
        const AmanziGeometry::Point& a = bisectors[i];
        const AmanziGeometry::Point& normal = mesh.getFaceNormal(f);
        double area = mesh.getFaceArea(f);

        double h_tmp = AmanziGeometry::norm(a);
        double s = area / h_tmp;
        double perm = (a * normal) * s;
        double dxn = a * normal;
        trans[f] += std::abs(dxn / perm);
      }
    }
    for (int f = 0; f < nfaces; ++f) {
      trans[f] = 1./trans[f];
    }


    std::vector<double> flux(nfaces);
    std::vector<int> flag(nfaces, 0);
    for (int c = 0; c < ncells; c++) {
      mesh.getCellFacesAndDirs(c, cfaces, &dirs);
      int ncfaces = cfaces.size();

      for (int n = 0; n < ncfaces; n++) {
        int f = cfaces[n];
        mesh.getFaceCells(f, Parallel_type::ALL, fcells);
        if (fcells.size() == 1) {
          double value = p_faces[f];
          flux[f] = dirs[n] * trans[f] * (p[c] - value);

        } else {
          if (!flag[f]) {
            if (fcells.size() <= 1) {
              Errors::Message msg("Flow PK: These boundary conditions are not supported by FV.");
              Exceptions::amanzi_throw(msg);
            }
            int c1 = fcells[0];
            int c2 = fcells[1];
            if (c == c1) {
              flux[f] = dirs[n] * trans[f] * (p[c1] - p[c2]);
            } else {
              flux[f] = dirs[n] * trans[f] * (p[c2] - p[c1]);
            }
            flag[f] = 1;
          }
        }
      }
    }

    // check: true gradient
    AmanziGeometry::Point truth(-1.0, -1.0, 0.);
    for (int f = 0; f != nfaces; ++f) {
      auto normal = mesh.getFaceNormal(f);
      bool result = close(normal * truth, flux[f], 1.e-6, 1.e-6);
      if (!result) {
        std::cout << "FAIL: " << f << std::endl
                  << "    normal = " << normal << std::endl
                  << "      truth = " << truth << std::endl
                  << "      flux = " << flux[f] << std::endl
                  << "      res = " << normal * truth << std::endl;
      }
      assert(result);
    }
  }
  Kokkos::finalize();
}
