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
  {
    auto framework_mesh = std::make_shared<MeshSimple>(0,0,0,1,1,1,3,3,3);
    MeshCache<MemSpace_type::DEVICE> mesh_on_device(framework_mesh);
    // mesh_on_device.cacheFaceCells();
    // mesh_on_device.cacheCellFaces();
    // mesh_on_device.cacheCellGeometry();
    // mesh_on_device.cacheFaceGeometry();

    MeshCache<MemSpace_type::HOST> mesh(mesh_on_device);
    static const AccessPattern AP = AccessPattern::FRAMEWORK;

    assert(3*3*3 == mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED));
    assert(close(0.3333333*0.3333333*0.3333333, mesh.getCellVolume<AP>(0), 1.e-5));

    // do some realish work
    Entity_ID ncells = mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);

#if 1
    const int nnodes = 20; 

    Kokkos::parallel_for(mesh.getPolicy(ncells,nnodes),
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type tm){
        int t = tm.league_rank () * tm.team_size () +
                tm.team_rank ();
        if(t >= ncells) return; 
        Kokkos::View<Entity_ID*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          nodes (tm.team_scratch(1), nnodes*tm.team_size());
        // Get my subbiew of the shared mem 
        auto sv = Kokkos::subview(nodes, Kokkos::make_pair(t*nnodes,(t+1)*nnodes));
        mesh.getCellNodes(t,sv);
    });
#endif 

    Entity_ID nfaces = mesh.getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);
    Kokkos::DualView<double*> sums("sums", ncells);
    auto sum_device = view<MemSpace_type::DEVICE>(sums);
    Kokkos::parallel_for("normal O(N) work", ncells,
                         KOKKOS_LAMBDA(const int& c) {
                           auto cfaces = mesh.getCellFaces(c);
                           auto cc = mesh.getCellCentroid<AP>(c);
                           AmanziGeometry::Point sum(0., 0., 0.);
                           for (int i=0; i!=cfaces.size(); ++i) {
                             auto fc = mesh.getFaceCentroid<AP>(cfaces[i]);
                             // compute sum of all bisectors
                             sum += (fc - cc);
                           }
                           sum_device(c) = AmanziGeometry::norm(sum);
                         });
    sums.modify<Kokkos::DefaultExecutionSpace>();
    sums.sync<Kokkos::HostSpace>();
    for (int c=0; c!=ncells; ++c) {
      assert(close(0., view<MemSpace_type::HOST>(sums)(c), 0., 1.e-6));
    }

    // compute the gradient of a field?
    Kokkos::View<double*> p("p_cell", ncells);
    Kokkos::parallel_for("fill p cells", ncells,
                         KOKKOS_LAMBDA(const int& c) {
                           auto cc = mesh.getCellCentroid<AP>(c);
                           p[c] = cc[0] + cc[1];
                         });

    Kokkos::View<double*> p_faces("p_face", nfaces);
    Kokkos::parallel_for("fill p faces", nfaces,
                         KOKKOS_LAMBDA(const int& f) {
                           auto fc = mesh.getFaceCentroid<AP>(f);
                           p_faces[f] = fc[0] + fc[1];
                         });


    // fill the transmissiblity vector
    Kokkos::View<double*> trans("trans", nfaces);
    Kokkos::parallel_for("compute trans", ncells,
                         KOKKOS_LAMBDA(const int& c) {
                           auto cf_cb = mesh.getCellFacesAndBisectors(c);
                           auto cfaces = cf_cb.first;
                           auto bisectors = cf_cb.second;

                           int nfaces = cfaces.size();
                           for (int i = 0; i < nfaces; i++) {
                             int f = cfaces[i];
                             const AmanziGeometry::Point& a = bisectors[i];
                             const AmanziGeometry::Point& normal = mesh.getFaceNormal<AP>(f);
                             double area = mesh.getFaceArea<AP>(f);

                             double h_tmp = AmanziGeometry::norm(a);
                             double s = area / h_tmp;
                             double perm = (a * normal) * s;
                             double dxn = a * normal;
                             Kokkos::atomic_add(&trans[f], std::abs(dxn / perm));
                           }
                         });
    Kokkos::parallel_for("compute trans 2", nfaces,
                         KOKKOS_LAMBDA(const int& f) {
                           trans[f] = 1./trans[f];
                         });


    // compute the flux
    Kokkos::DualView<double*> flux_dv("flux", nfaces);
    Kokkos::View<double*> flux = view<MemSpace_type::DEVICE>(flux_dv);
    Kokkos::parallel_for("compute flux", ncells,
                         KOKKOS_LAMBDA(const int& c) {
                           auto cf_cd = mesh.getCellFacesAndDirections(c);
                           auto cfaces = cf_cd.first;
                           auto dirs = cf_cd.second;
                           int nfaces = cfaces.size();

                           for (int n = 0; n < nfaces; n++) {
                             int f = cfaces[n];
                             auto fcells = mesh.getFaceCells(f, Parallel_type::ALL);
                             if (fcells.size() == 1) {
                               double value = p_faces[f];
                               flux[f] = dirs[n] * trans[f] * (p[c] - value);

                             } else {
                               // NOTE: in the original algorithm, there is a
                               // check on flag to see if we have already
                               // computed this from one cell.  If so, we can
                               // skip it.  However, here we choose to compute
                               // it twice and not care who writes it?
                               int c1 = fcells[0];
                               int c2 = fcells[1];
                               if (c == c1) {
                                 flux[f] = dirs[n] * trans[f] * (p[c1] - p[c2]);
                               } else {
                                 flux[f] = dirs[n] * trans[f] * (p[c2] - p[c1]);
                               }
                             }
                           }
                         });
    flux_dv.modify<Kokkos::DefaultExecutionSpace>();
    flux_dv.sync<Kokkos::HostSpace>();
    auto flux_h = view<MemSpace_type::HOST>(flux_dv);

    // check: true gradient
    AmanziGeometry::Point truth(-1.0, -1.0, 0.);
    for (int f = 0; f != nfaces; ++f) {
      auto normal = framework_mesh->getFaceNormal(f);
      bool result = close(normal * truth, flux_h[f], 1.e-6, 1.e-6);
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
