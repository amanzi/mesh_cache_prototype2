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
    std::size_t nx=10, ny=10, nz=10;
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 1) nx = atoi(argv[1]);

    std::cout<<"nx: "<<nx<<" ny: "<<ny<<" nz: "<<nz<<std::endl;

    Kokkos::Timer timer; 
    double start = timer.seconds(); 

    auto framework_mesh = std::make_shared<MeshSimple>(0,0,0,1,1,1,nx,ny,nz);
    MeshCache<MemSpace_type::DEVICE> mesh(framework_mesh);
    MeshCache<MemSpace_type::HOST> host_mesh(mesh); 
    assert(nx*ny*nz == host_mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED));
    //assert(close(0.3333333*0.3333333*0.3333333, host_mesh.getCellVolume<AP>(0), 1.e-5));

    Kokkos::fence(); 
    double stop = timer.seconds(); 

    std::cout<<"ncells: "<<mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED)<<std::endl;
    std::cout<<"Construction: "<<stop-start<<"s"<<std::endl;
    
    start = timer.seconds(); 

    mesh.cacheFaceCells();
    mesh.cacheCellFaces();
    mesh.cacheFaceNodes(); 
    mesh.cacheCellGeometry();
    mesh.cacheFaceGeometry();
    mesh.cacheNodeCoordinates();
    mesh.destroyFramework();

    // Host access mesh
    Kokkos::fence(); 
    stop = timer.seconds(); 
    std::cout<<"Caching: "<<stop-start<<"s"<<std::endl;
    start = timer.seconds();

    static const AccessPattern AP = AccessPattern::CACHE;


    // do some realish work
    Entity_ID ncells = host_mesh.getNumEntities(Entity_kind::CELL, Parallel_type::OWNED);
    Entity_ID nfaces = host_mesh.getNumEntities(Entity_kind::FACE, Parallel_type::OWNED);

    const int nnodes = 20; // upper limit

// Example using TeamPolicy 
#if 1
    Kokkos::parallel_for(mesh.getPolicy(ncells,nnodes),
      KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type tm){
        int t = tm.league_rank () * tm.team_size () +
                tm.team_rank ();
        if(t == 0){
          printf("league size: %d team size: %d\n",tm.league_size(),tm.team_size());
        }
        if(t >= ncells) return; 

        Entity_ID* mem = (Entity_ID*)
          tm.team_shmem().get_shmem(tm.team_size()*nnodes);
        
        Kokkos::View<Entity_ID*, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          nodes (tm.team_scratch(1), nnodes*tm.team_size()); 
        // Get my subbiew of the shared mem 
        auto sv = Kokkos::subview(nodes, Kokkos::make_pair(t*nnodes,(t+1)*nnodes));

        mesh.getCellNodes(t,sv); 

    }); 
#endif 

    // Check sum using COMPUTE for getFaceCentroid and getCellCentroid
    {
      Kokkos::DualView<double*> sums("sums", ncells);
      auto sum_device = view<MemSpace_type::DEVICE>(sums);
      Kokkos::parallel_for("normal O(N) work", ncells,
                          KOKKOS_LAMBDA(const int& c) {
                            auto cfaces = mesh.getCellFaces<AccessPattern::CACHE>(c);
                            auto cc = mesh.getCellCentroid<AccessPattern::COMPUTE>(c); 
                            AmanziGeometry::Point sum(0., 0., 0.);
                            for (int i=0; i<cfaces.size(); ++i) {
                              auto fc = mesh.getFaceCentroid<AccessPattern::COMPUTE>(cfaces[i]); 
                              // compute sum of all bisectors
                              // Need to check why the += is problematic
                              auto tmp = (fc - cc);
                              sum = sum + tmp;
                            }
                            sum_device(c) = AmanziGeometry::norm(sum);
                          });

      Kokkos::deep_copy(sums.view_host(),sums.view_device());
      for (int c=0; c!=ncells; ++c) {
        assert(close(0., view<MemSpace_type::HOST>(sums)(c), 0., 1.e-6));
      }
    }

    Kokkos::DualView<double*> sums("sums", ncells);
    auto sum_device = view<MemSpace_type::DEVICE>(sums);
    Kokkos::parallel_for("normal O(N) work", ncells,
                         KOKKOS_LAMBDA(const int& c) {
                           auto cfaces = mesh.getCellFaces<AP>(c);
                           auto cc = mesh.getCellCentroid<AP>(c);
                           AmanziGeometry::Point sum(0., 0., 0.);
                           for (int i=0; i<cfaces.size(); ++i) {
                             auto fc = mesh.getFaceCentroid<AP>(cfaces[i]);
                             // compute sum of all bisectors
                             // Need to check why the += is problematic
                             auto tmp = (fc - cc);
                             sum = sum + tmp;
                           }
                           sum_device(c) = AmanziGeometry::norm(sum);
                         });

    Kokkos::deep_copy(sums.view_host(),sums.view_device());
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

    Kokkos::deep_copy(flux_dv.view_host(),flux_dv.view_device()); 
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
                  << "      flux = " << flux_h[f] << std::endl
                  << "      res = " << normal * truth << std::endl;
      }
      assert(result);
    }
    Kokkos::fence(); 
    std::cout<<"Computation: "<<timer.seconds()-start<<"s"<<std::endl;
  }
  Kokkos::finalize();
}
