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
    MeshCache<MemSpace_type::DEVICE> mesh(framework_mesh);
    mesh.cacheCellFaces();

    {
      auto cf = mesh.getCellFaces(0);
      Kokkos::parallel_for("this should fail to compile", 6,
                           KOKKOS_LAMBDA(const int& i) {
                             cf(i) = -1;
                           });
    }
    {
      auto cf = mesh.getCellFaces(0);
      Kokkos::View<Entity_ID*, Kokkos::HostSpace> cf_host("cf host", cf.size());
      Kokkos::deep_copy(cf_host, cf);
      // if it does compile, this should be true
      assert(cf(0) >= 0);
      assert(cf(5) >= 0);
    }
  }
  Kokkos::finalize();
}
