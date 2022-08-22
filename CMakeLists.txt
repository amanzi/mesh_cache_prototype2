cmake_minimum_required(VERSION 3.10)
project(mesh_prototype)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()


find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)
#find_package(CUDA REQUIRED)


set(inc_files
  errors.hh
  exceptions.hh
  Point.hh
  Geometry.hh
  MeshDefs.hh
  Mesh_Helpers_decl.hh
  Mesh_Helpers_impl.hh
  MeshFramework.hh
  MeshSimple.hh
  )

set(src_files
  errors.cc
  exceptions.cc
  Geometry.cc
  MeshFramework.cc
  MeshSimple.cc
  )

set(required_libs Kokkos::kokkoscore)

add_library(${PROJECT_NAME} SHARED ${src_files})
target_link_libraries(${PROJECT_NAME} PUBLIC ${required_libs})

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${inc_files}")


enable_testing()

function ( add_test_fails_to_compile test_name )
  add_executable(${test_name} ${test_name}.cc)
  target_link_libraries(${test_name} ${PROJECT_NAME} ${required_libs})
  set_target_properties(${test_name} PROPERTIES
    EXCLUDE_FROM_ALL TRUE
    EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  add_test(NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} --build . --target ${test_name}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(${test_name} PROPERTIES WILL_FAIL TRUE)
endfunction()
  


#
# This test uses only the MeshFramework
#
add_executable(test_framework test_framework.cc)
target_link_libraries(test_framework ${PROJECT_NAME} ${required_libs})
add_test(test_framework test_framework)

#
# This test assumes EVERYTHING it uses is cached.
#
add_executable(test_ap_cache test_ap_cache.cc)
target_link_libraries(test_ap_cache ${PROJECT_NAME} ${required_libs})
add_test(test_ap_cache test_ap_cache)

#
# Everything from framework -- this is a trivial pass-through
#
add_executable(test_ap_framework test_ap_framework.cc)
target_link_libraries(test_ap_framework ${PROJECT_NAME} ${required_libs})
add_test(test_ap_framework test_ap_framework)


#
# This test assumes EVERYTHING it uses is cached, and uses
# MeshCache<HOST>
#
# NOTE: This one does not yet compile -- WIP!
add_executable(test_ap_cache_host test_ap_cache_host.cc)
target_link_libraries(test_ap_cache_host ${PROJECT_NAME} ${required_libs})
add_test(test_ap_cache_host test_ap_cache_host)


# NOTE: this adds a test that FAILS TO COMPILE
add_test_fails_to_compile( test_const_correct )
add_test_fails_to_compile( test_const_correct2 )

