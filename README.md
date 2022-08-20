# Mesh Cache Prototype

## Current Tests

* `test_framework`: does some work, only uses a framework mesh.  Passes.
* `test_ap_cached`: Mimics this test using on-device calls, assuming EVERYTHING it uses is cached.  Passes on Kokkos::Serial.
* `test_const_correct:` Need some tests to ensure the a poorly-written kernel cannot change the mesh data/cache.  Likely this test SHOULD NOT COMPILE, but if it does, it should pass.  Right now it can -- this test fails!


## Future Tests
* `test_ap_cached_cuda`: Same as above, but using CUDA.  This will check that the Cache implementation on device is correct.
* `test_ap_framework`: Develop a `MeshCache<HOST>` specialization and use it to run a mix of CACHED operations and FRAMEWORK operations on HOST.
* `test_ap_compute_host:`: See if we can get COMPUTE on HOST to work without relying on the framework.  Cache a few things (will need to add node coordinates) and then delete the framework and compute a few things (e.g. cell volumes, etc).
* `test_ap_compute_device:`: Bsed on `test_ap_cached_cuda`, see how much we can COMPUTE on DEVICE.

