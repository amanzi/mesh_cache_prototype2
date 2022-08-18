# Mesh Cache Prototype

## Tests

* `test_framework`: does some work, only uses a framework mesh.  Passes.
* `test_ap_cached`: Mimics this test using on-device calls, assuming EVERYTHING it uses is cached.  Passes on Kokkos::Serial.
* `test_ap_framework`: Concept -- use a `MeshCache<HOST>` and `AP == FRAMEWORK` and see if that works.
* ...
