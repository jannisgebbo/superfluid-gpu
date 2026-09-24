# Standalone checks, not part of the model. Built only for the GPU build.

# Toolchain check: DMDA on Kokkos vectors, a device kernel, and a periodic
# ghost exchange on the GPU
add_executable(kokkos_smoke.exe kokkos_smoke.cxx)
target_link_libraries(kokkos_smoke.exe PRIVATE modelg)
