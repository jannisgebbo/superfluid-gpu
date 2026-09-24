# Perlmutter CPU environment for the Kokkos/OpenMP build of SuperfluidGPU:
# PETSc 3.25.5 + Kokkos 5.1.0 (OpenMP backend), 64-bit indices, built from
# source with build-petsc-cpu-kokkos.sh. Source this in a fresh shell before
# cmake/make, and in CPU job scripts.
#   cmake -S . -B build-omp -DSFGPU_KOKKOS=ON
# For the plain MPI CPU build use setupprlm.sh; for GPUs setupprlm-gpu.sh.
module load cpu
module load PrgEnv-gnu
module load cray-fftw

export PETSC_DIR=/global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-kokkos-int64
export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$PETSC_DIR:$CMAKE_PREFIX_PATH

# Threads stay on their rank's cores, so memory is first touched (and
# placed) in the rank's own NUMA domain
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export MODELGPATH=$HOME/SuperfluidGPU
export MODELGEXEPATH=$MODELGPATH/build-omp/src
export PYTHONPATH=$MODELGPATH/scripts:$PYTHONPATH
