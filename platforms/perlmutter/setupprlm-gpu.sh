# Perlmutter GPU environment for SuperfluidGPU: PETSc 3.25.5 + Kokkos 5.1.0
# (CUDA + OpenMP backends) + CUDA 13.2, 64-bit indices, built from source.
# Source this in a fresh shell before cmake/make, and in GPU job scripts.
# For CPU-only builds use setupprlm.sh instead.
#
# How the PETSc install was made: platforms/perlmutter/Perlmutter.md.
module load PrgEnv-gnu
module load craype-accel-nvidia80
# CUDA 13.2 matches cray-mpich 9.1.0, whose GPU library (libmpi_gtl_cuda)
# links libcudart.so.13. Load it last; loading others afterwards can reset it.
module load cudatoolkit/13.2
module load cray-fftw

export PETSC_DIR=/global/cfs/cdirs/m3722/opt/prlm/petsc-gpu-int64
export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
export CMAKE_PREFIX_PATH=$PETSC_DIR:$CMAKE_PREFIX_PATH

# Lets cray-mpich send directly from GPU buffers (PETSc ghost exchange)
export MPICH_GPU_SUPPORT_ENABLED=1

export MODELGPATH=$HOME/SuperfluidGPU
# GPU build directory: cmake -B build-gpu -DSFGPU_KOKKOS=ON
export MODELGEXEPATH=$MODELGPATH/build-gpu/src
export PYTHONPATH=$MODELGPATH/scripts:$PYTHONPATH
