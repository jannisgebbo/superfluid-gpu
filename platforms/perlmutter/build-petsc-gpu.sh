#!/bin/bash
# Builds PETSc 3.25.5 + Kokkos/Kokkos-Kernels 5.1.0 (CUDA + OpenMP) + HDF5 for
# SuperfluidGPU and installs it to $CFS/m3722/opt/prlm/petsc-gpu-int64.
# Runs on a GPU node so `make check` exercises CUDA/Kokkos on a real A100.
#
#   sbatch build-petsc-gpu.sh
# or on an interactive GPU node (srun so it runs on the node, not the login):
#   salloc -N 1 -C gpu -q interactive -t 02:00:00 -A m3722_g --gpus-per-node=4 \
#     srun -n 1 -c 128 --gpus-per-node=4 bash build-petsc-gpu.sh \
#     > build-petsc-gpu-interactive.log 2>&1
#
#SBATCH -A m3722_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -J petsc-gpu-build
#SBATCH -o build-petsc-gpu-%j.log

module load PrgEnv-gnu
module load craype-accel-nvidia80
# CUDA 13.2 matches cray-mpich 9.1.0, whose GPU library (libmpi_gtl_cuda)
# links libcudart.so.13. Load it last; loading others afterwards can reset it.
module load cudatoolkit/13.2
module list 2>&1

# After the module loads: Lmod scripts are not safe under set -u
set -euo pipefail
echo "CUDA_HOME=$CUDA_HOME"

# configure and make check launch MPI test programs with srun. Under
# salloc+srun they run inside our step, which holds the whole node, so they
# must be allowed to share its resources or they wait until timeout.
export SLURM_OVERLAP=1

PREFIX=/global/cfs/cdirs/m3722/opt/prlm/petsc-gpu-int64
export PETSC_DIR=/pscratch/sd/d/dteaney/petsc-build/petsc-3.25.5
export PETSC_ARCH=arch-prlm-gpu-int64
# Tarballs fetched on a login node, so the build needs no network access
PKGS=/pscratch/sd/d/dteaney/petsc-build/pkgs
cd $PETSC_DIR

echo "== configure  $(date)"
./configure --prefix=$PREFIX \
  --with-cc=cc --with-cxx=CC --with-fc=0 \
  --with-64-bit-indices --with-debugging=0 \
  --with-cuda --with-cuda-dir=$CUDA_HOME --with-cuda-arch=80 \
  --with-openmp \
  --download-kokkos=$PKGS/kokkos-5.1.0.tar.gz \
  --download-kokkos-kernels=$PKGS/kokkos-kernels-5.1.0.tar.gz \
  --download-hdf5=$PKGS/hdf5-1.14.6.tar.gz \
  --with-mpiexec=srun \
  --with-make-np=32 \
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3 CUDAOPTFLAGS=-O3

echo "== make all  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

echo "== make install  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH install

# Group-readable for everyone in m3722
chmod -R g+rX $PREFIX

echo "== make check (installed)  $(date)"
make PETSC_DIR=$PREFIX PETSC_ARCH="" check || echo "make check FAILED"

echo "== done  $(date)"
