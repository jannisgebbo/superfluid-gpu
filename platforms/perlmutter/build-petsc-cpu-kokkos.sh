#!/bin/bash
# Builds PETSc 3.25.5 + Kokkos/Kokkos-Kernels 5.1.0 (OpenMP backend only, no
# CUDA) + HDF5 for SuperfluidGPU on CPU nodes, and installs it to
# $CFS/m3722/opt/prlm/petsc-cpu-kokkos-int64. Same source tree and package
# tarballs as build-petsc-gpu.sh, separate PETSC_ARCH.
#
# On an interactive CPU node (srun so it runs on the node, not the login;
# 4 tasks so make check can run its 2-rank tests):
#   salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m3722 \
#     --ntasks=4 --cpus-per-task=64 \
#     srun -n 1 -c 64 bash build-petsc-cpu-kokkos.sh > build-cpu-kokkos.log 2>&1
#
#SBATCH -A m3722
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J petsc-cpu-kokkos-build
#SBATCH -o build-petsc-cpu-kokkos-%j.log

# cpu replaces the default gpu module set: no cudatoolkit and no
# craype-accel-nvidia80, so CC does not link the CUDA MPI library
module load cpu
module load PrgEnv-gnu
module list 2>&1

# After the module loads: Lmod scripts are not safe under set -u
set -euo pipefail

# configure and make check launch MPI test programs with srun. Under
# salloc+srun they run inside our step, which holds the whole node, so they
# must be allowed to share its resources or they wait until timeout.
export SLURM_OVERLAP=1

PREFIX=/global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-kokkos-int64
export PETSC_DIR=/pscratch/sd/d/dteaney/petsc-build/petsc-3.25.5
export PETSC_ARCH=arch-prlm-cpu-kokkos-int64
# Tarballs fetched on a login node, so the build needs no network access
PKGS=/pscratch/sd/d/dteaney/petsc-build/pkgs
cd $PETSC_DIR

echo "== configure  $(date)"
./configure --prefix=$PREFIX \
  --with-cc=cc --with-cxx=CC --with-fc=0 \
  --with-64-bit-indices --with-debugging=0 \
  --with-cuda=0 \
  --with-openmp \
  --download-kokkos=$PKGS/kokkos-5.1.0.tar.gz \
  --download-kokkos-kernels=$PKGS/kokkos-kernels-5.1.0.tar.gz \
  --download-hdf5=$PKGS/hdf5-1.14.6.tar.gz \
  --with-mpiexec=srun \
  --with-make-np=32 \
  COPTFLAGS=-O3 CXXOPTFLAGS=-O3

echo "== make all  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

echo "== make install  $(date)"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH install

# Group-readable for everyone in m3722
chmod -R g+rX $PREFIX

echo "== make check (installed)  $(date)"
make PETSC_DIR=$PREFIX PETSC_ARCH="" check || echo "make check FAILED"

echo "== done  $(date)"
