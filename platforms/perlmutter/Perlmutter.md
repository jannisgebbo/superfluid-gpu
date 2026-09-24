# Script
========

From the ModelG directory one executes
```bash
ModelG> source ./Makefiles/setupprlm.sh
ModelG> make -f ./Makefiles/Makefile.prlm
```
You can look at Makefiles/setupprlm.sh to see what it does. 

# How it works
=============

```bash
# The system wide spack instance (installed with git is in $SPACK_ROOT)
module load spack
# This sets the system wide defaults which are in $SPACK_SYSTEM_CONFIG_PATH
module load spack-config
module load cray-fftw
# Activate the spack environment in our common file system area
spack env activate $CFS/m3722/opt/prlm/petsc-cpu-int64
# print out the envirnoment to make sure we know what we are working with
spack env status
spack load petsc
# From the ModelG directory issue the build command
make -f Makefiles/Makefile.prlm
```

# Installing Petsc In A Home Directory Spack package manager 
==========================================================

Suppose we want our own Petsc with whatever

```bash
module load spack
module load spack-config
```
The second one sets up  the spack compilers at nersc. 
Now we can do 

```bash
spack env create mypetsc
spack env activate mypetsc
# Print out the environment to make sure we know what we are working with
spack env status
# Add the software we want
spack add petsc +int64 +hdf5
spack install
```
Ok so we created the petsc-int64 environmnet with the software we need

Then, imagine we create a new login shell.  The process of using mypetsc environment  is

```bash
module load spack
module load spack-config
module load cray-fftw
spack env activate mypetsc
spack load petsc
```

With these steps we are ready to use the petsc universe

# Petsc Common Environment On Perlmutter
========================================

Derek set up PETSc to work in  $CFS/m3722/opt/ so everyone doesn't need to spin up there own version of PETSc. Though this wasn't too bad

Here we first created the shared spack environment in the shared directory. Note the '-d'

```bash
spack create -d ${shared_directory}/petsc-int64
spack acivate ${shared_directory}/petsc-int64
spack add petsc+hdf5+int64
```
Before installation, in the environment directory we editted the "spack.yaml" file

```yaml
spack:
  # add package specs to the `specs` list
  specs:
  - petsc+hdf5+int64
  config:
    install_tree:
      root: $env/opt
  view: true
  concretizer:
    unify: true
```
We added the lines
```yaml
  config:
    # This tells Spack to install packages into a subfolder named 'opt'
    # right inside your project directory.
    # $env is the director of the environment
    install_tree:
      root: $env/opt
```
This guarantees that the software is built in the common directory and not in my home directory.

Then we install in the usual way
```bash
# This may not be needed, and just makes sure that after editting spack.yaml everything is ok
spack concretize -f
spack install
```

# Spack View
============

The directory where the binaries are stored is 

```bash
Updating view at /global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-int64/.spack-env/view
```

Since petsc is installing its own hdf5 we find 

```bash
(base) dteaney@perlmutter:login35:~> which h5dump
/global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-int64/.spack-env/view/bin/h5dump
```

Similarly we find
```bash
pkg-config --variable=pcfiledir PETSc
/global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-int64/.spack-env/view/lib/pkgconfig
```
One can look at the file and it is kind of helpful, e.g. 
```bash
(base) dteaney@perlmutter:login35:~> cat `pkg-config --variable=pcfiledir PETSc`/PETSc.pc
prefix=/global/cfs/cdirs/m3722/opt/prlm/petsc-cpu-int64/opt/linux-zen3/petsc-3.24.1-zjjlfkajgqfrhgmefahoq7trcjuwecuy
ccompiler=/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/bin/mpicc
# etc, etc 
```

# GPU build: PETSc + Kokkos + CUDA from source
===============================================

Spack could not concretize `petsc +cuda +kokkos` (the solver ran for more than
40 minutes without finishing), so the GPU PETSc is built from source with
PETSc's own `configure`.

**Install:** `/global/cfs/cdirs/m3722/opt/prlm/petsc-gpu-int64`
(PETSc 3.25.5, Kokkos + Kokkos-Kernels 5.1.0, HDF5 1.14.6, 64-bit indices,
CUDA 13.2 for sm_80, OpenMP). The Kokkos devices are OPENMP, SERIAL and CUDA.

**Why CUDA 13.2:** cray-mpich 9.1.0 (CPE 26.03) has a GPU-aware library,
`libmpi_gtl_cuda`, that links `libcudart.so.13`. With the
`craype-accel-nvidia80` module loaded, the Cray `cc`/`CC` wrappers link it
automatically.

**How it was built:** `build-petsc-gpu.sh` (in this directory). The source and
build tree were in `/pscratch/sd/d/dteaney/petsc-build/`. The package tarballs
were downloaded on a login node first (`configure
--with-packages-download-dir=<dir>` prints the URLs), so the build needs no
network access. It ran on an interactive GPU node:

```bash
salloc -N 1 -C gpu -q interactive -t 02:00:00 -A m3722_g --gpus-per-node=4 \
  srun -n 1 -c 128 --gpus-per-node=4 bash build-petsc-gpu.sh > build.log 2>&1
```

A few things matter here:
- `cudatoolkit/13.2` must be loaded *after* `craype-accel-nvidia80`.
  Loading them the other way round can reset the version.
- The script sets `SLURM_OVERLAP=1`. configure and `make check` launch MPI
  test programs with `srun` inside our step, and without it they wait until
  they time out.
- Configure took about 11 minutes, `make` 2 minutes, install 3 minutes.

## Building SuperfluidGPU for GPU

```bash
source platforms/perlmutter/setupprlm-gpu.sh
cmake -S . -B build-gpu -DSFGPU_KOKKOS=ON
cmake --build build-gpu -j 16
```
`-DSFGPU_KOKKOS=ON` switches to C++20 (required by Kokkos 5), links
`Kokkos::kokkos`, and builds `tests/`. Kokkos's launcher compiles every
source file of targets that link Kokkos with `nvcc`. The CPU build (`build/`,
`setupprlm.sh`) is unchanged.

## Running on GPUs

```bash
# one allocation, reused for many runs
salloc -N 1 -C gpu -q interactive -t 01:00:00 -A m3722_g \
       --ntasks=4 --gpus-per-node=4 --no-shell
srun --jobid=<id> -n 4 --gpus-per-node=4 --gpu-bind=none \
     build-gpu/tests/kokkos_smoke.exe -n 256 --kokkos-map-device-id-by=mpi_rank
```
- **`--gpu-bind=none` is required for more than one rank.** cray-mpich sends
  GPU buffers between ranks with CUDA IPC, which needs every rank to see all
  GPUs. With `--gpus-per-task=1` the exchange hangs with
  `cuIpcOpenMemHandle: invalid argument`. Kokkos then picks one GPU per rank
  (`--kokkos-map-device-id-by=mpi_rank`).
- `MPICH_GPU_SUPPORT_ENABLED=1` is set by `setupprlm-gpu.sh`.
- `tests/kokkos_smoke.exe` checks the whole toolchain: a DMDA on Kokkos
  vectors, a device kernel, and a periodic ghost exchange. It prints PASS
  and `vec=mpikokkos exec=Cuda`. It passed on 1 GPU and on 4 GPUs
  (2026-09-24).
