#ifndef KOKKOSTYPES_H
#define KOKKOSTYPES_H

// Kokkos views of the model's DMDA vectors (GPU build only). A 3D DMDA vector
// with dof is viewed as (k, j, i, c) with c fastest (LayoutRight), which is
// the same memory layout as G_node: c = 0..3 is f, 4..6 is A, 7..9 is V.
#ifdef SFGPU_KOKKOS
#include <Kokkos_Core.hpp>
#include <petscdmda_kokkos.hpp>

using KokkosMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
using NodeView = Kokkos::Experimental::OffsetView<PetscScalar ****,
                                                  Kokkos::LayoutRight,
                                                  KokkosMemSpace>;
using ConstNodeView =
    Kokkos::Experimental::OffsetView<const PetscScalar ****,
                                     Kokkos::LayoutRight, KokkosMemSpace>;

// Loop over the owned sites (k, j, i) of a rank. Iterate::Right makes i
// (contiguous in memory) vary fastest across threads, so neighboring threads
// read neighboring sites; the CUDA default iterates k fastest.
using SitePolicy = Kokkos::MDRangePolicy<
    Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>,
    Kokkos::IndexType<PetscInt>>;

// True if v is a Kokkos vector (-dm_vec_type kokkos), i.e. lives on the GPU
inline bool is_kokkos_vec(Vec v) {
  PetscBool flg = PETSC_FALSE;
  PetscObjectTypeCompareAny((PetscObject)v, &flg, VECSEQKOKKOS, VECMPIKOKKOS,
                            "");
  return flg;
}
#endif

#endif
